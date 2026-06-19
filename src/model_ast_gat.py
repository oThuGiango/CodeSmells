from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class ASTGATClassifier(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        edge_feature_dim: int = 7,
        num_labels: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        convs = []
        current_dim = node_feature_dim
        for _ in range(layers):
            convs.append(GATConv(current_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_feature_dim))
            current_dim = hidden_dim * heads
        self.convs = nn.ModuleList(convs)
        self.classifier = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(current_dim // 2, num_labels),
        )

    def encode_graph(self, graph_batch) -> torch.Tensor:
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = getattr(graph_batch, "edge_attr", None)
        batch = graph_batch.batch
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, input_ids=None, attention_mask=None, graph_batch=None) -> torch.Tensor:
        graph_embedding = self.encode_graph(graph_batch)
        return self.classifier(graph_embedding)
