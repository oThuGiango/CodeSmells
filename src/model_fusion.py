from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import AutoModel


class FusionClassifier(nn.Module):
    def __init__(
        self,
        unixcoder_name: str,
        node_feature_dim: int,
        gat_hidden_dim: int = 128,
        gat_heads: int = 4,
        gat_layers: int = 2,
        edge_feature_dim: int = 6,
        num_labels: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.semantic_encoder = AutoModel.from_pretrained(unixcoder_name)
        self.dropout = dropout

        convs = []
        current_dim = node_feature_dim
        for _ in range(gat_layers):
            convs.append(GATConv(current_dim, gat_hidden_dim, heads=gat_heads, dropout=dropout, edge_dim=edge_feature_dim))
            current_dim = gat_hidden_dim * gat_heads
        self.gat_convs = nn.ModuleList(convs)

        fusion_dim = self.semantic_encoder.config.hidden_size + current_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_labels),
        )

    def encode_code(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.semantic_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def encode_graph(self, graph_batch) -> torch.Tensor:
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = getattr(graph_batch, "edge_attr", None)
        batch = graph_batch.batch
        for conv in self.gat_convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, graph_batch) -> torch.Tensor:
        semantic_embedding = self.encode_code(input_ids, attention_mask)
        structural_embedding = self.encode_graph(graph_batch)
        fusion_embedding = torch.cat([semantic_embedding, structural_embedding], dim=-1)
        return self.classifier(fusion_embedding)
