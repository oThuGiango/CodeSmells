from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class UnixCoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, graph_batch=None) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)
