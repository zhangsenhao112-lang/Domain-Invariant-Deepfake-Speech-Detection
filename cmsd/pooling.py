"""Pooling layers used by HLA and the utterance-level classifier."""

import torch
from torch import nn
import torch.nn.functional as F


class ClassicAttention(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.context = nn.Parameter(torch.randn(embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.projection(inputs)
        scores = torch.tanh(projected @ self.context)
        return F.softmax(scores, dim=1)


class AttentiveStatisticsPooling(nn.Module):
    """Attentive mean/variance pooling used in the original experiment code."""

    def __init__(self, embedding_dim: int, input_dim: int, proj: bool = False):
        super().__init__()
        self.attention = ClassicAttention(input_dim, embedding_dim)
        self.projection = (
            nn.Linear(embedding_dim * 2, embedding_dim) if proj else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.attention(inputs).unsqueeze(-1)
        weighted = inputs * weights
        mean = weighted.mean(dim=1)
        variance = (inputs * weighted).sum(dim=1) - mean.square()
        return self.projection(torch.cat((mean, variance), dim=-1))


class HierarchicalLayerAggregation(nn.Module):
    """Aggregate all SSL layers with sample-adaptive channel weights (HLA)."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.pool = AttentiveStatisticsPooling(
            embedding_dim=num_layers, input_dim=num_layers, proj=True
        )

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        if not layers:
            raise ValueError("HLA needs at least one hidden layer")
        stacked = torch.stack(layers, dim=1)  # [B, L, T, D]
        descriptors = stacked.mean(dim=2).transpose(1, 2)  # [B, D, L]
        weights = torch.sigmoid(self.pool(descriptors))[:, :, None, None]
        return (stacked * weights).sum(dim=1)
