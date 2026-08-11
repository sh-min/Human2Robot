"""Learned spatial pooling over frozen dense V-JEPA patch features."""

import math

import torch
import torch.nn as nn

from skill_classifier.models.registry import register_model


@register_model("spatial_attention_mlp")
class SpatialAttentionMLP(nn.Module):
    """Content attention over patches followed by temporal mean and an MLP."""

    def __init__(
        self,
        vjepa_dim=1024,
        hand_dim=0,
        window_size=8,
        num_classes=6,
        hidden_dims=(256, 128),
        dropout=0.3,
    ):
        super().__init__()
        if hand_dim != 0:
            raise ValueError("spatial_attention_mlp currently supports V-JEPA-only input")
        self.vjepa_dim = int(vjepa_dim)
        self.hand_dim = int(hand_dim)
        self.window_size = int(window_size)
        self.spatial_norm = nn.LayerNorm(self.vjepa_dim)
        self.spatial_query = nn.Parameter(
            torch.randn(self.vjepa_dim) / math.sqrt(self.vjepa_dim)
        )
        layers = []
        input_dim = self.vjepa_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def pool_dense(self, vjepa):
        if vjepa.ndim != 4 or vjepa.shape[-1] != self.vjepa_dim:
            raise ValueError(
                "spatial_attention_mlp expects [B,W,S,D] V-JEPA features, "
                f"got {tuple(vjepa.shape)}"
            )
        values = vjepa.float()
        normalized = self.spatial_norm(values)
        scores = torch.einsum(
            "bwsd,d->bws", normalized, self.spatial_query
        ) / math.sqrt(self.vjepa_dim)
        weights = torch.softmax(scores, dim=-1)
        spatial = torch.sum(values * weights.unsqueeze(-1), dim=2)
        return spatial.mean(dim=1), weights

    def representation(self, vjepa):
        pooled, weights = self.pool_dense(vjepa)
        return self.net[:-1](pooled), weights

    def forward(self, vjepa, hand):
        pooled, _ = self.pool_dense(vjepa)
        return self.net(pooled)
