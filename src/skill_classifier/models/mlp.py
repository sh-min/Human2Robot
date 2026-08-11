"""MLP baseline for skill classification."""

import torch
import torch.nn as nn

from skill_classifier.models.registry import register_model


@register_model("mlp")
class SkillMLP(nn.Module):
    """
    Concatenates V-JEPA + hand features over the window, then MLP.

    Input:  vjepa [B, W, D_v], hand [B, W, D_h]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        vjepa_dim=1024,
        hand_dim=126,
        window_size=8,
        num_classes=12,
        hidden_dims=(512, 256),
        dropout=0.3,
        pool="mean",
    ):
        super().__init__()
        self.pool = pool
        self.vjepa_dim = vjepa_dim
        self.hand_dim = hand_dim

        feat_dim = vjepa_dim + hand_dim
        if pool == "concat":
            in_dim = feat_dim * window_size
        else:  # mean
            in_dim = feat_dim

        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, vjepa, hand):
        if self.vjepa_dim > 0 and self.hand_dim > 0:
            x = torch.cat([vjepa, hand], dim=-1)
        elif self.hand_dim > 0:
            x = hand
        else:
            x = vjepa
        if self.pool == "concat":
            x = x.flatten(1)
        else:
            x = x.mean(dim=1)
        return self.net(x)
