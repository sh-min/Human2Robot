"""Temporal Transformer for skill classification."""

import torch
import torch.nn as nn

from skill_classifier.models.registry import register_model


@register_model("transformer")
class SkillTransformer(nn.Module):
    """
    Small temporal transformer over the feature window.

    Input:  vjepa [B, W, D_v], hand [B, W, D_h]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        vjepa_dim=1024,
        hand_dim=126,
        window_size=8,
        num_classes=12,
        d_model=256,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.vjepa_dim = vjepa_dim
        self.hand_dim = hand_dim
        self.input_proj = nn.Linear(vjepa_dim + hand_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, vjepa, hand):
        if self.vjepa_dim > 0 and self.hand_dim > 0:
            x = torch.cat([vjepa, hand], dim=-1)
        elif self.hand_dim > 0:
            x = hand
        else:
            x = vjepa
        x = self.input_proj(x) + self.pos_embed
        # Use math SDPA backend to avoid NaN on some GPU/PyTorch combos
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            x = self.encoder(x)
        x = x[:, -1]  # take last token (most recent time step)
        return self.head(x)
