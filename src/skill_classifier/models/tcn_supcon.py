"""TCN over V-JEPA + Euler MANO window features with SupCon projection head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from skill_classifier.models.registry import register_model
from skill_classifier.models.tcn import TemporalBlock


@register_model("tcn_supcon")
class SkillTCNSupCon(nn.Module):
    """
    Temporal Convolutional Network over a window of (V-JEPA, Euler-MANO)
    features, with a classifier head and an L2-normalized projection head
    for supervised contrastive learning.

    Input signature matches SkillMLPSupCon so all downstream callers
    (pseudo-label script, ClassifierFeatureSkillConditioning) work without
    branching on model type.

    forward(vjepa, hand, return_projection=False)
        vjepa: [B, W, vjepa_dim]
        hand:  [B, W, hand_dim]
    """

    def __init__(
        self,
        vjepa_dim=1024,
        hand_dim=96,
        window_size=8,
        num_classes=12,
        hidden_channels=256,
        num_blocks=3,
        kernel_size=3,
        dropout=0.2,
        proj_dim=128,
        vjepa_proj_dim=None,
        **_,
    ):
        super().__init__()
        # Optional linear projection to compress V-JEPA features before fusion.
        # If vjepa_proj_dim is set, reduces 1024-d -> vjepa_proj_dim before concat.
        if vjepa_proj_dim is not None and vjepa_proj_dim < vjepa_dim:
            self.vjepa_proj = nn.Linear(vjepa_dim, vjepa_proj_dim)
            fused_dim = vjepa_proj_dim + hand_dim
        else:
            self.vjepa_proj = None
            fused_dim = vjepa_dim + hand_dim
        in_dim = fused_dim
        self.input_proj = nn.Conv1d(in_dim, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            TemporalBlock(hidden_channels, kernel_size,
                          dilation=2 ** i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.classifier_head = nn.Linear(hidden_channels, num_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, proj_dim),
        )

    def trunk(self, vjepa, hand):
        if self.vjepa_proj is not None:
            vjepa = self.vjepa_proj(vjepa)         # [B, W, vjepa_proj_dim]
        x = torch.cat([vjepa, hand], dim=-1)       # [B, W, D]
        x = x.transpose(1, 2)                      # [B, D, W]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=2)                       # [B, hidden_channels]

    def forward(self, vjepa, hand, return_projection=False):
        h = self.trunk(vjepa, hand)
        logits = self.classifier_head(h)
        if not return_projection:
            return logits
        return logits, F.normalize(self.projection_head(h), dim=-1)
