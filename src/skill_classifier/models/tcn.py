"""Temporal Convolutional Network for skill classification."""

import torch
import torch.nn as nn

from skill_classifier.models.registry import register_model


class CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        # x: [B, C, T]
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Residual block: 2 causal convolutions + skip connection."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return out + residual


@register_model("tcn")
class SkillTCN(nn.Module):
    """
    Temporal Convolutional Network for MANO-only skill classification.

    Input:  hand [B, W, D]
    Output: logits [B, num_classes]
    """

    def __init__(
        self,
        hand_dim=288,
        window_size=8,
        num_classes=12,
        hidden_channels=180,
        num_blocks=5,
        kernel_size=3,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Conv1d(hand_dim, hidden_channels, 1)

        # Temporal blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(
                TemporalBlock(hidden_channels, kernel_size, dilation, dropout)
            )

        # Classification head
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, hand):
        # hand: [B, W, D]
        x = hand.transpose(1, 2)       # [B, D, W]
        x = self.input_proj(x)         # [B, H, W]
        for block in self.blocks:
            x = block(x)               # [B, H, W]
        x = x.mean(dim=2)              # [B, H]  global avg pool
        return self.head(x)            # [B, num_classes]
