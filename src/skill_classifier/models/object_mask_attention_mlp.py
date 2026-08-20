"""Object-grounded pooling over frozen dense V-JEPA patch features."""

import math

import torch
import torch.nn as nn

from skill_classifier.models.registry import register_model


@register_model("object_mask_attention_mlp")
class ObjectMaskAttentionMLP(nn.Module):
    """Fuse global V-JEPA attention with VLM+SAM object-mask pooling.

    ``context`` uses the existing second model input to preserve the training
    loop API.  Its last dimension contains flattened patch occupancies for K
    canonical objects followed by K grounding confidence values.
    """

    def __init__(
        self,
        vjepa_dim=1024,
        hand_dim=0,
        window_size=8,
        num_classes=6,
        hidden_dims=(256, 128),
        dropout=0.4,
        object_prompt_count=0,
        object_mask_spatial_tokens=0,
        object_projection_dim=64,
        use_global_features=True,
        use_object_features=True,
        use_confidence_features=True,
        use_occupancy_features=True,
        use_confidence_gate=True,
    ):
        super().__init__()
        self.vjepa_dim = int(vjepa_dim)
        self.hand_dim = int(hand_dim)
        self.window_size = int(window_size)
        self.object_prompt_count = int(object_prompt_count)
        self.object_mask_spatial_tokens = int(object_mask_spatial_tokens)
        self.object_projection_dim = int(object_projection_dim)
        self.use_global_features = bool(use_global_features)
        self.use_object_features = bool(use_object_features)
        self.use_confidence_features = bool(use_confidence_features)
        self.use_occupancy_features = bool(use_occupancy_features)
        self.use_confidence_gate = bool(use_confidence_gate)
        expected_context = self.object_prompt_count * (
            self.object_mask_spatial_tokens + 1
        )
        if self.object_prompt_count <= 0 or self.object_mask_spatial_tokens <= 0:
            raise ValueError("object prompt and spatial-token counts must be positive")
        if self.hand_dim != expected_context:
            raise ValueError(
                f"object context dimension {self.hand_dim} != {expected_context}"
            )

        if not any(
            (
                self.use_global_features,
                self.use_object_features,
                self.use_confidence_features,
                self.use_occupancy_features,
            )
        ):
            raise ValueError("at least one object-fusion input must be enabled")

        if self.use_global_features:
            self.global_norm = nn.LayerNorm(self.vjepa_dim)
            self.global_query = nn.Parameter(
                torch.randn(self.vjepa_dim) / math.sqrt(self.vjepa_dim)
            )
        if self.use_object_features:
            self.object_norm = nn.LayerNorm(self.vjepa_dim)
            self.object_projection = nn.Sequential(
                nn.Linear(self.vjepa_dim, self.object_projection_dim),
                nn.GELU(),
            )
            self.object_embeddings = nn.Parameter(
                torch.randn(
                    self.object_prompt_count, self.object_projection_dim
                ) * 0.02
            )

        input_dim = 0
        if self.use_global_features:
            input_dim += self.vjepa_dim
        if self.use_object_features:
            input_dim += self.object_prompt_count * self.object_projection_dim
        if self.use_confidence_features:
            input_dim += self.object_prompt_count
        if self.use_occupancy_features:
            input_dim += self.object_prompt_count
        layers = []
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

    def split_context(self, context):
        if context.ndim != 3 or context.shape[-1] != self.hand_dim:
            raise ValueError(
                "object_mask_attention_mlp expects context [B,W,C], "
                f"got {tuple(context.shape)}"
            )
        mask_size = self.object_prompt_count * self.object_mask_spatial_tokens
        masks = context[..., :mask_size].reshape(
            *context.shape[:2],
            self.object_prompt_count,
            self.object_mask_spatial_tokens,
        )
        confidence = context[..., mask_size:].reshape(
            *context.shape[:2], self.object_prompt_count
        )
        return masks.float().clamp(0.0, 1.0), confidence.float().clamp(0.0, 1.0)

    def pool_dense(self, vjepa, context):
        if (
            vjepa.ndim != 4
            or vjepa.shape[-2] != self.object_mask_spatial_tokens
            or vjepa.shape[-1] != self.vjepa_dim
        ):
            raise ValueError(
                "object_mask_attention_mlp expects V-JEPA [B,W,S,D], "
                f"got {tuple(vjepa.shape)}"
            )
        values = vjepa.float()
        masks, confidence = self.split_context(context)

        occupancy = masks.mean(dim=-1)
        fused_parts = []
        global_weights = None
        if self.use_global_features:
            normalized = self.global_norm(values)
            scores = torch.einsum(
                "bwsd,d->bws", normalized, self.global_query
            ) / math.sqrt(self.vjepa_dim)
            global_weights = torch.softmax(scores, dim=-1)
            global_per_time = torch.sum(
                values * global_weights.unsqueeze(-1), dim=2
            )
            fused_parts.append(global_per_time.mean(dim=1))

        if self.use_object_features:
            denominator = masks.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            object_per_time = torch.einsum("bwks,bwsd->bwkd", masks, values)
            object_per_time = object_per_time / denominator
            presence = (occupancy > 0).to(confidence.dtype)
            gate = confidence * presence if self.use_confidence_gate else presence
            projected = self.object_projection(self.object_norm(object_per_time))
            projected = (
                projected + self.object_embeddings[None, None]
            ) * gate[..., None]
            fused_parts.append(projected.mean(dim=1).flatten(1))
        if self.use_confidence_features:
            fused_parts.append(confidence.mean(dim=1))
        if self.use_occupancy_features:
            fused_parts.append(occupancy.mean(dim=1))
        fused = torch.cat(fused_parts, dim=-1)
        diagnostics = {
            "global_attention": global_weights,
            "object_masks": masks,
            "object_confidence": confidence,
            "object_occupancy": occupancy,
        }
        return fused, diagnostics

    def representation(self, vjepa, context):
        fused, diagnostics = self.pool_dense(vjepa, context)
        return self.net[:-1](fused), diagnostics

    def forward(self, vjepa, hand):
        fused, _ = self.pool_dense(vjepa, hand)
        return self.net(fused)
