"""Object-grounded V-JEPA classifier aligned to frozen action text prototypes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from skill_classifier.models.object_mask_attention_mlp import ObjectMaskAttentionMLP
from skill_classifier.models.registry import register_model


@register_model("object_text_prototype_mlp")
class ObjectTextPrototypeMLP(ObjectMaskAttentionMLP):
    """Classify against all frozen CLIP action descriptions simultaneously."""

    def __init__(
        self,
        *args,
        num_classes=6,
        hidden_dims=(256, 128),
        text_embedding_dim=512,
        text_head_mode="prototype",
        **kwargs,
    ):
        if not hidden_dims:
            raise ValueError("text prototype model requires hidden_dims")
        if text_head_mode not in ("prototype", "hybrid"):
            raise ValueError("text_head_mode must be prototype or hybrid")
        super().__init__(
            *args,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            **kwargs,
        )
        self.num_classes = int(num_classes)
        self.text_embedding_dim = int(text_embedding_dim)
        self.text_head_mode = text_head_mode
        # Replace the conventional learned class layer with text similarity.
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        feature_dim = int(hidden_dims[-1])
        self.video_to_text = nn.Linear(feature_dim, self.text_embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.register_buffer(
            "action_text_embeddings",
            torch.zeros(self.num_classes, self.text_embedding_dim),
        )
        if self.text_head_mode == "hybrid":
            self.learned_classifier = nn.Linear(feature_dim, self.num_classes)
            self.text_logit_weight = nn.Parameter(torch.tensor(0.0))

    def set_action_text_embeddings(self, embeddings: torch.Tensor) -> None:
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if embeddings.shape != self.action_text_embeddings.shape:
            raise ValueError(
                "action text embeddings must have shape "
                f"{tuple(self.action_text_embeddings.shape)}, got "
                f"{tuple(embeddings.shape)}"
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("action text embeddings contain non-finite values")
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        self.action_text_embeddings.copy_(
            embeddings.to(self.action_text_embeddings.device)
        )

    def forward(self, vjepa, hand):
        fused, _ = self.pool_dense(vjepa, hand)
        features = self.net(fused)
        video_embedding = nn.functional.normalize(
            self.video_to_text(features), dim=-1
        )
        prototypes = nn.functional.normalize(
            self.action_text_embeddings, dim=-1
        )
        text_logits = (
            self.logit_scale.exp().clamp(max=100.0)
            * video_embedding
            @ prototypes.transpose(0, 1)
        )
        if self.text_head_mode == "prototype":
            return text_logits
        weight = torch.sigmoid(self.text_logit_weight)
        return self.learned_classifier(features) + weight * text_logits
