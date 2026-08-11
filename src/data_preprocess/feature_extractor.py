"""
V-JEPA 2/2.1 feature extractors for manipulation videos.

Loads the pretrained V-JEPA encoder (target_encoder from EMA)
and extracts per-clip features from video tensors.
"""

import sys
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

# Add vjepa2 repo (third_party submodule) to path so we can import its modules
# File: skill2policy/src/data_preprocess/feature_extractor.py
# vjepa2 lives at skill2policy/third_party/vjepa2
VJEPA2_ROOT = Path(__file__).resolve().parent.parent.parent / "third_party" / "vjepa2"
if str(VJEPA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA2_ROOT))

import src.models.vision_transformer as video_vit
from src.utils.wrappers import MultiSeqWrapper


VJEPA2_BACKBONE = "vjepa2_vitl256"
VJEPA21_BACKBONE = "vjepa2_1_vitl384"
SUPPORTED_BACKBONES = (VJEPA2_BACKBONE, VJEPA21_BACKBONE)


def _clean_target_encoder_state_dict(state_dict):
    """Map the official DDP checkpoint onto ``MultiSeqWrapper`` exactly.

    The released V-JEPA 2 checkpoint may retain a positional-embedding entry
    even though the published ViT-L configuration uses RoPE.  Meta's hub
    loader therefore uses ``strict=False``.  We keep the stronger contract:
    remove only that known-unused entry, then require every remaining model
    parameter to match strictly.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint target_encoder must be a state-dict mapping")

    cleaned = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint state-dict keys must be strings")
        new_key = key.removeprefix("module.")
        if new_key in cleaned:
            raise ValueError(f"duplicate checkpoint key after DDP cleanup: {new_key}")
        cleaned[new_key] = value

    # Official hub inference ignores this legacy key because the encoder uses
    # rotary position embeddings.  Cover both the wrapped and bare spellings
    # so a clear strict-load error is reserved for genuinely incompatible
    # checkpoints.
    cleaned.pop("backbone.pos_embed", None)
    cleaned.pop("pos_embed", None)
    return cleaned


def _clean_vjepa21_encoder_state_dict(state_dict):
    """Map the released 2.1 ``ema_encoder`` onto the bare encoder."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint ema_encoder must be a state-dict mapping")
    cleaned = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint state-dict keys must be strings")
        new_key = key.removeprefix("module.").removeprefix("backbone.")
        if new_key in cleaned:
            raise ValueError(f"duplicate checkpoint key after DDP cleanup: {new_key}")
        cleaned[new_key] = value
    return cleaned


def build_vjepa_encoder(
    model_name="vit_large",
    crop_size=256,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    uniform_power=True,
    use_sdpa=True,
    use_rope=True,
    use_silu=False,
    wide_silu=True,
    use_activation_checkpointing=False,
):
    """Build V-JEPA encoder matching the pretrain config."""
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    encoder = MultiSeqWrapper(encoder)
    return encoder


def load_pretrained_encoder(checkpoint_path, device="cuda", **model_kwargs):
    """
    Load the target_encoder (EMA) weights from a V-JEPA pretrain checkpoint.

    The checkpoint stores DDP keys like 'module.backbone.blocks.0...',
    which map directly to MultiSeqWrapper(VisionTransformer) structure.

    Returns the encoder in eval mode on the specified device.
    """
    encoder = build_vjepa_encoder(**model_kwargs)

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    # Use target_encoder (EMA) for better features
    if not isinstance(ckpt, Mapping) or "target_encoder" not in ckpt:
        raise ValueError(
            "not a raw V-JEPA 2 pretraining checkpoint: "
            "missing target_encoder"
        )
    cleaned = _clean_target_encoder_state_dict(ckpt["target_encoder"])

    msg = encoder.load_state_dict(cleaned, strict=True)
    print(f"Loaded target_encoder from {checkpoint_path} (epoch {ckpt.get('epoch', '?')}): {msg}")

    del ckpt
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def build_vjepa21_encoder(
    crop_size=384,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
):
    """Build the official V-JEPA 2.1 ViT-L/16 video encoder."""

    if crop_size != 384 or patch_size != 16 or tubelet_size != 2:
        raise ValueError(
            "V-JEPA 2.1 ViT-L uses crop_size=384, patch_size=16, "
            "and tubelet_size=2"
        )
    from app.vjepa_2_1.models import vision_transformer as vjepa21_vit

    return vjepa21_vit.vit_large(
        img_size=(crop_size, crop_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )


def load_pretrained_vjepa21_encoder(
    checkpoint_path,
    device="cuda",
    **model_kwargs,
):
    """Load the official V-JEPA 2.1 ViT-L distributed-student EMA encoder."""

    encoder = build_vjepa21_encoder(**model_kwargs)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping) or "ema_encoder" not in checkpoint:
        raise ValueError(
            "not an official V-JEPA 2.1 ViT-L checkpoint: "
            "missing ema_encoder"
        )
    state_dict = _clean_vjepa21_encoder_state_dict(checkpoint["ema_encoder"])
    message = encoder.load_state_dict(state_dict, strict=True)
    print(
        f"Loaded V-JEPA 2.1 ema_encoder from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')}): {message}"
    )
    del checkpoint
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


class VJEPAFeatureExtractor(nn.Module):
    """
    Wraps a pretrained V-JEPA encoder for feature extraction.

    Input:  [B, C, T, H, W] video tensor (C=3, T=16, H=W=256)
    Output: [B, D] pooled feature vector (D=1024 for ViT-L)

    Pooling strategies:
        - 'mean': mean over all patch tokens  (default)
        - 'cls':  CLS token (if available)
        - 'none': return all tokens [B, N, D]
    """

    def __init__(self, encoder, pool="mean"):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.embed_dim = encoder.backbone.embed_dim

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] normalized video tensor
        Returns:
            features: [B, D] if pool='mean', [B, N, D] if pool='none'
        """
        # MultiSeqWrapper expects a list of clips, returns list of outputs
        # For feature extraction, no masks needed
        out = self.encoder([x], masks=None)  # list of 1 element
        tokens = out[0]  # [B, N, D] where N = (T/t) * (H/p) * (W/p)

        if self.pool == "mean":
            return tokens.mean(dim=1)  # [B, D]
        elif self.pool == "none":
            return tokens  # [B, N, D]
        else:
            raise ValueError(f"Unknown pool mode: {self.pool}")


class VJEPA21FeatureExtractor(nn.Module):
    """Pool features from the bare official V-JEPA 2.1 video encoder."""

    def __init__(self, encoder, pool="mean"):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.embed_dim = encoder.embed_dim

    @torch.no_grad()
    def forward(self, x):
        tokens = self.encoder(x)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError(
                "V-JEPA 2.1 encoder must return a [B,N,D] token tensor"
            )
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "none":
            return tokens
        raise ValueError(f"Unknown pool mode: {self.pool}")
