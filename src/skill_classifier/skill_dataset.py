"""
Windowed dataset for skill classification from per-recording bundled features.

Each recording has a `features.pt` (built by preprocess.py) with:
    vjepa_orig:        [T, 1024]
    vjepa_orig_masked: [T, 1024]   (optional)
    mano:              [T, 96]
    labels_per_token:  [T]         int per-token skill label, or -1 if no GT

The dataset enumerates token positions across recordings and yields windows.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


VARIANT_VJEPA_KEY = {
    "mano_only": None,
    "vjepa_orig": "vjepa_orig",
    "masked_vjepa_orig": "vjepa_orig_masked",
    "vjepa_robot": "vjepa_robot",
    "vjepa_orig_dense": "vjepa_orig_dense",
}
HAND_REPRESENTATIONS = ("none", "axis_angle", "rot6d")


def axis_angle_to_rotation_6d(axis_angle):
    """Convert ``(..., 3)`` rotation vectors to continuous 6-D rotations.

    The output concatenates the first and second columns of the Rodrigues
    rotation matrix.  Unlike raw axis-angle, this representation does not
    jump between opposite vectors at the +/-pi branch boundary.
    """

    value = torch.as_tensor(axis_angle)
    if not value.is_floating_point() or value.shape[-1] != 3:
        raise ValueError(
            "axis_angle must be a floating tensor with final dimension 3"
        )
    theta = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    axis = value / theta.clamp_min(torch.finfo(value.dtype).eps)
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(*value.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    identity = identity.expand(*value.shape[:-1], 3, 3)
    sin_theta = torch.sin(theta)[..., None]
    one_minus_cos = (1.0 - torch.cos(theta))[..., None]
    matrix = identity + sin_theta * skew + one_minus_cos * (skew @ skew)
    return matrix[..., :, :2].transpose(-1, -2).reshape(*value.shape[:-1], 6)


def mano_axis_angle_to_rotation_6d(mano, validity):
    """Convert ``(T,96)`` two-hand MANO rotations and re-mask invalid hands."""

    mano = torch.as_tensor(mano)
    validity = torch.as_tensor(validity)
    if mano.ndim != 2 or mano.shape[1] != 96 or not mano.is_floating_point():
        raise ValueError(
            f"rot6d requires floating MANO axis-angle shape (T,96), got {tuple(mano.shape)}"
        )
    if validity.shape != (len(mano), 2) or validity.dtype != torch.bool:
        raise ValueError(
            "rot6d requires mano_valid_per_token with bool shape (T,2)"
        )
    rotations = mano.reshape(len(mano), 2, 16, 3)
    rotation_6d = axis_angle_to_rotation_6d(rotations)
    rotation_6d = rotation_6d * validity[..., None, None].to(rotation_6d.dtype)
    return rotation_6d.reshape(len(mano), 192)


def sampling_signature(recording):
    """Return the temporal/spatial contract that classifier weights inherit."""

    return (
        recording.get("sampling_profile", "legacy_dense"),
        float(recording.get("sample_fps", recording.get("source_fps", 0.0))),
        float(recording.get("token_rate_hz", 0.0)),
        int(recording.get("clip_frames", 16)),
        int(recording.get("tubelet_size", 2)),
        recording.get("spatial_profile", "legacy_stretch"),
    )


def load_recordings(data_root, recording_glob="*"):
    """Load all recording bundles under {data_root}/{recording_glob}/features.pt."""
    patterns = [p.strip() for p in recording_glob.split(",") if p.strip()]
    paths = sorted({
        path
        for pattern in patterns
        for path in Path(data_root).glob(f"{pattern}/features.pt")
    })
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


class SkillWindowDataset(Dataset):
    """
    Sliding-window dataset over per-recording bundled features.

    Sample = (rec_idx, t, label) where t is a token index inside that recording
    and the label comes from per-token GT. Tokens with no GT (-1) are skipped.

    Window of `window_size` ending at token t is built on-the-fly. Tokens
    earlier than `window_size - 1` are zero-padded at the start.
    """

    def __init__(
        self,
        recordings,
        window_size=8,
        variant="mano_only",
        vjepa_diff=False,
        hand_representation="axis_angle",
    ):
        """
        Args:
            recordings:  list of dicts from load_recordings()
            window_size: number of past tokens to use as context
            variant:     'mano_only' | 'vjepa_orig' | 'masked_vjepa_orig'
            vjepa_diff:  if True, replace V-JEPA with vjepa[t]-vjepa[t-1] (per-recording)
            hand_representation: ``none`` (V-JEPA only), ``axis_angle``
                (legacy), or continuous ``rot6d``
        """
        if variant not in VARIANT_VJEPA_KEY:
            raise ValueError(f"Unknown variant: {variant}")
        if hand_representation not in HAND_REPRESENTATIONS:
            raise ValueError(
                f"Unknown hand representation: {hand_representation}"
            )
        self.window_size = window_size
        self.variant = variant
        self.vjepa_diff = vjepa_diff
        self.hand_representation = hand_representation
        vjepa_key = VARIANT_VJEPA_KEY[variant]

        self.recordings = []   # list of dicts with vjepa, hand tensors
        self.samples = []      # list of (rec_idx, t, label)
        signatures = set()

        for rec in recordings:
            mano = rec["mano"]
            if mano.ndim != 2 or mano.shape[0] <= 0:
                raise ValueError(f"invalid MANO feature shape: {tuple(mano.shape)}")
            if hand_representation == "none":
                mano = mano.new_zeros((len(mano), 0))
            elif hand_representation == "rot6d":
                if "mano_valid_per_token" not in rec:
                    raise ValueError(
                        "rot6d requires mano_valid_per_token in every recording"
                    )
                mano = mano_axis_angle_to_rotation_6d(
                    mano, rec["mano_valid_per_token"]
                )
            T = mano.shape[0]
            labels = rec["labels_per_token"]
            if labels.ndim != 1 or len(labels) != T:
                raise ValueError(
                    f"labels/MANO length mismatch: {tuple(labels.shape)} vs {T}"
                )

            if vjepa_key is None:
                vjepa = torch.zeros(T, 0)
            else:
                if vjepa_key not in rec:
                    raise ValueError(f"recording missing requested feature: {vjepa_key}")
                vjepa = rec[vjepa_key]
                if vjepa.ndim not in (2, 3) or len(vjepa) != T:
                    raise ValueError(
                        f"{vjepa_key}/MANO length mismatch: "
                        f"{tuple(vjepa.shape)} vs {T}"
                    )
                if vjepa_diff:
                    diff = torch.zeros_like(vjepa)
                    diff[1:] = vjepa[1:] - vjepa[:-1]
                    vjepa = diff

            self.recordings.append({"vjepa": vjepa, "hand": mano})
            signatures.add(sampling_signature(rec))

            for t in range(T):
                lb = int(labels[t])
                if lb >= 0:
                    self.samples.append((len(self.recordings) - 1, t, lb))

        if not self.recordings:
            raise ValueError("no recordings supplied")
        if len(signatures) != 1:
            raise ValueError(
                "mixed feature sampling contracts are not allowed: "
                f"{sorted(signatures, key=repr)}"
            )
        self.sampling_signature = next(iter(signatures))

        # Dim auto-detect for downstream model build
        self.hand_dim = self.recordings[0]["hand"].shape[1]
        self.vjepa_dim = self.recordings[0]["vjepa"].shape[-1]
        self.vjepa_spatial_tokens = (
            self.recordings[0]["vjepa"].shape[1]
            if self.recordings[0]["vjepa"].ndim == 3
            else 1
        )
        expected_vjepa_shape = self.recordings[0]["vjepa"].shape[1:]
        if any(
            recording["vjepa"].shape[1:] != expected_vjepa_shape
            for recording in self.recordings
        ):
            raise ValueError("recordings use mixed V-JEPA feature shapes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec_idx, t, label = self.samples[idx]
        rec = self.recordings[rec_idx]
        W = self.window_size
        vjepa = rec["vjepa"]
        hand = rec["hand"]
        D_v = vjepa.shape[1:]
        D_h = hand.shape[1]

        start = t - W + 1
        if start >= 0:
            vjepa_win = vjepa[start:t + 1]
            hand_win = hand[start:t + 1]
        else:
            pad_len = -start
            vjepa_win = torch.cat(
                [vjepa.new_zeros((pad_len, *D_v)), vjepa[:t + 1]], dim=0
            )
            hand_win = torch.cat(
                [hand.new_zeros((pad_len, D_h)), hand[:t + 1]], dim=0
            )

        return vjepa_win, hand_win, label
