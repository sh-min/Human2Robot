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
}


def load_recordings(data_root, recording_glob="*"):
    """Load all recording bundles under {data_root}/{recording_glob}/features.pt."""
    paths = sorted(Path(data_root).glob(f"{recording_glob}/features.pt"))
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


class SkillWindowDataset(Dataset):
    """
    Sliding-window dataset over per-recording bundled features.

    Sample = (rec_idx, t, label) where t is a token index inside that recording
    and the label comes from per-token GT. Tokens with no GT (-1) are skipped.

    Window of `window_size` ending at token t is built on-the-fly. Tokens
    earlier than `window_size - 1` are zero-padded at the start.
    """

    def __init__(self, recordings, window_size=8, variant="mano_only", vjepa_diff=False):
        """
        Args:
            recordings:  list of dicts from load_recordings()
            window_size: number of past tokens to use as context
            variant:     'mano_only' | 'vjepa_orig' | 'masked_vjepa_orig'
            vjepa_diff:  if True, replace V-JEPA with vjepa[t]-vjepa[t-1] (per-recording)
        """
        if variant not in VARIANT_VJEPA_KEY:
            raise ValueError(f"Unknown variant: {variant}")
        self.window_size = window_size
        self.variant = variant
        self.vjepa_diff = vjepa_diff
        vjepa_key = VARIANT_VJEPA_KEY[variant]

        self.recordings = []   # list of dicts with vjepa, hand tensors
        self.samples = []      # list of (rec_idx, t, label)

        for rec in recordings:
            mano = rec["mano"]
            T = mano.shape[0]

            if vjepa_key is None:
                vjepa = torch.zeros(T, 0)
            else:
                vjepa = rec[vjepa_key]
                if vjepa_diff:
                    diff = torch.zeros_like(vjepa)
                    diff[1:] = vjepa[1:] - vjepa[:-1]
                    vjepa = diff

            self.recordings.append({"vjepa": vjepa, "hand": mano})

            labels = rec["labels_per_token"]
            for t in range(T):
                lb = int(labels[t])
                if lb >= 0:
                    self.samples.append((len(self.recordings) - 1, t, lb))

        # Dim auto-detect for downstream model build
        self.hand_dim = self.recordings[0]["hand"].shape[1]
        self.vjepa_dim = self.recordings[0]["vjepa"].shape[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec_idx, t, label = self.samples[idx]
        rec = self.recordings[rec_idx]
        W = self.window_size
        vjepa = rec["vjepa"]
        hand = rec["hand"]
        D_v = vjepa.shape[1]
        D_h = hand.shape[1]

        start = t - W + 1
        if start >= 0:
            vjepa_win = vjepa[start:t + 1]
            hand_win = hand[start:t + 1]
        else:
            pad_len = -start
            vjepa_win = torch.cat([torch.zeros(pad_len, D_v), vjepa[:t + 1]], dim=0)
            hand_win = torch.cat([torch.zeros(pad_len, D_h), hand[:t + 1]], dim=0)

        return vjepa_win, hand_win, label
