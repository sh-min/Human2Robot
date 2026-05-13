"""Repo-relative paths and import shims for the inpainting module.

Vendored deps:
    third_party/sam2     — SAM2 image+video segmentation
    third_party/E2FGVI   — flow-guided video inpainting

xhand assets (URDF, meshes, R_MANO_XHAND) come from src/retargeting/.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

# Vendored upstream
SAM2_DIR    = os.path.join(REPO_ROOT, "third_party", "sam2")
E2FGVI_DIR  = os.path.join(REPO_ROOT, "third_party", "E2FGVI")

# Model checkpoints (download instructions in README)
SAM2_CHECKPOINT  = os.path.join(SAM2_DIR, "checkpoints", "sam2_hiera_large.pt")
SAM2_CONFIG_NAME = "sam2_hiera_l.yaml"   # SAM2 looks this up via its hydra config
E2FGVI_CHECKPOINT = os.path.join(E2FGVI_DIR, "release_model", "E2FGVI-HQ-CVPR22.pth")

# xhand assets are owned by src/retargeting/ — reuse them.
RETARGET_DIR = os.path.abspath(os.path.join(REPO_ROOT, "src", "retargeting"))
XHAND_URDF_ROOT = os.path.join(RETARGET_DIR, "assets")
XHAND_URDF_RIGHT = os.path.join(XHAND_URDF_ROOT, "xhand", "xhand_right.urdf")
XHAND_URDF_LEFT  = os.path.join(XHAND_URDF_ROOT, "xhand", "xhand_left.urdf")


def ensure_sam2_importable() -> None:
    """Make `import sam2` work without installing the submodule.

    SAM2 ships as a pip-installable package, but we keep skill2policy
    install-free by editing sys.path. The submodule layout is
    `third_party/sam2/sam2/` (an inner directory named `sam2/`).
    """
    if SAM2_DIR not in sys.path:
        sys.path.insert(0, SAM2_DIR)


def ensure_e2fgvi_importable() -> None:
    """Make `from model.e2fgvi_hq import InpaintGenerator` work.

    Upstream E2FGVI has `model/` and `core/` at the repo root (not under an
    inner package directory), so we add the repo root to sys.path. This
    pollutes the namespace with top-level `model` / `core` modules — keep
    inpaint_hands.py's imports tight.
    """
    if E2FGVI_DIR not in sys.path:
        sys.path.insert(0, E2FGVI_DIR)


def load_R_mano_xhand():
    """Procrustes-fit rotation aligning xhand root frame with MANO wrist frame.

    Built once by `src/retargeting/compute_R_mano_xhand.py`. Falls back to a
    hand-picked 90° rotation if the .npy isn't present. Mirrors
    `src/retargeting/_paths.py:load_R_mano_xhand`.
    """
    import numpy as np
    fallback = {
        "right": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64),
        "left":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    }
    out = {}
    for h in ("right", "left"):
        p = os.path.join(XHAND_URDF_ROOT, f"R_mano_xhand_{h}.npy")
        out[h] = np.load(p).astype(np.float64) if os.path.exists(p) else fallback[h]
    return out


R_MANO_XHAND = load_R_mano_xhand()
