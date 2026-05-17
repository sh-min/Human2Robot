"""Repo-relative paths and import shims for the inpainting module.

Vendored deps:
    third_party/sam2               — SAM2 image+video segmentation
    third_party/E2FGVI             — flow-guided video inpainting
    third_party/Depth-Anything-V2  — monocular metric/relative depth (depth-aware path)

xhand assets (URDF, meshes, R_MANO_XHAND) come from src/retargeting/.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

# Vendored upstream
SAM2_DIR           = os.path.join(REPO_ROOT, "third_party", "sam2")
E2FGVI_DIR         = os.path.join(REPO_ROOT, "third_party", "E2FGVI")
DEPTH_ANYTHING_DIR = os.path.join(REPO_ROOT, "third_party", "Depth-Anything-V2")
DIFFUSION_VAS_DIR  = os.path.join(REPO_ROOT, "third_party", "diffusion-vas")

# Model checkpoints (download instructions in README)
SAM2_CHECKPOINT  = os.path.join(SAM2_DIR, "checkpoints", "sam2_hiera_large.pt")
SAM2_CONFIG_NAME = "sam2_hiera_l.yaml"   # SAM2 looks this up via its hydra config
E2FGVI_CHECKPOINT = os.path.join(E2FGVI_DIR, "release_model", "E2FGVI-HQ-CVPR22.pth")

# Diffusion-VAS checkpoints (~7.6 GB each); live under /ckpt per global CLAUDE.md.
DIFFUSION_VAS_CKPT_DIR  = "/ckpt/diffusion_vas"
DIFFUSION_VAS_MASK_CKPT = os.path.join(DIFFUSION_VAS_CKPT_DIR, "diffusion-vas-amodal-segmentation")
DIFFUSION_VAS_RGB_CKPT  = os.path.join(DIFFUSION_VAS_CKPT_DIR, "diffusion-vas-content-completion")

# Depth Anything V2 checkpoints live under /ckpt per the global CLAUDE.md convention.
DEPTH_ANYTHING_CKPT_DIR = "/ckpt/depth_anything"
DEPTH_ANYTHING_CKPTS = {
    "vits": os.path.join(DEPTH_ANYTHING_CKPT_DIR, "depth_anything_v2_vits.pth"),
    "vitb": os.path.join(DEPTH_ANYTHING_CKPT_DIR, "depth_anything_v2_vitb.pth"),
    "vitl": os.path.join(DEPTH_ANYTHING_CKPT_DIR, "depth_anything_v2_vitl.pth"),
}
DEPTH_ANYTHING_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

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


def ensure_diffusion_vas_importable() -> None:
    """Make `from models.diffusion_vas.pipeline_diffusion_vas import ...` work.

    Upstream diffusion-vas keeps `models/` (which bundles its own copy of
    Depth-Anything-V2) at the repo root, so we add the submodule root to
    sys.path. Like E2FGVI this pollutes the namespace with a top-level
    `models` package — only import it inside the diffusion_vas conda env
    (amodal_cube.py), never alongside the inpaint-env stages.
    """
    if DIFFUSION_VAS_DIR not in sys.path:
        sys.path.insert(0, DIFFUSION_VAS_DIR)


def ensure_depth_anything_importable() -> None:
    """Make `from depth_anything_v2.dpt import DepthAnythingV2` work.

    Upstream layout is `third_party/Depth-Anything-V2/depth_anything_v2/`,
    so adding the submodule root to sys.path is enough.
    """
    if DEPTH_ANYTHING_DIR not in sys.path:
        sys.path.insert(0, DEPTH_ANYTHING_DIR)


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
