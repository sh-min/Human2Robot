"""Repo-relative paths for the retargeting module.

Importable by sibling scripts so they all agree on where things live without
hard-coding absolute paths.
"""
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

# Vendored upstream
DEX_RETARGETING_DIR = os.path.join(REPO_ROOT, "third_party", "dex-retargeting")

# Local xhand assets (URDF + meshes) and configs
ASSETS_DIR = os.path.join(THIS_DIR, "assets")
URDF_ROOT = ASSETS_DIR                           # parent that contains "xhand/" subdir
CONFIG_DIR = os.path.join(THIS_DIR, "configs")

# Convenience yml paths
YML_RIGHT = os.path.join(CONFIG_DIR, "xhand_right_dexpilot.yml")
YML_LEFT = os.path.join(CONFIG_DIR, "xhand_left_dexpilot.yml")


def load_R_mano_xhand():
    """Loaded R_MANO_XHAND dict (right/left), built once by
    compute_R_mano_xhand.py via Procrustes alignment of MCP knuckles.

    Falls back to a hand-picked 90° rotation if the .npy is missing.
    """
    import numpy as np
    fallback = {
        "right": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64),
        "left":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    }
    out = {}
    for h in ("right", "left"):
        p = os.path.join(ASSETS_DIR, f"R_mano_xhand_{h}.npy")
        out[h] = np.load(p).astype(np.float64) if os.path.exists(p) else fallback[h]
    return out


R_MANO_XHAND = load_R_mano_xhand()
