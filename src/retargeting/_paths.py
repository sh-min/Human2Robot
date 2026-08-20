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


# The policy and renderer are intentionally single-embodiment: XHand only.
# A stale or foreign embodiment name must fail instead of changing the hand.
EMBODIMENTS = {
    "xhand": dict(
        urdf_root=ASSETS_DIR,
        urdf_rel="xhand/xhand_{hand}.urdf",
        config_dir=CONFIG_DIR,
        config_name="xhand_{hand}_dexpilot.yml",
        r_mano="R_mano_xhand_{hand}.npy",
        wrist_offset="wrist_offset_xhand_{hand}.npy",
        use_offset=False,
        forearm_sign=+1,
    ),
}
EMBODIMENT_NAMES = sorted(EMBODIMENTS)
DEFAULT_EMBODIMENT = "xhand"


def _spec(embodiment: str) -> dict:
    try:
        return EMBODIMENTS[embodiment]
    except KeyError:
        raise ValueError(
            f"unknown embodiment {embodiment!r}; choose from {EMBODIMENT_NAMES}"
        ) from None


def urdf_root(embodiment: str) -> str:
    """Dir that the config's relative `urdf_path` resolves against.

    Differs per embodiment, so `RetargetingConfig.set_default_urdf_dir` must be
    called per hand rather than once at startup.
    """
    return _spec(embodiment)["urdf_root"]


def urdf_path(embodiment: str, hand: str) -> str:
    s = _spec(embodiment)
    return os.path.join(s["urdf_root"], s["urdf_rel"].format(hand=hand))


def config_path(embodiment: str, hand: str) -> str:
    s = _spec(embodiment)
    return os.path.join(s["config_dir"], s["config_name"].format(hand=hand))


def forearm_sign(embodiment: str) -> int:
    return _spec(embodiment)["forearm_sign"]


def uses_offset(embodiment: str) -> bool:
    return _spec(embodiment)["use_offset"]


def load_wrist_offset(embodiment: str, hand: str):
    """Translation (3,) in the robot wrist frame that moves the hand root so
    its knuckles land on MANO's, applied as `wrist_pos + R_cam @ offset`.

    Zero when no .npy exists, which is the case for "raw"-fit embodiments —
    so xhand behaves exactly as it did before offsets existed.
    """
    import numpy as np
    p = os.path.join(
        ASSETS_DIR, _spec(embodiment)["wrist_offset"].format(hand=hand)
    )
    return np.load(p).astype(np.float64) if os.path.exists(p) else np.zeros(3)


def load_R_mano(embodiment: str, hand: str):
    """Rotation taking a vector in MANO canonical wrist frame to the robot
    hand's wrist link frame, Procrustes-fit by compute_R_mano.py.

    Falls back to a hand-picked 90° rotation if the .npy is missing. The
    fallback was eyeballed for xhand, so it is wrong for any other embodiment
    — a missing .npy shows up as a subtly misoriented hand, not an error.
    """
    import numpy as np
    fallback = {
        "right": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64),
        "left":  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    }
    p = os.path.join(ASSETS_DIR, _spec(embodiment)["r_mano"].format(hand=hand))
    return np.load(p).astype(np.float64) if os.path.exists(p) else fallback[hand]


def load_R_mano_xhand():
    """Back-compat: {right,left} rotations for xhand only."""
    return {h: load_R_mano("xhand", h) for h in ("right", "left")}


R_MANO_XHAND = load_R_mano_xhand()
