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

# Upstream dex-retargeting ships URDFs (via its own `assets` submodule, which
# must be initialised: `git -C third_party/dex-retargeting submodule update
# --init assets`) and configs for its supported hands.
DEX_URDF_ROOT = os.path.join(DEX_RETARGETING_DIR, "assets", "robots", "hands")
DEX_CONFIG_DIR = os.path.join(
    DEX_RETARGETING_DIR, "src", "dex_retargeting", "configs", "teleop"
)

# Convenience yml paths
YML_RIGHT = os.path.join(CONFIG_DIR, "xhand_right_dexpilot.yml")
YML_LEFT = os.path.join(CONFIG_DIR, "xhand_left_dexpilot.yml")


# ---- Hand embodiments -------------------------------------------------
# Right and left are chosen independently (--right_embodiment/--left_embodiment),
# so nothing here may assume both hands share a model.
#
# forearm_sign: which way the wrist frame's z-axis points along the forearm.
#   +1 = +z runs wrist->elbow, -1 = +z runs wrist->fingertips.
#   xhand's fingers sit at -z, inspire's at +z, so the two are opposite.
#   Used by the renderer to place the forearm; getting it wrong points the
#   arm out through the fingers.
#
# R_mano is always the uncentred Procrustes fit. That rotation is dominated by
# aligning the wrist->fingers axis, which is what DexPilot's wrist->fingertip
# vectors care about; fitting on centred knuckles instead aligns the flatter
# knuckle-spread pattern and measurably degrades retargeting (inspire: DexPilot
# last distance 0.092 centred vs 0.028 uncentred).
#
# use_offset: whether to also fit a wrist translation. A rotation-only fit
# silently assumes the URDF root sits where the MANO wrist does. That is roughly
# true for xhand (~15 mm), which is how it was calibrated — leave it off so its
# output stays bit-identical. It is badly false for inspire, whose `base` is the
# arm mount flange ~51 mm further out; without the offset its knuckles land
# 51.6 mm from the human's (8.1 mm with it).
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
    "inspire": dict(
        # URDF+meshes come from the dex-retargeting assets submodule, but the
        # config is a local copy — upstream's scaling_factor leaves the fingers
        # curled on MANO input (see the note at the top of the yml).
        urdf_root=DEX_URDF_ROOT,
        urdf_rel="inspire_hand/inspire_hand_{hand}.urdf",
        config_dir=CONFIG_DIR,
        config_name="inspire_hand_{hand}_dexpilot.yml",
        r_mano="R_mano_inspire_{hand}.npy",
        wrist_offset="wrist_offset_inspire_{hand}.npy",
        use_offset=True,
        forearm_sign=-1,
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
