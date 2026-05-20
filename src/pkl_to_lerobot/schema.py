"""Action/state field definitions for LeRobot dataset conversion.

Defines how retargeted xHand pkl data maps to the flat vectors expected by
LeRobot (observation.state / action) and generates the GR00T-compatible
modality.json metadata.

The canonical state/action vector layout (bimanual, 38-D):

    Index  Field
    -----  -----
     0:12  right_hand_joint   (12 finger DoFs)
    12:15  right_wrist_pos    (xyz in camera frame)
    15:19  right_wrist_quat   (xyzw quaternion)
    19:31  left_hand_joint    (12 finger DoFs)
    31:34  left_wrist_pos
    34:38  left_wrist_quat

For single-hand episodes the unused side is zero-filled and a
``hand_mask`` (2,) boolean is stored alongside for downstream filtering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

HANDS = ("right", "left")

FINGER_DOF = 12
WRIST_POS_DIM = 3
WRIST_QUAT_DIM = 4
PER_HAND_DIM = FINGER_DOF + WRIST_POS_DIM + WRIST_QUAT_DIM  # 19
BIMANUAL_DIM = PER_HAND_DIM * 2  # 38


@dataclass
class FieldSlice:
    name: str
    start: int
    end: int

    @property
    def dim(self) -> int:
        return self.end - self.start


def _hand_fields(side: str, offset: int) -> list[FieldSlice]:
    """Field slices for one hand starting at ``offset``."""
    o = offset
    return [
        FieldSlice(f"{side}_hand_joint", o, o + FINGER_DOF),
        FieldSlice(f"{side}_wrist_pos", o + FINGER_DOF, o + FINGER_DOF + WRIST_POS_DIM),
        FieldSlice(
            f"{side}_wrist_quat",
            o + FINGER_DOF + WRIST_POS_DIM,
            o + FINGER_DOF + WRIST_POS_DIM + WRIST_QUAT_DIM,
        ),
    ]


STATE_FIELDS = _hand_fields("right", 0) + _hand_fields("left", PER_HAND_DIM)
ACTION_FIELDS = list(STATE_FIELDS)  # same layout for action


@dataclass
class EpisodeSchema:
    """Describes which hands are present and the fps of the episode."""

    hands: tuple[str, ...] = ("right", "left")
    fps: float = 30.0
    action_mode: Literal["absolute", "delta"] = "absolute"

    @property
    def state_dim(self) -> int:
        return BIMANUAL_DIM

    @property
    def action_dim(self) -> int:
        return BIMANUAL_DIM


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert (T, 3, 3) rotation matrices to (T, 4) quaternions (xyzw)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat().astype(np.float32)  # (T, 4) xyzw


def final_pose_to_state_action(
    final_pose: dict,
    *,
    action_mode: Literal["absolute", "delta"] = "absolute",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert final_pose.pkl to aligned (T, 38) state and action arrays.

    Parameters
    ----------
    final_pose : dict loaded from final_pose.pkl
        Top-level keys: "right", "left", "T".
        Each hand dict has: "qpos" (T, 12), "wrist_pos" (T, 3),
        "wrist_rot" (T, 3, 3), "valid" (T,), "joint_names" (list).

    action_mode : "absolute" or "delta"
        "absolute" -> action[t] = state[t+1]
        "delta"    -> action[t] = state[t+1] - state[t]

    Returns
    -------
    states : (T, 38) float32
    actions : (T, 38) float32  (last frame's action repeats the final state)
    valid : (T,) bool  (AND of both hands' valid masks)
    """
    T = int(final_pose["T"])
    states = np.zeros((T, BIMANUAL_DIM), dtype=np.float32)
    valid = np.ones(T, dtype=bool)

    for side in HANDS:
        if side not in final_pose:
            continue
        p = final_pose[side]
        offset = 0 if side == "right" else PER_HAND_DIM

        qpos = np.asarray(p["qpos"], dtype=np.float32)
        wrist_pos = np.asarray(p["wrist_pos"], dtype=np.float32)
        wrist_rot = np.asarray(p["wrist_rot"], dtype=np.float32)
        wrist_quat = _rotmat_to_quat(wrist_rot)

        states[:, offset : offset + FINGER_DOF] = qpos
        states[:, offset + FINGER_DOF : offset + FINGER_DOF + WRIST_POS_DIM] = wrist_pos
        states[:, offset + FINGER_DOF + WRIST_POS_DIM : offset + PER_HAND_DIM] = wrist_quat
        valid &= np.asarray(p["valid"]).astype(bool)

    if action_mode == "absolute":
        actions = np.empty_like(states)
        actions[:-1] = states[1:]
        actions[-1] = states[-1]
    else:
        actions = np.zeros_like(states)
        actions[:-1] = states[1:] - states[:-1]

    return states, actions, valid


def pkl_to_state_action(
    pkls: dict[str, dict],
    *,
    action_mode: Literal["absolute", "delta"] = "absolute",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-hand qpos pkl dicts to aligned (T, 38) state and action arrays.

    This is the fallback for older pkl files that lack wrist data.
    Only finger qpos is filled; wrist fields remain zero.

    Parameters
    ----------
    pkls : {"right": pkl_dict, "left": pkl_dict}
        Each pkl_dict has keys ``data`` (T, 12), ``valid`` (T,).
        Optionally ``wrist_pos`` (T, 3) and ``wrist_quat`` (T, 4).
    action_mode : "absolute" or "delta"

    Returns
    -------
    states : (T, 38) float32
    actions : (T, 38) float32
    valid : (T,) bool
    """
    lengths = [v["data"].shape[0] for v in pkls.values()]
    T = lengths[0]
    assert all(l == T for l in lengths), f"Hand lengths mismatch: {lengths}"

    states = np.zeros((T, BIMANUAL_DIM), dtype=np.float32)
    valid = np.ones(T, dtype=bool)

    for side in HANDS:
        if side not in pkls:
            continue
        p = pkls[side]
        offset = 0 if side == "right" else PER_HAND_DIM
        states[:, offset : offset + FINGER_DOF] = p["data"]
        if "wrist_pos" in p:
            states[:, offset + FINGER_DOF : offset + FINGER_DOF + WRIST_POS_DIM] = p["wrist_pos"]
        if "wrist_quat" in p:
            states[:, offset + FINGER_DOF + WRIST_POS_DIM : offset + PER_HAND_DIM] = p[
                "wrist_quat"
            ]
        valid &= np.asarray(p["valid"]).astype(bool)

    if action_mode == "absolute":
        actions = np.empty_like(states)
        actions[:-1] = states[1:]
        actions[-1] = states[-1]
    else:
        actions = np.zeros_like(states)
        actions[:-1] = states[1:] - states[:-1]

    return states, actions, valid


def build_modality_json(
    image_keys: list[str] | None = None,
) -> dict:
    """Build a GR00T-compatible modality.json dict.

    Parameters
    ----------
    image_keys : list of video observation keys.  Defaults to
        ``["observation.images.head_cam"]``.
    """
    if image_keys is None:
        image_keys = ["observation.images.head_cam"]

    state_spec = {f.name: {"start": f.start, "end": f.end} for f in STATE_FIELDS}
    action_spec = {f.name: {"start": f.start, "end": f.end} for f in ACTION_FIELDS}
    video_spec = {k: {"original_key": k} for k in image_keys}

    return {
        "state": state_spec,
        "action": action_spec,
        "video": video_spec,
    }


def write_modality_json(out_dir: str | Path, **kwargs) -> Path:
    """Write modality.json into ``out_dir/meta/``."""
    meta = Path(out_dir) / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    path = meta / "modality.json"
    path.write_text(json.dumps(build_modality_json(**kwargs), indent=2) + "\n")
    return path
