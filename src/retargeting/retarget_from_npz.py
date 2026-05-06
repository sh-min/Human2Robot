"""HaWoR npz -> xhand qpos sequence (DexPilot retargeting).

Usage:
    conda activate RFM_retarget
    python retarget_from_npz.py \
        --npz /path/to/<seq>_hawor/retarget_input.npz
"""
import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from dex_retargeting.retargeting_config import RetargetingConfig

from _paths import URDF_ROOT, CONFIG_DIR

# MANO canonical wrist frame -> xhand wrist link frame, hand-specific.
#   right: MANO (x, y, z) -> xhand_right (z, -y, x)
#   left : MANO (x, y, z) -> xhand_left  (-z,  y, x)
R_MANO_XHAND = {
    "right": np.array([[0,  0, 1],
                       [0, -1, 0],
                       [1,  0, 0]], dtype=np.float32),
    "left":  np.array([[0, 0, -1],
                       [0, 1,  0],
                       [1, 0,  0]], dtype=np.float32),
}


def retarget_one_hand(data, hand, out_dir):
    hand_idx = 0 if hand == "left" else 1
    joints_world = data[f"joints_{hand}"].astype(np.float32)
    root_orient = data["mano_global_orient"][hand_idx].astype(np.float32)
    valid = data["valid"][hand_idx]
    T = joints_world.shape[0]

    # Joints in MANO canonical frame (wrist-relative + R_root inverted).
    R_root = Rscipy.from_rotvec(root_orient).as_matrix().astype(np.float32)
    rel = joints_world - joints_world[:, 0:1, :]
    joints_canon = np.einsum("tji,tnj->tni", R_root, rel)

    # MANO canonical -> xhand wrist link frame.
    R = R_MANO_XHAND[hand]
    joints_mp = joints_canon @ R

    cfg_path = os.path.join(CONFIG_DIR, f"xhand_{hand}_dexpilot.yml")
    retargeting = RetargetingConfig.load_from_file(cfg_path).build()
    joint_names = retargeting.joint_names
    dof = len(joint_names)
    indices = retargeting.optimizer.target_link_human_indices  # (2, N) for DexPilot

    qpos_seq = np.zeros((T, dof), dtype=np.float32)
    last = None
    n_valid = 0
    for t in range(T):
        if not valid[t]:
            if last is not None:
                qpos_seq[t] = last
            continue
        ref = joints_mp[t, indices[1]] - joints_mp[t, indices[0]]
        qpos = retargeting.retarget(ref.astype(np.float32))
        qpos_seq[t] = qpos
        last = qpos
        n_valid += 1

    out_path = os.path.join(out_dir, f"qpos_xhand_{hand}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(
            dict(
                data=qpos_seq,
                valid=valid,
                joint_names=joint_names,
                config_path=cfg_path,
                hand=hand,
                dof=dof,
            ),
            f,
        )
    print(f"[{hand}] -> {out_path}  shape={qpos_seq.shape}  "
          f"valid={n_valid}/{T}  dof={dof}")
    retargeting.verbose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hand", default="both", choices=["right", "left", "both"])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    RetargetingConfig.set_default_urdf_dir(URDF_ROOT)
    data = np.load(args.npz)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)

    hands = ["right", "left"] if args.hand == "both" else [args.hand]
    for h in hands:
        retarget_one_hand(data, h, out_dir)


if __name__ == "__main__":
    main()
