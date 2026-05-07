"""HaWoR npz + HACO contact -> xhand qpos sequence (DexPilot retargeting).

Same as retarget_from_npz.py but fingertip joints for contacting fingers are
replaced by the centroid of their HACO contact vertices before retargeting.

Usage:
    conda activate RFM_retarget
    python retarget_from_npz_contact.py \
        --npz /path/to/rgb_hawor/retarget_input.npz \
        --contact_dir /path/to/contact
"""
import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from dex_retargeting.retargeting_config import RetargetingConfig

from _paths import URDF_ROOT, CONFIG_DIR

R_MANO_XHAND = {
    "right": np.array([[0,  0, 1],
                       [0, -1, 0],
                       [1,  0, 0]], dtype=np.float32),
    "left":  np.array([[0, 0, -1],
                       [0, 1,  0],
                       [1, 0,  0]], dtype=np.float32),
}

# MANO 21-keypoint -> finger index (0=thumb..4=pinky, 5=palm/wrist)
_KPT_TO_FINGER = {0: 5}
for _f, _start in enumerate([1, 5, 9, 13, 17]):
    for _k in range(_start, _start + 4):
        _KPT_TO_FINGER[_k] = _f

# Fingertip keypoint index per finger: thumb=4, index=8, middle=12, ring=16, pinky=20
_FINGERTIP_KPT = [4, 8, 12, 16, 20]


def _vertex_finger_labels(joints_world_t, verts_world_t):
    dists = np.linalg.norm(verts_world_t[:, None] - joints_world_t[None, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    return np.array([_KPT_TO_FINGER[k] for k in nearest], dtype=np.int32)


def retarget_one_hand(data, hand, out_dir, contact_dir):
    hand_idx = 0 if hand == "left" else 1
    joints_world = data[f"joints_{hand}"].astype(np.float32)   # (T, 21, 3)
    verts_world  = data[f"verts_{hand}"].astype(np.float32)    # (T, 778, 3)
    root_orient  = data["mano_global_orient"][hand_idx].astype(np.float32)
    valid        = data["valid"][hand_idx]
    start_idx    = int(data["start_idx"])
    T = joints_world.shape[0]

    R_root = Rscipy.from_rotvec(root_orient).as_matrix().astype(np.float32)  # (T,3,3)

    rel_joints = joints_world - joints_world[:, 0:1, :]
    rel_verts  = verts_world  - joints_world[:, 0:1, :]
    joints_canon = np.einsum("tji,tnj->tni", R_root, rel_joints)  # (T, 21, 3)
    verts_canon  = np.einsum("tji,tnj->tni", R_root, rel_verts)   # (T, 778, 3)

    R = R_MANO_XHAND[hand]
    joints_mp = joints_canon @ R   # (T, 21, 3)
    verts_mp  = verts_canon  @ R   # (T, 778, 3)

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

        jp = joints_mp[t].copy()  # (21, 3)

        frame_key = f"rgb_frame{start_idx + t:05d}"
        contact_path = os.path.join(contact_dir, f"{frame_key}.npz")
        if os.path.exists(contact_path):
            contact_data = np.load(contact_path)
            mask_key = f"{hand}_contact_mask"
            if mask_key in contact_data:
                contact_mask = contact_data[mask_key]
                if contact_mask.any():
                    labels = _vertex_finger_labels(joints_world[t], verts_world[t])
                    for finger_idx, tip_kpt in enumerate(_FINGERTIP_KPT):
                        contacting = contact_mask & (labels == finger_idx)
                        if contacting.any():
                            jp[tip_kpt] = verts_mp[t][contacting].mean(axis=0)

        ref = jp[indices[1]] - jp[indices[0]]
        qpos = retargeting.retarget(ref.astype(np.float32))
        qpos_seq[t] = qpos
        last = qpos
        n_valid += 1

    out_path = os.path.join(out_dir, f"qpos_xhand_contact_{hand}.pkl")
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
    ap.add_argument("--contact_dir", default=None,
                    help="Directory of per-frame contact .npz files "
                         "(default: <npz parent>/contact)")
    ap.add_argument("--hand", default="both", choices=["right", "left", "both"])
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    RetargetingConfig.set_default_urdf_dir(URDF_ROOT)
    data = np.load(args.npz)
    npz_dir = os.path.dirname(os.path.abspath(args.npz))
    out_dir = args.out_dir or npz_dir
    contact_dir = args.contact_dir or os.path.join(os.path.dirname(npz_dir), "contact")
    os.makedirs(out_dir, exist_ok=True)

    hands = ["right", "left"] if args.hand == "both" else [args.hand]
    for h in hands:
        retarget_one_hand(data, h, out_dir, contact_dir)


if __name__ == "__main__":
    main()
