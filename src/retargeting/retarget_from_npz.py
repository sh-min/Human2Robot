"""HaWoR npz -> xhand qpos sequence (DexPilot retargeting).

Two modes:
  1) default:        pure DexPilot vector retargeting (wrist + 5 fingertip vectors)
  2) --contact:      stage-1 DexPilot, then stage-2 refinement that nudges qpos
                     so each xhand fingertip is closer to the centroid of the
                     human MANO contact verts on that finger (palmar + fingertip
                     filtered). Uses pinocchio FK + scipy.optimize.

Usage:
    conda activate RFM_retarget
    python retarget_from_npz.py --npz /path/to/<seq>_hawor/retarget_input.npz
    python retarget_from_npz.py --npz <...> --contact \
        --contact_dir /path/to/<seq>/contact
"""
import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

from dex_retargeting.retargeting_config import RetargetingConfig

from _paths import URDF_ROOT, CONFIG_DIR, R_MANO_XHAND

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# MANO canonical wrist frame -> xhand wrist link frame.
# Loaded from assets/R_mano_xhand_{right,left}.npy (Procrustes-fit). Cast to
# float32 to match the rest of the script's dtype.
R_MANO_XHAND = {h: R.astype(np.float32) for h, R in R_MANO_XHAND.items()}

# Map MANO joint index (skinning-weight argmax, 1..15) -> xhand finger index.
# xhand fingers ordered as in TIP_LINK_NAMES below.
_MANO_JOINT_TO_FINGER = {
    1: 1, 2: 1, 3: 1,        # index
    4: 2, 5: 2, 6: 2,        # middle
    7: 4, 8: 4, 9: 4,        # pinky
    10: 3, 11: 3, 12: 3,     # ring
    13: 0, 14: 0, 15: 0,     # thumb
}

def _tip_link_names(hand):
    """Ordered to match _MANO_JOINT_TO_FINGER values: [thumb, index, mid, ring, pinky]."""
    return [
        f"{hand}_hand_thumb_rota_tip",   # 0 thumb
        f"{hand}_hand_index_rota_tip",   # 1 index
        f"{hand}_hand_mid_tip",          # 2 middle
        f"{hand}_hand_ring_tip",         # 3 ring
        f"{hand}_hand_pinky_tip",        # 4 pinky
    ]


def _build_contact_refiner(hand, joint_names):
    """Stage-2 refiner: normal-aware vertex matching between human contact
    verts and xhand fingertip mesh verts.

    For each human contact vertex h (with outward normal n_h), among the
    xhand fingertip mesh verts r whose outward normal n_r satisfies
    n_h · n_r > normal_thr, take the closest r and minimize ||h - r||². This
    keeps xhand's palmar surface (whichever verts happen to face the same way
    as the human contact normal) close to the contact points, without
    pre-baked palmar masks on the robot side.

    refine_fn(qpos_init, human_by_finger, alpha, normal_thr) -> refined qpos.
    human_by_finger[f] = dict(verts=(Nf,3), normals=(Nf,3)) in xhand wrist frame.
    """
    import pinocchio as pin
    import trimesh
    from scipy.optimize import minimize

    urdf_path = os.path.join(URDF_ROOT, "xhand", f"xhand_{hand}.urdf")
    model = pin.buildModelFromUrdf(urdf_path)
    data_p = model.createData()

    pin_names = [model.names[i] for i in range(1, len(model.names))]
    dex_to_pin = np.array([pin_names.index(n) for n in joint_names], dtype=int)
    pin_to_dex = np.array([joint_names.index(n) for n in pin_names], dtype=int)

    q_lo = model.lowerPositionLimit.copy()[dex_to_pin]
    q_hi = model.upperPositionLimit.copy()[dex_to_pin]

    tip_link_names = _tip_link_names(hand)
    tip_fids = [model.getFrameId(n) for n in tip_link_names]

    # Load tip mesh verts + vertex normals (link-local frame, once).
    mesh_dir = os.path.join(URDF_ROOT, "xhand", "meshes")
    tip_local = []                # list of (verts (M,3), normals (M,3))
    for ln in tip_link_names:
        stl = os.path.join(mesh_dir, f"{ln}.STL")
        m = trimesh.load(stl, force="mesh")
        v = np.asarray(m.vertices, dtype=np.float64)
        n = np.asarray(m.vertex_normals, dtype=np.float64)
        tip_local.append((v, n))

    def refine(q_init_dex, human_by_finger, alpha=0.001, normal_thr=0.3):
        if not human_by_finger:
            return q_init_dex

        fingers = sorted(human_by_finger.keys())
        humans_v = [human_by_finger[f]["verts"].astype(np.float64) for f in fingers]
        humans_n = [human_by_finger[f]["normals"].astype(np.float64) for f in fingers]
        fids_a = [tip_fids[f] for f in fingers]
        locals_a = [tip_local[f] for f in fingers]

        def loss(q_dex):
            q_pin = q_dex[pin_to_dex]
            pin.forwardKinematics(model, data_p, q_pin)
            pin.updateFramePlacements(model, data_p)
            cost = 0.0
            for h_v, h_n, fid, (v_loc, n_loc) in zip(humans_v, humans_n, fids_a, locals_a):
                R = np.asarray(data_p.oMf[fid].rotation)
                t = np.asarray(data_p.oMf[fid].translation)
                r_v = v_loc @ R.T + t                  # (M,3) in wrist frame
                r_n = n_loc @ R.T                       # (M,3) in wrist frame
                d_pos = np.sum((h_v[:, None, :] - r_v[None, :, :]) ** 2, axis=-1)
                nd = h_n @ r_n.T                        # (Nh, M)
                compat = nd > normal_thr
                d_masked = np.where(compat, d_pos, np.inf)
                min_d = d_masked.min(axis=1)
                has = compat.any(axis=1)
                if has.any():
                    cost += float(min_d[has].mean())
            anchor = alpha * float(np.sum((q_dex - q_init_dex) ** 2))
            return float(cost + anchor)

        bounds = list(zip(q_lo.tolist(), q_hi.tolist()))
        res = minimize(loss, q_init_dex.astype(np.float64), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 60, "ftol": 1e-7})
        return res.x.astype(np.float32)

    return refine


def _load_masks():
    out = {}
    for h in ("right", "left"):
        out[h] = dict(
            palmar=np.load(os.path.join(ASSETS_DIR, f"palmar_mask_{h}.npy")).astype(bool),
            tip=np.load(os.path.join(ASSETS_DIR, f"fingertip_mask_{h}.npy")).astype(bool),
            part=np.load(os.path.join(ASSETS_DIR, f"finger_part_{h}.npy")).astype(int),
        )
    return out


def retarget_one_hand(data, hand, out_dir, *, contact_dir=None, alpha=0.001,
                       normal_thr=0.3):
    hand_idx = 0 if hand == "left" else 1
    joints_world = data[f"joints_{hand}"].astype(np.float32)
    root_orient = data["mano_global_orient"][hand_idx].astype(np.float32)
    valid = data["valid"][hand_idx]
    start_idx_data = int(data["start_idx"])
    T = joints_world.shape[0]

    R_root = Rscipy.from_rotvec(root_orient).as_matrix().astype(np.float32)
    rel = joints_world - joints_world[:, 0:1, :]
    joints_canon = np.einsum("tji,tnj->tni", R_root, rel)
    R = R_MANO_XHAND[hand]
    joints_mp = joints_canon @ R

    use_contact = contact_dir is not None
    if use_contact:
        import trimesh
        verts_world = data[f"verts_{hand}"].astype(np.float32)
        rel_v = verts_world - joints_world[:, 0:1, :]
        verts_mp = np.einsum("tji,tnj->tni", R_root, rel_v) @ R
        masks = _load_masks()[hand]   # palmar, tip, part
        mano_faces = np.load(os.path.join(ASSETS_DIR, f"mano_faces_{hand}.npy")).astype(np.int32)

    cfg_path = os.path.join(CONFIG_DIR, f"xhand_{hand}_dexpilot.yml")
    retargeting = RetargetingConfig.load_from_file(cfg_path).build()
    joint_names = retargeting.joint_names
    dof = len(joint_names)
    indices = retargeting.optimizer.target_link_human_indices

    refine_fn = _build_contact_refiner(hand, joint_names) if use_contact else None

    qpos_seq = np.zeros((T, dof), dtype=np.float32)
    last = None
    n_valid = 0
    n_refined = 0
    for t in range(T):
        if not valid[t]:
            if last is not None:
                qpos_seq[t] = last
            continue
        # Stage 1: DexPilot vector retargeting
        ref = joints_mp[t, indices[1]] - joints_mp[t, indices[0]]
        qpos = retargeting.retarget(ref.astype(np.float32))

        # Stage 2: normal-aware Chamfer refinement
        if use_contact:
            cpath = os.path.join(
                contact_dir, f"rgb_frame{start_idx_data + t:05d}.npz"
            )
            human_by_finger = {}
            if os.path.exists(cpath):
                cd = np.load(cpath)
                mk = f"{hand}_contact_mask"
                if mk in cd.files:
                    cm = cd[mk].astype(bool) & masks["palmar"] & masks["tip"]
                    if cm.any():
                        # MANO mesh vertex normals (in xhand wrist frame).
                        mesh_t = trimesh.Trimesh(
                            verts_mp[t].astype(np.float64), mano_faces,
                            process=False,
                        )
                        mano_normals = np.asarray(mesh_t.vertex_normals)
                        for v_idx in np.where(cm)[0]:
                            f_idx = _MANO_JOINT_TO_FINGER.get(int(masks["part"][v_idx]))
                            if f_idx is None:
                                continue
                            slot = human_by_finger.setdefault(
                                f_idx, {"verts": [], "normals": []}
                            )
                            slot["verts"].append(verts_mp[t, v_idx])
                            slot["normals"].append(mano_normals[v_idx])
                        human_by_finger = {
                            f: {"verts": np.asarray(s["verts"]),
                                "normals": np.asarray(s["normals"])}
                            for f, s in human_by_finger.items()
                        }
            if human_by_finger:
                qpos = refine_fn(qpos, human_by_finger,
                                  alpha=alpha, normal_thr=normal_thr)
                n_refined += 1

        qpos_seq[t] = qpos
        last = qpos
        n_valid += 1

    suffix = "_contact" if use_contact else ""
    out_path = os.path.join(out_dir, f"qpos_xhand{suffix}_{hand}.pkl")
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
    extra = f"  refined={n_refined}" if use_contact else ""
    print(f"[{hand}] -> {out_path}  shape={qpos_seq.shape}  "
          f"valid={n_valid}/{T}  dof={dof}{extra}")
    retargeting.verbose()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hand", default="both", choices=["right", "left", "both"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--contact", action="store_true",
                    help="Enable stage-2 contact-aware refinement (Chamfer "
                         "matching of xhand fingertip mesh verts to human "
                         "MANO contact verts, palmar + fingertip filtered).")
    ap.add_argument("--contact_dir", default=None,
                    help="Per-frame HACO contact npz dir. Default: "
                         "<npz parent>/../contact")
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="Anchor weight on ||q - q_stage1|| in stage-2 loss.")
    args = ap.parse_args()

    RetargetingConfig.set_default_urdf_dir(URDF_ROOT)
    data = np.load(args.npz)
    npz_dir = os.path.dirname(os.path.abspath(args.npz))
    out_dir = args.out_dir or npz_dir
    os.makedirs(out_dir, exist_ok=True)

    contact_dir = None
    if args.contact:
        contact_dir = args.contact_dir or os.path.join(os.path.dirname(npz_dir), "contact")
        if not os.path.isdir(contact_dir):
            raise FileNotFoundError(f"--contact set but {contact_dir} not found")
        print(f"contact_dir: {contact_dir}")

    hands = ["right", "left"] if args.hand == "both" else [args.hand]
    for h in hands:
        retarget_one_hand(data, h, out_dir,
                          contact_dir=contact_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
