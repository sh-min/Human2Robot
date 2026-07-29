"""Adapter: HaWoR wrist trajectory + xhand qpos -> RB5 arm joint angles (via
pinocchio IK) + base pose + per-frame hand data, saved as an npz the Isaac
renderer consumes. Runs in a pinocchio env (RFM_retarget); the Isaac renderer
runs in the isaac_lab env, so this file bridges them.

Output npz (default /result/skill2policy/rb5_overlay_input.npz):
    rb5_q        (T,6)   RB5 joint angles [base,shoulder,elbow,wrist1-3]
    T_cam_base   (4,4)   base pose in the OpenCV camera frame (auto-fit / override)
    wrist_pos    (T,3)   MANO wrist position, cam frame (for placing the hand)
    wrist_rot    (T,3,3) xhand wrist orientation in cam frame (R_cam_mano@R_mano_xhand)
    qpos         (T,12)  xhand finger angles
    valid        (T,)    bool
    img_focal    scalar
    joint_names  xhand finger joint names (json in a sidecar)
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

import rb5_arm_ik as ik

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor_npz", required=True)
    ap.add_argument("--pkl", required=True, help="qpos_xhand_right(_smooth).pkl")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--out", default="/result/skill2policy/rb5_overlay_input.npz")
    ap.add_argument("--base_override", default=None,
                    help="path to a 4x4 .npy base->cam transform (else auto-fit)")
    ap.add_argument("--base_shift", default="0,0,0",
                    help="x,y,z (m, CV cam frame) nudge on the auto-fit base; "
                         "+x = right, +y = down, +z = away.")
    ap.add_argument("--optimize_base", action="store_true",
                    help="search base placements for the lowest reach-error + jitter "
                         "(overrides --base_shift auto-fit).")
    ap.add_argument("--mount", choices=["floor", "ceiling"], default="floor",
                    help="floor: base below, arm rises from the bottom. "
                         "ceiling: base above, arm hangs from the top.")
    ap.add_argument("--w_ori", type=float, default=1.0,
                    help="orientation-tracking weight (0=position-only smooth arm; "
                         "1=follow the wrist orientation, more jitter near singularities).")
    ap.add_argument("--smooth_win", type=int, default=15,
                    help="savgol window (odd) on the joint trajectory; larger tames jitter.")
    args = ap.parse_args()

    import pickle
    ri = np.load(args.hawor_npz)
    hidx = 1 if args.side == "right" else 0
    joints = ri[f"joints_{args.side}"].astype(np.float64)   # (T,21,3) cam frame
    go = ri["mano_global_orient"]; valid = ri["valid"][hidx].astype(bool)
    focal = float(ri["img_focal"])
    Rmx = np.load(REPO / "src/retargeting/assets" / f"R_mano_xhand_{args.side}.npy").astype(np.float64)
    dq = pickle.load(open(args.pkl, "rb"))
    qpos = np.asarray(dq["data"]); jnames = list(dq["joint_names"])

    T = joints.shape[0]
    # Use the smoothed wrist from the retarget pkl (retarget --smooth smooths
    # wrist_pos + wrist_quat with proper quaternion unwrap). wrist_quat is
    # R_cam_xhand as scipy .as_quat() (xyzw). This smooths both the hand
    # placement and the RB5 IK target.
    wrist_pos = np.asarray(dq["wrist_pos"], np.float64)
    wrist_rot = Rotation.from_quat(np.asarray(dq["wrist_quat"], np.float64)).as_matrix()

    # Build the RB5 flange (link6) IK target so the flange MOUNTING FACE mates flush
    # with the xhand mount plate (right_arm_flange_link), like bolting the hand on.
    #   - xhand mount surface: centre = wrist_pos (xhand root), outward normal = +z of
    #     the root (fingers/palm extend along -z, so the back plate faces +z_hand).
    #   - RB5 flange face = tcp, FLANGE_TCP along link6 -y, outward normal -y_link6.
    #   - mate: tcp == wrist_pos  AND  y_link6 == +z_hand, so the flange normal
    #     (-y_link6) is anti-parallel to the hand normal (+z_hand) -> the two plates
    #     face each other, centres coincident. Clocking about y is free; align it with
    #     the wrist x so wrist3 tracks the hand smoothly.
    # link6 origin therefore sits FLANGE_TCP behind the wrist along +z_hand.
    FLANGE_TCP = 0.0967
    z_hand = np.einsum("tij,j->ti", wrist_rot, np.array([0., 0., 1.]))
    x_ref = np.einsum("tij,j->ti", wrist_rot, np.array([1., 0., 0.]))
    flange = np.tile(np.eye(4), (T, 1, 1))
    for t in range(T):
        y = z_hand[t] / (np.linalg.norm(z_hand[t]) + 1e-9)
        x = x_ref[t] - x_ref[t].dot(y) * y
        x /= (np.linalg.norm(x) + 1e-9)
        flange[t, :3, :3] = np.column_stack([x, y, np.cross(x, y)])
        flange[t, :3, 3] = wrist_pos[t] + FLANGE_TCP * y

    model, data, fid = ik.load_model()
    override = np.load(args.base_override) if args.base_override else None
    if args.optimize_base and override is None:
        T_cam_base, _ = ik.optimize_base(flange, wrist_pos, valid, model, data, fid,
                                         w_ori=args.w_ori, mount=args.mount)
    else:
        shift = tuple(float(x) for x in args.base_shift.split(","))
        T_cam_base = ik.auto_fit_base(wrist_pos[valid], model, data, fid, override=override,
                                      shift=shift, mount=args.mount)
    q, perr, reach = ik.solve_sequence(flange, valid, T_cam_base, model, data, fid,
                                       w_ori=args.w_ori, smooth_win=args.smooth_win)
    dq = np.abs(np.diff(q[valid], axis=0)).sum(1)
    print(f"[rb5] reachable {reach[valid].mean()*100:.0f}%  pos-err med "
          f"{np.nanmedian(perr[valid])*1000:.1f}mm  base(cam)={T_cam_base.translation.round(3)}  "
          f"w_ori={args.w_ori} jitter-snaps>0.3rad={(dq>0.3).mean()*100:.1f}%")

    Tcb = np.eye(4); Tcb[:3, :3] = T_cam_base.rotation; Tcb[:3, 3] = T_cam_base.translation
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, rb5_q=q.astype(np.float32), T_cam_base=Tcb.astype(np.float64),
             wrist_pos=wrist_pos.astype(np.float64), wrist_rot=wrist_rot.astype(np.float64),
             qpos=qpos.astype(np.float32), valid=valid, img_focal=focal, side=args.side)
    json.dump({"joint_names": jnames}, open(os.path.splitext(args.out)[0] + "_jointnames.json", "w"))
    print(f"[ok] wrote {args.out}  T={T}")


if __name__ == "__main__":
    main()
