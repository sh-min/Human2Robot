"""Adapter: HaWoR wrist trajectory + xhand qpos -> RB5 arm joint angles (via
pinocchio IK) + base pose + per-frame hand data, saved as an npz the Isaac
renderer consumes. Runs in a pinocchio env (RFM_retarget); the Isaac renderer
runs in the isaac_lab env, so this file bridges them.

Output npz (default /result/skill2policy/rb5_overlay_input.npz):
    rb5_q        (T,6)   RB5 joint angles [base,shoulder,elbow,wrist1-3]
    T_cam_base   (4,4)   fixed base pose in the OpenCV camera frame
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

# Canonical RB5 placement, accepted on No1__IMG_5368.  This is deliberately the
# only base pose in the pipeline: x is slightly farther right than the previous
# placement and y puts link0 completely below the 1920x1080 image.  Do not infer,
# mirror, optimise, or tune this transform per video.
FIXED_T_CAM_BASE = np.array(
    [
        [1.0, 0.0, 0.0, -0.61814264],
        [0.0, 0.0, -1.0, 0.55264644],
        [0.0, 1.0, 0.0, 0.42629680],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Canonical IK branch used by the accepted full-arm RB5 render.  A fixed seed
# prevents the same wrist target from selecting a visually different folded-arm
# solution on another video or machine.
FIXED_IK_INITIAL_Q = np.array(
    [
        -0.15171454846858978,
        0.8020381927490234,
        2.247098684310913,
        -0.1559724360704422,
        -2.27298641204834,
        -0.46591874957084656,
    ],
    dtype=np.float64,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor_npz", required=True)
    ap.add_argument("--pkl", required=True, help="qpos_xhand_<side>(_smooth).pkl")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--out", default="/result/skill2policy/rb5_overlay_input.npz")
    ap.add_argument("--img_w", type=int, default=1920)
    ap.add_argument("--img_h", type=int, default=1080,
                    help="source image size stored in the overlay contract.")
    ap.add_argument("--w_ori", type=float, default=1.0,
                    help="orientation-tracking weight (0=position-only smooth arm; "
                         "1=follow the wrist orientation, more jitter near singularities).")
    ap.add_argument("--smooth_win", type=int, default=21,
                    help="savgol window (odd) on the joint trajectory; larger tames jitter.")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="command/render frame rate used to enforce URDF velocity limits.")
    ap.add_argument("--position_margin", type=float, default=0.02,
                    help="radians kept away from every finite URDF hard stop.")
    ap.add_argument("--velocity_scale", type=float, default=0.90,
                    help="fraction of URDF velocity limits allowed (default leaves 10%% headroom).")
    args = ap.parse_args()

    if args.img_w <= 0 or args.img_h <= 0:
        raise ValueError(f"invalid image size: {args.img_w}x{args.img_h}")
    if args.fps <= 0 or not np.isfinite(args.fps):
        raise ValueError("--fps must be finite and positive")

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
    if not valid.any():
        raise ValueError(f"HaWoR contains no valid {args.side} frames")
    pkl_side = dq.get("hand")
    if pkl_side is not None and pkl_side != args.side:
        raise ValueError(
            f"retarget pkl side {pkl_side!r} does not match --side {args.side!r}"
        )
    embodiment = dq.get("embodiment", "xhand")
    if embodiment != "xhand":
        raise ValueError(f"Isaac RB5 overlay requires an xhand pkl, got {embodiment!r}")
    pkl_valid = np.asarray(dq.get("valid", valid), dtype=bool)
    if pkl_valid.shape != valid.shape or not np.array_equal(pkl_valid, valid):
        raise ValueError("retarget pkl validity does not match the HaWoR trajectory")
    if qpos.shape != (T, len(jnames)):
        raise ValueError(
            f"qpos/joint-name contract mismatch: qpos={qpos.shape}, "
            f"T={T}, joint_names={len(jnames)}"
        )
    if len(jnames) != 12 or len(set(jnames)) != len(jnames):
        raise ValueError(f"XHand renderer requires 12 unique joints, got {jnames}")
    # Use the smoothed wrist from the retarget pkl (retarget --smooth smooths
    # wrist_pos + wrist_quat with proper quaternion unwrap). wrist_quat is
    # R_cam_xhand as scipy .as_quat() (xyzw). This smooths both the hand
    # placement and the RB5 IK target.
    wrist_pos = np.asarray(dq["wrist_pos"], np.float64)
    wrist_quat = np.asarray(dq["wrist_quat"], np.float64)
    if wrist_pos.shape != (T, 3) or wrist_quat.shape != (T, 4):
        raise ValueError(
            f"wrist trajectory mismatch: pos={wrist_pos.shape}, quat={wrist_quat.shape}, T={T}"
        )
    if not (np.isfinite(qpos).all() and np.isfinite(wrist_pos).all() and np.isfinite(wrist_quat).all()):
        raise ValueError("retarget pkl contains non-finite trajectory values")
    quat_norm = np.linalg.norm(wrist_quat, axis=1)
    if np.any(quat_norm < 1e-6):
        raise ValueError("retarget pkl contains a zero-length wrist quaternion")
    wrist_rot = Rotation.from_quat(wrist_quat).as_matrix()

    # Retargeting smoothing is not constraint preserving. Clamp and rate-limit
    # the XHand command in the adapter as the final, unavoidable safety gate.
    hand_urdf = REPO / "src/retargeting/assets/xhand" / f"xhand_{args.side}.urdf"
    hand_model = pin.buildModelFromUrdf(str(hand_urdf))
    hand_names = [hand_model.names[i] for i in range(1, len(hand_model.names))]
    try:
        hand_order = np.asarray([hand_names.index(name) for name in jnames], dtype=int)
    except ValueError as exc:
        raise ValueError("XHand pkl joints do not match the selected XHand URDF") from exc
    hand_lo = hand_model.lowerPositionLimit[hand_order]
    hand_hi = hand_model.upperPositionLimit[hand_order]
    hand_vel = hand_model.velocityLimit[hand_order]
    raw_qpos = qpos.copy()
    qpos = ik.rate_limit_trajectory(
        qpos, hand_vel, args.fps, velocity_scale=args.velocity_scale,
        lower=hand_lo, upper=hand_hi,
    )

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
    T_cam_base = pin.SE3(
        FIXED_T_CAM_BASE[:3, :3], FIXED_T_CAM_BASE[:3, 3]
    )
    print(
        "[rb5] fixed base: "
        f"t_cam={T_cam_base.translation.round(3).tolist()}"
    )
    q, perr, reach = ik.solve_sequence(
        flange, valid, T_cam_base, model, data, fid,
        w_ori=args.w_ori, smooth_win=args.smooth_win,
        initial_q=FIXED_IK_INITIAL_Q,
        fps=args.fps, position_margin=args.position_margin,
        velocity_scale=args.velocity_scale,
    )
    dq = np.abs(np.diff(q, axis=0)).sum(1)
    rb5_velocity = np.abs(np.diff(q, axis=0)) * args.fps
    hand_velocity = np.abs(np.diff(qpos, axis=0)) * args.fps
    if np.any(qpos < hand_lo - 1e-7) or np.any(qpos > hand_hi + 1e-7):
        raise RuntimeError("internal error: XHand position safety gate failed")
    if len(qpos) > 1 and np.any(
        hand_velocity > hand_vel * args.velocity_scale + 1e-7
    ):
        raise RuntimeError("internal error: XHand velocity safety gate failed")

    rb5_collision_pairs = ik.assert_no_self_collision(
        model, data, ik.RB5_URDF, q, label="RB5-850e"
    )
    hand_q_pin = np.empty((T, hand_model.nq), dtype=np.float64)
    for pin_index, name in enumerate(hand_names):
        hand_q_pin[:, pin_index] = qpos[:, jnames.index(name)]
    hand_q_pin, hand_collision_adjusted, hand_collision_pairs = (
        ik.project_collision_free_trajectory(
            hand_model, hand_model.createData(), hand_urdf, hand_q_pin,
            max_step=(
                hand_model.velocityLimit * args.velocity_scale / args.fps
            ),
        )
    )
    for pin_index, name in enumerate(hand_names):
        qpos[:, jnames.index(name)] = hand_q_pin[:, pin_index]
    # Blending toward the preceding safe command preserves the existing
    # position/velocity bounds; assert that contract before publishing.
    hand_velocity = np.abs(np.diff(qpos, axis=0)) * args.fps
    if np.any(qpos < hand_lo - 1e-7) or np.any(qpos > hand_hi + 1e-7):
        raise RuntimeError("XHand collision projection violated position bounds")
    if len(qpos) > 1 and np.any(
        hand_velocity > hand_vel * args.velocity_scale + 1e-7
    ):
        raise RuntimeError("XHand collision projection violated velocity bounds")
    hand_collision_pairs = ik.assert_no_self_collision(
        hand_model, hand_model.createData(), hand_urdf, hand_q_pin,
        label=f"XHand {args.side}",
    )

    # The hand is bolted to link6. Once velocity constraints alter the arm IK,
    # derive the rendered hand root from the executable flange FK; retaining
    # the unconstrained human wrist pose would visually detach the two robots.
    T_base_cam = T_cam_base.inverse()
    mounted_pos = np.empty_like(wrist_pos)
    mounted_rot = np.empty_like(wrist_rot)
    for t in range(T):
        pin.forwardKinematics(model, data, q[t])
        pin.updateFramePlacements(model, data)
        flange_cam = T_cam_base * data.oMf[fid]
        x_hand = flange_cam.rotation[:, 0]
        z_hand = flange_cam.rotation[:, 1]
        y_hand = np.cross(z_hand, x_hand)
        mounted_rot[t] = np.column_stack([x_hand, y_hand, z_hand])
        mounted_pos[t] = flange_cam.translation - FLANGE_TCP * z_hand
    wrist_pos = mounted_pos
    wrist_rot = mounted_rot
    print(f"[rb5] reachable {reach[valid].mean()*100:.0f}%  pos-err med "
          f"{np.nanmedian(perr[valid])*1000:.1f}mm  base(cam)={T_cam_base.translation.round(3)}  "
          f"w_ori={args.w_ori} jitter-snaps>0.3rad={(dq>0.3).mean()*100:.1f}%  "
          f"max-vel={rb5_velocity.max():.2f}rad/s")
    print(
        f"[xhand] constrained-frames="
        f"{np.count_nonzero(np.any(np.abs(qpos - raw_qpos) > 1e-7, axis=1))}/{T}  "
        f"max-vel={hand_velocity.max():.2f}rad/s  "
        f"collision-adjusted={hand_collision_adjusted}  "
        f"self-collision-pairs={hand_collision_pairs} clear"
    )
    print(f"[rb5] self-collision-pairs={rb5_collision_pairs} clear")

    Tcb = np.eye(4); Tcb[:3, :3] = T_cam_base.rotation; Tcb[:3, 3] = T_cam_base.translation
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, rb5_q=q.astype(np.float32), T_cam_base=Tcb.astype(np.float64),
             wrist_pos=wrist_pos.astype(np.float64), wrist_rot=wrist_rot.astype(np.float64),
             qpos=qpos.astype(np.float32), valid=valid, img_focal=focal,
             img_width=np.int32(args.img_w), img_height=np.int32(args.img_h),
             side=args.side, command_fps=np.float64(args.fps),
             safety_position_margin_rad=np.float64(args.position_margin),
             safety_velocity_scale=np.float64(args.velocity_scale),
             safety_constraints_enforced=np.bool_(True))
    json.dump(
        {
            "joint_names": jnames,
            "side": args.side,
            "embodiment": embodiment,
            "trajectory_constraints": {
                "enforced": True,
                "fps": args.fps,
                "position_margin_rad": args.position_margin,
                "velocity_scale": args.velocity_scale,
                "rb5_max_velocity_rad_s": float(rb5_velocity.max()),
                "xhand_max_velocity_rad_s": float(hand_velocity.max()),
                "hand_root_source": "rb5_link6_fk_rigid_mount",
                "rb5_self_collision_pairs_checked": rb5_collision_pairs,
                "xhand_self_collision_pairs_checked": hand_collision_pairs,
                "xhand_collision_adjusted_frames": hand_collision_adjusted,
            },
        },
        open(os.path.splitext(args.out)[0] + "_jointnames.json", "w"),
        indent=2,
    )
    print(f"[ok] wrote {args.out}  T={T}")


if __name__ == "__main__":
    main()
