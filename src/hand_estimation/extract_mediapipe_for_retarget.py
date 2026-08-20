"""Create Skill2Policy-compatible hand data from a video with MediaPipe.

This is a lightweight fallback for machines where the full HaWoR conda
environment is unavailable.  It writes the same core artifacts consumed by
the inpainting and xhand rendering stages:

* ``retarget_input.npz`` with camera-space 21-joint hand trajectories;
* ``qpos_xhand_{right,left}.pkl`` with approximate 12-DOF xhand poses;
* ``hand_processor/hand_data_*.npz`` and ``bbox_processor/bbox_data.npz``;
* a landmark projection video for visual verification.

The camera-space depth is estimated from the apparent palm width and a nominal
physical palm width.  It is therefore suitable for visual overlay and depth
ordering, not metric evaluation.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


PALM_IDXS = (5, 9, 13, 17)
FINGER_CHAINS = {
    "index": (5, 6, 7, 8),
    "mid": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _fill_short_gaps(valid: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(valid, dtype=bool).copy()
    good = np.flatnonzero(out)
    for left, right in zip(good[:-1], good[1:]):
        if right - left - 1 <= max_gap:
            out[left:right + 1] = True
    return out


def _interp_smooth(values: np.ndarray, valid: np.ndarray, window: int = 15) -> np.ndarray:
    """Interpolate missing rows, then Savitzky-Golay smooth along time."""
    arr = np.asarray(values, dtype=np.float64).copy()
    t = np.arange(len(arr))
    good = np.flatnonzero(valid)
    if not len(good):
        return np.zeros_like(arr)
    flat = arr.reshape(len(arr), -1)
    if len(good) == 1:
        flat[:] = flat[good[0]]
    else:
        for col in range(flat.shape[1]):
            flat[:, col] = np.interp(t, good, flat[good, col])
    win = min(window, len(arr) if len(arr) % 2 else len(arr) - 1)
    if win >= 5:
        flat[:] = savgol_filter(flat, win, 2, axis=0, mode="interp")
    return flat.reshape(arr.shape)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


def _bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Zero when a-b-c is straight; positive as the joint flexes."""
    return float(np.pi - _angle(a - b, c - b))


def _xhand_qpos(joints: np.ndarray, side: str) -> tuple[list[str], np.ndarray]:
    prefix = f"{side}_hand_"
    names = [
        prefix + "thumb_bend_joint",
        prefix + "thumb_rota_joint1",
        prefix + "thumb_rota_joint2",
        prefix + "index_bend_joint",
        prefix + "index_joint1",
        prefix + "index_joint2",
        prefix + "mid_joint1",
        prefix + "mid_joint2",
        prefix + "ring_joint1",
        prefix + "ring_joint2",
        prefix + "pinky_joint1",
        prefix + "pinky_joint2",
    ]
    result = np.zeros((len(joints), len(names)), dtype=np.float32)
    for frame, j in enumerate(joints):
        palm_width = max(float(np.linalg.norm(j[5] - j[17])), 1e-5)
        thumb_near = np.clip(
            1.0 - float(np.linalg.norm(j[4] - j[5])) / (1.8 * palm_width),
            0.0, 1.0,
        )
        thumb_mcp = _bend(j[1], j[2], j[3])
        thumb_ip = _bend(j[2], j[3], j[4])
        result[frame, 0] = np.clip(0.25 + 1.15 * thumb_near, 0.0, 1.83)
        result[frame, 1] = np.clip(0.15 + 0.65 * thumb_near + 0.35 * thumb_mcp,
                                           -1.05, 1.57)
        result[frame, 2] = np.clip(0.75 * thumb_ip + 0.35 * thumb_mcp,
                                           -0.175, 1.83)

        # Index lateral spread is small on xhand.  The remaining eight joints
        # are flexion pairs for index/middle/ring/pinky.
        result[frame, 3] = 0.0
        out_col = 4
        for chain in FINGER_CHAINS.values():
            mcp, pip, dip, tip = chain
            mcp_flex = _angle(j[pip] - j[mcp], j[mcp] - j[0])
            pip_flex = _bend(j[mcp], j[pip], j[dip])
            dip_flex = _bend(j[pip], j[dip], j[tip])
            result[frame, out_col] = np.clip(
                0.55 * mcp_flex + 0.55 * pip_flex, 0.0, 1.92)
            result[frame, out_col + 1] = np.clip(
                0.80 * pip_flex + 0.55 * dip_flex, 0.0, 1.92)
            out_col += 2
    return names, result


def _bbox_from_points(points: np.ndarray, valid: np.ndarray,
                      width: int, height: int, pad: float = 0.35):
    mins = points.min(axis=1)
    maxs = points.max(axis=1)
    centers = (mins + maxs) * 0.5
    side = np.maximum(maxs[:, 0] - mins[:, 0], maxs[:, 1] - mins[:, 1])
    side *= 1.0 + pad
    boxes = np.stack([
        centers[:, 0] - side * 0.5,
        centers[:, 1] - side * 0.5,
        centers[:, 0] + side * 0.5,
        centers[:, 1] + side * 0.5,
    ], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    min_edge = np.stack([
        boxes[:, 0], boxes[:, 1], width - 1 - boxes[:, 2],
        height - 1 - boxes[:, 3],
    ], axis=1).min(axis=1)
    boxes[~valid] = 0
    centers[~valid] = 0
    min_edge[~valid] = 0
    return boxes.astype(np.float32), centers.astype(np.float32), min_edge.astype(np.float32)


def _rotation_trajectory(joints: np.ndarray, side: str, align: np.ndarray) -> np.ndarray:
    rotations = []
    previous = np.eye(3)
    for j in joints:
        wrist = j[0]
        palm = j[list(PALM_IDXS)].mean(axis=0)
        z_axis = wrist - palm  # xhand +z: wrist toward elbow
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
        if side == "right":
            x_axis = j[5] - j[17]  # pinky -> index
        else:
            x_axis = j[17] - j[5]
        x_axis -= z_axis * float(np.dot(x_axis, z_axis))
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-8:
            rotations.append(previous)
            continue
        x_axis /= x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1e-8)
        x_axis = np.cross(y_axis, z_axis)
        r_cam_hand = np.column_stack([x_axis, y_axis, z_axis])
        r_cam_mano = r_cam_hand @ align.T
        previous = r_cam_mano
        rotations.append(r_cam_mano)
    rotvec = Rotation.from_matrix(np.stack(rotations)).as_rotvec()
    rotvec = savgol_filter(rotvec, min(15, len(rotvec) if len(rotvec) % 2 else len(rotvec) - 1),
                           2, axis=0, mode="interp") if len(rotvec) >= 5 else rotvec
    return rotvec.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--focal", type=float, default=925.0)
    parser.add_argument("--palm_width_m", type=float, default=0.075)
    parser.add_argument("--max_gap", type=int, default=16)
    parser.add_argument("--min_detection_confidence", type=float, default=0.30)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.30)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    uv = np.zeros((frame_count, 21, 2), dtype=np.float32)
    landmark_z = np.zeros((frame_count, 21), dtype=np.float32)
    detected = np.zeros(frame_count, dtype=bool)
    labels: list[str] = []

    hands_api = mp.solutions.hands
    with hands_api.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        idx = 0
        while idx < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                # With one active hand, largest apparent palm is the most stable
                # choice and avoids relying on mirrored-camera handedness labels.
                candidates = []
                for lm_idx, landmarks in enumerate(result.multi_hand_landmarks):
                    pts = np.array([[p.x * width, p.y * height, p.z]
                                    for p in landmarks.landmark], dtype=np.float32)
                    candidates.append((float(np.linalg.norm(pts[5, :2] - pts[17, :2])),
                                       lm_idx, pts))
                _, lm_idx, pts = max(candidates, key=lambda item: item[0])
                uv[idx] = pts[:, :2]
                landmark_z[idx] = pts[:, 2]
                detected[idx] = True
                if result.multi_handedness and lm_idx < len(result.multi_handedness):
                    labels.append(result.multi_handedness[lm_idx].classification[0].label)
            idx += 1
    cap.release()
    frame_count = idx
    uv = uv[:idx]
    landmark_z = landmark_z[:idx]
    detected = detected[:idx]
    if detected.sum() < 2:
        raise RuntimeError(f"MediaPipe detected the hand in only {int(detected.sum())} frames")

    valid = _fill_short_gaps(detected, args.max_gap)
    uv_s = _interp_smooth(uv, detected, window=15).astype(np.float32)
    z_s = _interp_smooth(landmark_z, detected, window=15).astype(np.float32)
    palm_px = np.linalg.norm(uv_s[:, 5] - uv_s[:, 17], axis=1)
    depth = args.focal * args.palm_width_m / np.clip(palm_px, 20.0, None)
    depth = np.clip(_interp_smooth(depth[:, None], valid, window=21)[:, 0], 0.30, 1.50)
    dz = (z_s - z_s[:, :1]) * width * depth[:, None] / args.focal
    joint_z = depth[:, None] + 0.55 * dz
    joint_x = (uv_s[..., 0] - width * 0.5) * joint_z / args.focal
    joint_y = (uv_s[..., 1] - height * 0.5) * joint_z / args.focal
    joints = np.stack([joint_x, joint_y, joint_z], axis=-1).astype(np.float32)

    align_path = (Path(__file__).resolve().parents[1] / "retargeting" / "assets" /
                  f"R_mano_xhand_{args.side}.npy")
    align = np.load(align_path).astype(np.float64)
    rotvec = _rotation_trajectory(joints, args.side, align)
    joint_names, qpos = _xhand_qpos(joints, args.side)
    qpos = _interp_smooth(qpos, valid, window=15).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inactive = "left" if args.side == "right" else "right"
    zeros_joints = np.zeros_like(joints)
    valid_both = np.zeros((2, frame_count), dtype=bool)
    active_idx = 1 if args.side == "right" else 0
    valid_both[active_idx] = valid
    joints_left = joints if args.side == "left" else zeros_joints
    joints_right = joints if args.side == "right" else zeros_joints
    orient = np.zeros((2, frame_count, 3), dtype=np.float32)
    orient[active_idx] = rotvec
    trans = np.zeros((2, frame_count, 3), dtype=np.float32)
    trans[active_idx] = joints[:, 0]
    npz_path = args.out_dir / "retarget_input.npz"
    np.savez_compressed(
        npz_path,
        joints_left=joints_left,
        joints_right=joints_right,
        verts_left=np.zeros((frame_count, 778, 3), dtype=np.float32),
        verts_right=np.zeros((frame_count, 778, 3), dtype=np.float32),
        mano_trans=trans,
        mano_global_orient=orient,
        mano_hand_pose=np.zeros((2, frame_count, 15, 3), dtype=np.float32),
        mano_betas=np.zeros((2, frame_count, 10), dtype=np.float32),
        valid=valid_both,
        img_focal=np.float32(args.focal),
        start_idx=np.int64(0),
        end_idx=np.int64(frame_count),
        frame_is_cam_space=np.bool_(True),
        source=np.array("mediapipe_approx"),
    )

    wrist_quat = Rotation.from_matrix(
        Rotation.from_rotvec(rotvec).as_matrix() @ align
    ).as_quat().astype(np.float32)
    active_pkl = dict(
        data=qpos,
        wrist_pos=joints[:, 0],
        wrist_quat=wrist_quat,
        valid=valid,
        joint_names=joint_names,
        hand=args.side,
        dof=len(joint_names),
        embodiment="xhand",
        source="mediapipe_approx",
    )
    with open(args.out_dir / f"qpos_xhand_{args.side}.pkl", "wb") as handle:
        pickle.dump(active_pkl, handle)
    inactive_names = [name.replace(args.side, inactive, 1) for name in joint_names]
    with open(args.out_dir / f"qpos_xhand_{inactive}.pkl", "wb") as handle:
        pickle.dump(dict(active_pkl, data=np.zeros_like(qpos), valid=np.zeros_like(valid),
                         joint_names=inactive_names, hand=inactive), handle)

    # Inputs expected by segment_arms.py.
    hp = args.processed_demo / "hand_processor"
    bp = args.processed_demo / "bbox_processor"
    hp.mkdir(parents=True, exist_ok=True)
    bp.mkdir(parents=True, exist_ok=True)
    frame_indices = np.arange(frame_count, dtype=np.int64)
    inactive_uv = np.zeros_like(uv_s)
    inactive_j = np.zeros_like(joints)
    for side in ("left", "right"):
        active = side == args.side
        np.savez_compressed(
            hp / f"hand_data_{side}.npz",
            frame_indices=frame_indices,
            hand_detected=valid if active else np.zeros_like(valid),
            kpts_2d=uv_s if active else inactive_uv,
            kpts_3d=joints if active else inactive_j,
        )
    boxes, centers, min_edge = _bbox_from_points(uv_s, valid, width, height)
    zero_boxes = np.zeros_like(boxes)
    zero_centers = np.zeros_like(centers)
    zero_edge = np.zeros_like(min_edge)
    np.savez_compressed(
        bp / "bbox_data.npz",
        left_hand_detected=valid if args.side == "left" else np.zeros_like(valid),
        right_hand_detected=valid if args.side == "right" else np.zeros_like(valid),
        left_bboxes=boxes if args.side == "left" else zero_boxes,
        right_bboxes=boxes if args.side == "right" else zero_boxes,
        left_bboxes_ctr=centers if args.side == "left" else zero_centers,
        right_bboxes_ctr=centers if args.side == "right" else zero_centers,
        left_bbox_min_dist_to_edge=min_edge if args.side == "left" else zero_edge,
        right_bbox_min_dist_to_edge=min_edge if args.side == "right" else zero_edge,
    )

    # Projection preview uses the smoothed prompts actually consumed downstream.
    debug_path = args.out_dir / "mediapipe_projection.mp4"
    cap = cv2.VideoCapture(str(args.input))
    writer = cv2.VideoWriter(str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    connections = list(hands_api.HAND_CONNECTIONS)
    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        if valid[frame_idx]:
            pts = np.rint(uv_s[frame_idx]).astype(int)
            for a, b in connections:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (40, 220, 40), 2,
                         cv2.LINE_AA)
            for point in pts:
                cv2.circle(frame, tuple(point), 3, (0, 80, 255), -1, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()

    label_summary = {label: labels.count(label) for label in sorted(set(labels))}
    print(f"[ok] {npz_path}")
    print(f"[ok] {debug_path}")
    print(f"[info] frames={frame_count}, detected={int(detected.sum())}, "
          f"valid_after_gap_fill={int(valid.sum())}, labels={label_summary}")
    print(f"[info] estimated wrist depth median={float(np.median(depth[valid])):.3f} m")


if __name__ == "__main__":
    main()
