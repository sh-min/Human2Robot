#!/usr/bin/env python3
"""Jointly refine Choco pose from synchronized MH and SH silhouettes.

The missing checker-square length is estimated as a single global scale from
cross-view silhouette agreement.  This makes SH an actual optimization term,
while keeping the result explicitly scale-estimated rather than metric GT.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1/choco"
DATASET = ROOT / "data/kitchen_dataset/26.08.05_stereo_calibrated/1"
PROC = DATASET / "camera_2/inpainting/processed/view/0"
OUTPUT = PILOT / "object_pose_tracking/mh_sh_joint_mesh_track"
sys.path.insert(0, str(ROOT / "scripts"))
from register_spar3d_mesh_pilot import MeshRenderer, load_canonical_mesh, make_sim3, mask_bbox, scaled_camera_matrix  # noqa: E402
from track_choco_mesh_pose import centroid, decompose_sim3, resize_mask  # noqa: E402


def amodalize_box_mask(mask: np.ndarray) -> np.ndarray:
    """Fill finger-cut notches with a conservative convex-hull proxy."""
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.concatenate(contours, axis=0) if contours else np.empty((0, 1, 2), np.int32)
    if len(points) < 3:
        return binary.astype(bool)
    hull = cv2.convexHull(points)
    output = np.zeros_like(binary)
    cv2.fillConvexPoly(output, hull, 1)
    return output.astype(bool)


def silhouette_score(target: np.ndarray, rendered: np.ndarray) -> tuple[float, dict[str, float]]:
    intersection = int(np.count_nonzero(target & rendered))
    union = int(np.count_nonzero(target | rendered))
    iou = intersection / max(union, 1)
    tc = centroid(target)
    rc = centroid(rendered) if rendered.any() else (1e4, 1e4)
    center_error = math.dist(tc, rc) / max(math.hypot(*target.shape), 1.0)
    try:
        tb, rb = mask_bbox(target), mask_bbox(rendered)
        te = np.asarray((tb[2] - tb[0], tb[3] - tb[1]), dtype=np.float64)
        re = np.asarray((rb[2] - rb[0], rb[3] - rb[1]), dtype=np.float64)
        extent_error = float(np.mean(np.abs(np.log(np.maximum(re, 1) / np.maximum(te, 1)))))
    except Exception:
        extent_error = 2.0
    loss = 0.72 * (1.0 - iou) + 0.18 * min(center_error / 0.05, 2.0) + 0.10 * min(extent_error, 2.0)
    return float(loss), {"iou": float(iou), "centroid_error_fraction": float(center_error), "extent_log_error": float(extent_error)}


def extrinsic_matrix(rotation: np.ndarray, translation_checker: np.ndarray, square_scale: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = square_scale * translation_checker
    return result


def estimate_square_scale(renderer: MeshRenderer, poses: np.ndarray, targets: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    sample_indices = sorted(set(list(range(0, len(poses), 4)) + [len(poses) - 1]))
    candidates = np.linspace(0.020, 0.031, 45)
    records = []
    for scale in candidates:
        ext = extrinsic_matrix(rotation, translation, float(scale))
        ious = []
        for index in sample_indices:
            _, _, rendered = renderer.render(ext @ poses[index])
            _, metrics = silhouette_score(targets[index], rendered)
            ious.append(metrics["iou"])
        records.append({"checker_square_scale_proxy_m": float(scale), "mean_sampled_sh_iou": float(np.mean(ious))})
    selected = max(records, key=lambda item: item["mean_sampled_sh_iou"])
    return float(selected["checker_square_scale_proxy_m"]), records


def render_sequence(mesh, k, poses, extrinsic=None):
    renderer = MeshRenderer(mesh, width=1280, height=720, camera_matrix=k, textured=False)
    masks = np.zeros((len(poses), 720, 1280), dtype=bool)
    depths = np.zeros((len(poses), 720, 1280), dtype=np.float16)
    try:
        for i, pose in enumerate(poses):
            matrix = pose if extrinsic is None else extrinsic @ pose
            _, depth, mask = renderer.render(matrix)
            masks[i] = mask
            depths[i] = depth.astype(np.float16)
    finally:
        try:
            renderer.close()
        except Exception as exc:
            print(f"warning: EGL cleanup after rendering: {exc}", file=sys.stderr)
    return masks, depths


def outline(image: np.ndarray, target: np.ndarray, rendered: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    target_contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    render_contours, _ = cv2.findContours(rendered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, target_contours, -1, (0, 0, 255), 3)
    cv2.drawContours(out, render_contours, -1, (0, 255, 255), 3)
    panel = np.zeros((416, 640, 3), dtype=np.uint8)
    cv2.putText(panel, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    panel[56:] = cv2.resize(out, (640, 360), interpolation=cv2.INTER_AREA)
    return panel


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PILOT / "inputs/manifest.json").read_text(encoding="utf-8"))
    frames = np.load(PILOT / "object_pose_tracking/mh_mesh_track/frame_indices.npy")
    sh_frames = np.load(PILOT / "object_pose_tracking/sh_sam2/frame_indices.npy")
    mono_poses = np.load(PILOT / "object_pose_tracking/mh_mesh_track/pose_canonical_to_mh_camera_proxy.npy")
    sh_masks_raw = np.load(PILOT / "object_pose_tracking/sh_sam2/sh_choco_mask_sam2.npy")
    mh_masks_raw = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_amodal.npy", mmap_mode="r")[frames]
    if not np.array_equal(sh_frames - 5, frames):
        raise ValueError("verified SH=MH+5 frame mapping is not satisfied")

    mh_cal = manifest["calibration"]["intrinsics_by_view"]["MH"]
    sh_cal = manifest["calibration"]["intrinsics_by_view"]["SH"]
    kmh, ksh = np.asarray(mh_cal["camera_matrix"], float), np.asarray(sh_cal["camera_matrix"], float)
    dmh, dsh = np.asarray(mh_cal["distortion_k1_k2_p1_p2_k3"], float), np.asarray(sh_cal["distortion_k1_k2_p1_p2_k3"], float)
    mh_x, mh_y = cv2.initUndistortRectifyMap(kmh, dmh, np.eye(3), kmh, (1280, 720), cv2.CV_32FC1)
    sh_x, sh_y = cv2.initUndistortRectifyMap(ksh, dsh, np.eye(3), ksh, (1280, 720), cv2.CV_32FC1)
    mh_masks = np.stack([cv2.remap(np.asarray(mask, np.uint8), mh_x, mh_y, cv2.INTER_NEAREST).astype(bool) for mask in mh_masks_raw])
    sh_modal = np.stack([cv2.remap(mask.astype(np.uint8), sh_x, sh_y, cv2.INTER_NEAREST).astype(bool) for mask in sh_masks_raw])
    sh_targets = np.stack([amodalize_box_mask(mask) for mask in sh_modal])
    low_shape = (180, 320)
    mh_low = np.stack([resize_mask(mask, low_shape) for mask in mh_masks])
    sh_low = np.stack([resize_mask(mask, low_shape) for mask in sh_targets])

    relative = np.asarray(manifest["calibration"]["relative_extrinsics"]["T_camera2_from_camera1"], float)
    r_mh_sh, t_mh_sh = relative[:3, :3], relative[:3, 3]
    mesh, mesh_stats = load_canonical_mesh(PILOT / "spar3d/mesh.glb")
    center = np.asarray(mesh.bounds, float).mean(axis=0)
    fixed_scale, _, _ = decompose_sim3(mono_poses[0], center)
    mh_renderer = MeshRenderer(mesh, width=320, height=180, camera_matrix=scaled_camera_matrix(kmh, (720, 1280), low_shape), textured=False)
    sh_renderer = MeshRenderer(mesh, width=320, height=180, camera_matrix=scaled_camera_matrix(ksh, (720, 1280), low_shape), textured=False)
    checker_scale, scale_search = estimate_square_scale(sh_renderer, mono_poses, sh_low, r_mh_sh, t_mh_sh)
    extrinsic = extrinsic_matrix(r_mh_sh, t_mh_sh, checker_scale)
    baseline_proxy_m = float(np.linalg.norm(extrinsic[:3, 3]))
    print(f"selected checker-square scale={checker_scale:.6f}, baseline={baseline_proxy_m:.3f} proxy-m", flush=True)

    dual_poses = np.zeros_like(mono_poses)
    records = []
    try:
        for index, frame in enumerate(frames):
            _, base_rotation, base_center = decompose_sim3(mono_poses[index], center)
            base_u = kmh[0, 0] * base_center[0] / base_center[2] + kmh[0, 2]
            base_v = kmh[1, 1] * base_center[1] / base_center[2] + kmh[1, 2]

            def decode(parameters):
                rotation = Rotation.from_rotvec(parameters[:3]).as_matrix() @ base_rotation
                u, v, z = base_u + parameters[3], base_v + parameters[4], base_center[2] + parameters[5]
                position = np.asarray(((u - kmh[0, 2]) * z / kmh[0, 0], (v - kmh[1, 2]) * z / kmh[1, 1], z))
                return rotation, position, make_sim3(fixed_scale, rotation, position, canonical_center=center)

            def evaluate(matrix, parameters):
                _, _, mh_render = mh_renderer.render(matrix)
                _, _, sh_render = sh_renderer.render(extrinsic @ matrix)
                mh_loss, mh_metrics = silhouette_score(mh_low[index], mh_render)
                sh_loss, sh_metrics = silhouette_score(sh_low[index], sh_render)
                regularizer = 0.025 * min(float(np.linalg.norm(parameters[:3])) / math.radians(25), 2.0) + 0.015 * min(math.hypot(float(parameters[3]), float(parameters[4])) / 25.0, 2.0)
                return 0.53 * mh_loss + 0.43 * sh_loss + regularizer, mh_metrics, sh_metrics

            def objective(parameters):
                try:
                    return evaluate(decode(parameters)[2], parameters)[0]
                except Exception:
                    return 10.0

            angle = math.radians(30)
            result = minimize(objective, np.zeros(6), method="Powell", bounds=[(-angle, angle)] * 3 + [(-30, 30), (-30, 30), (-0.08, 0.08)], options={"maxfev": 150, "xtol": 1e-3, "ftol": 2e-4})
            _, _, optimized = decode(result.x)
            candidates = [(mono_poses[index], np.zeros(6), "mono_fallback"), (optimized, result.x, "joint_optimized")]
            scored = [(evaluate(matrix, params), matrix, stage, params) for matrix, params, stage in candidates]
            (joint_loss, mh_metrics, sh_metrics), selected_pose, stage, parameters = min(scored, key=lambda item: item[0][0])
            dual_poses[index] = selected_pose
            records.append({"mh_frame": int(frame), "sh_frame": int(sh_frames[index]), "stage": stage, "joint_loss": float(joint_loss), "mh_iou": mh_metrics["iou"], "sh_iou": sh_metrics["iou"], "pose_delta": np.asarray(parameters).tolist()})
            print(f"MH={frame} SH={sh_frames[index]} mh_iou={mh_metrics['iou']:.3f} sh_iou={sh_metrics['iou']:.3f} {stage}", flush=True)
    finally:
        for renderer in (sh_renderer, mh_renderer):
            try:
                renderer.close()
            except Exception:
                pass

    mh_rendered, mh_depth = render_sequence(mesh, kmh, dual_poses)
    sh_rendered, _ = render_sequence(mesh, ksh, dual_poses, extrinsic=extrinsic)
    np.save(OUTPUT / "frame_indices_mh.npy", frames)
    np.save(OUTPUT / "frame_indices_sh.npy", sh_frames)
    np.save(OUTPUT / "pose_canonical_to_mh_camera_joint_proxy.npy", dual_poses)
    np.save(OUTPUT / "mh_tracked_front_depth_proxy_m.npy", mh_depth)
    np.save(OUTPUT / "mh_tracked_silhouette.npy", mh_rendered)
    np.save(OUTPUT / "sh_reprojected_silhouette.npy", sh_rendered)

    video = cv2.VideoWriter(str(OUTPUT / "mh_sh_joint_pose_alignment.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 416))
    for index, (mf, sf) in enumerate(zip(frames, sh_frames, strict=True)):
        mh_image = cv2.remap(cv2.imread(str(DATASET / f"camera_2/rgb/rgb_frame{int(mf):06d}.jpg")), mh_x, mh_y, cv2.INTER_LINEAR)
        sh_image = cv2.remap(cv2.imread(str(DATASET / f"camera_1/rgb/rgb_frame{int(sf):06d}.jpg")), sh_x, sh_y, cv2.INTER_LINEAR)
        left = outline(mh_image, mh_masks[index], mh_rendered[index], f"MH {mf} | joint mesh IoU {records[index]['mh_iou']:.3f}")
        right = outline(sh_image, sh_targets[index], sh_rendered[index], f"SH {sf} | USED in loss | IoU {records[index]['sh_iou']:.3f}")
        video.write(np.hstack((left, right)))
    video.release()

    mono_sh_ious = []
    temp_renderer = MeshRenderer(mesh, width=320, height=180, camera_matrix=scaled_camera_matrix(ksh, (720, 1280), low_shape), textured=False)
    try:
        for index, pose in enumerate(mono_poses):
            _, _, rendered = temp_renderer.render(extrinsic @ pose)
            mono_sh_ious.append(silhouette_score(sh_low[index], rendered)[1]["iou"])
    finally:
        try: temp_renderer.close()
        except Exception: pass
    report = {
        "schema_version": 1,
        "kind": "mh_sh_joint_silhouette_pose_track",
        "actual_dual_camera_optimization": True,
        "frame_mapping": "SH = MH + 5",
        "extrinsic": {"direction": "X_SH = R_SH_from_MH X_MH + t", "checker_square_scale_proxy_m": checker_scale, "baseline_proxy_m": baseline_proxy_m, "metric_verified": False, "scale_estimation": "global grid search maximizing sampled SH silhouette IoU"},
        "objective_weights": {"mh_silhouette": 0.53, "sh_silhouette": 0.43, "pose_regularization": 0.04},
        "sh_target": "SAM2 modal mask followed by convex-hull occlusion completion",
        "mesh": mesh_stats,
        "quality": {"mono_initial_mean_mh_iou": float(json.load(open(PILOT / 'object_pose_tracking/mh_mesh_track/report.json'))['quality']['mean_iou']), "mono_pose_mean_sh_iou": float(np.mean(mono_sh_ious)), "joint_mean_mh_iou": float(np.mean([r['mh_iou'] for r in records])), "joint_mean_sh_iou": float(np.mean([r['sh_iou'] for r in records])), "joint_selected_frames": int(sum(r['stage'] == 'joint_optimized' for r in records))},
        "scale_search": scale_search,
        "per_frame": records,
        "limitations": ["Checker-square physical size was absent; baseline scale is inferred from the proxy-size SPAR3D mesh and is not metric ground truth.", "SH masks and their convex-hull completion are model-inferred evidence.", "Both cameras now affect pose, but results remain visual overlay estimates rather than physical collision calibration."],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["quality"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
