#!/usr/bin/env python3
"""Track the approximately registered Choco mesh through MH frames 187--238.

This is a silhouette/depth-proxy 6DoF tracker.  The SPAR3D scale estimated on
frame 187 is held fixed (rigid object); only rotation and translation change.
Completed/amodal masks guide silhouette alignment, while depth is evaluated
only on the observed modal object pixels.  Results remain proxy-camera poses,
not metric ground truth.
"""

from __future__ import annotations

import argparse
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
DATASET = ROOT / "data/cube_dataset/26.08.05_stereo_calibrated/1"
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1/choco"
PROC = DATASET / "camera_2/inpainting/processed/view/0"
sys.path.insert(0, str(ROOT / "scripts"))
from register_spar3d_mesh_pilot import (  # noqa: E402
    MeshRenderer,
    load_canonical_mesh,
    make_sim3,
    mask_bbox,
    scaled_camera_matrix,
)


def decompose_sim3(matrix: np.ndarray, center: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    linear = np.asarray(matrix[:3, :3], dtype=np.float64)
    scale = float(np.cbrt(np.linalg.det(linear)))
    rotation = linear / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    center_position = matrix[:3, 3] + scale * rotation @ center
    return scale, rotation, center_position


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty object mask")
    return float(xs.mean()), float(ys.mean())


def median_depth(depth: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(depth, dtype=np.float32)[mask]
    values = values[np.isfinite(values) & (values > 0.05) & (values < 5.0)]
    if len(values) < 30:
        raise ValueError("insufficient valid object depth")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    keep = np.abs(values - median) <= max(0.025, 3.5 * mad)
    return float(np.median(values[keep]))


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def score_pose(
    target_amodal: np.ndarray,
    target_modal: np.ndarray,
    observed_depth: np.ndarray,
    rendered_mask: np.ndarray,
    rendered_depth: np.ndarray,
    temporal_rotation_radians: float,
    pixel_offset: float,
) -> tuple[float, dict[str, float]]:
    intersection = int(np.count_nonzero(target_amodal & rendered_mask))
    union = int(np.count_nonzero(target_amodal | rendered_mask))
    iou = intersection / max(union, 1)
    tc = centroid(target_amodal)
    rc = centroid(rendered_mask) if rendered_mask.any() else (1e4, 1e4)
    centroid_error = math.dist(tc, rc) / max(1.0, math.hypot(*target_amodal.shape))
    try:
        tb = mask_bbox(target_amodal)
        rb = mask_bbox(rendered_mask)
        te = np.asarray((tb[2] - tb[0], tb[3] - tb[1]), dtype=np.float64)
        re = np.asarray((rb[2] - rb[0], rb[3] - rb[1]), dtype=np.float64)
        extent_error = float(np.mean(np.abs(np.log(np.maximum(re, 1) / np.maximum(te, 1)))))
    except Exception:
        extent_error = 2.0
    paired = target_modal & rendered_mask & (observed_depth > 0) & (rendered_depth > 0)
    if int(paired.sum()) >= 20:
        depth_error = float(np.median(np.abs(observed_depth[paired] - rendered_depth[paired])))
    else:
        depth_error = 0.12
    loss = (
        0.64 * (1.0 - iou)
        + 0.10 * min(centroid_error / 0.04, 2.0)
        + 0.10 * min(extent_error, 2.0)
        + 0.10 * min(depth_error / 0.05, 2.0)
        + 0.05 * min(temporal_rotation_radians / math.radians(12.0), 2.0)
        + 0.01 * min(pixel_offset / 12.0, 2.0)
    )
    return float(loss), {
        "iou": float(iou),
        "centroid_error_fraction": float(centroid_error),
        "extent_log_error": float(extent_error),
        "depth_error_proxy_m": float(depth_error),
    }


def draw_pose_overlay(image: np.ndarray, target: np.ndarray, rendered: np.ndarray, frame: int, metrics: dict[str, float]) -> np.ndarray:
    out = image.copy()
    tint = np.zeros_like(out)
    tint[:, :, 2] = 255
    out[target] = cv2.addWeighted(out, 0.62, tint, 0.38, 0)[target]
    contours, _ = cv2.findContours(rendered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 3)
    cv2.rectangle(out, (0, 0), (out.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(out, f"MH {frame} | red=amodal evidence  yellow=tracked mesh  IoU={metrics['iou']:.3f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=187)
    parser.add_argument("--end", type=int, default=238)
    parser.add_argument("--optimization-height", type=int, default=180)
    parser.add_argument("--maxfev", type=int, default=90)
    parser.add_argument("--output", type=Path, default=PILOT / "object_pose_tracking/mh_mesh_track")
    args = parser.parse_args()
    frames = np.arange(args.start, args.end + 1, dtype=np.int32)
    if not len(frames) or frames[0] != 187:
        raise ValueError("this track must start from the verified frame-187 registration")

    manifest = json.loads((PILOT / "inputs/manifest.json").read_text(encoding="utf-8"))
    mh = manifest["calibration"]["intrinsics_by_view"]["MH"]
    k = np.asarray(mh["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(mh["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    height, width = 720, 1280
    map_x, map_y = cv2.initUndistortRectifyMap(k, distortion, np.eye(3), k, (width, height), cv2.CV_32FC1)

    modal_all = np.load(PROC / "object_layer/object_mask_modal.npy", mmap_mode="r")
    amodal_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_amodal.npy", mmap_mode="r")
    observed_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_observed_clean.npy", mmap_mode="r")
    depth_all = np.load(PROC / "depth_processor/depth_aligned_metric.npy", mmap_mode="r")
    transform_data = np.load(PILOT / "spar3d_registered_mh/registration_transform.npz")
    initial_matrix = np.asarray(transform_data["sim3_canonical_to_mh_camera"], dtype=np.float64)
    mesh, mesh_stats = load_canonical_mesh(PILOT / "spar3d/mesh.glb")
    center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    scale, rotation, center_position = decompose_sim3(initial_matrix, center)

    low_shape = (args.optimization_height, int(round(width * args.optimization_height / height)))
    low_k = scaled_camera_matrix(k, (height, width), low_shape)
    renderer = MeshRenderer(mesh, width=low_shape[1], height=low_shape[0], camera_matrix=low_k, textured=False)
    full_renderer = MeshRenderer(mesh, width=width, height=height, camera_matrix=k, textured=False)
    poses = np.zeros((len(frames), 4, 4), dtype=np.float64)
    poses[0] = initial_matrix
    metrics_list: list[dict[str, float | int | bool]] = []
    front_depth = np.zeros((len(frames), height, width), dtype=np.float16)
    silhouettes = np.zeros((len(frames), height, width), dtype=bool)
    args.output.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output / "mh_tracked_mesh_alignment.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("failed to create alignment video")

    previous_observed_z: float | None = None
    try:
        for local_index, frame in enumerate(frames):
            image = cv2.imread(str(DATASET / f"camera_2/rgb/rgb_frame{frame:06d}.jpg"), cv2.IMREAD_COLOR)
            amodal = cv2.remap(np.asarray(amodal_all[frame], dtype=np.uint8), map_x, map_y, cv2.INTER_NEAREST).astype(bool)
            modal = cv2.remap(np.asarray(observed_all[frame] | modal_all[frame], dtype=np.uint8), map_x, map_y, cv2.INTER_NEAREST).astype(bool)
            depth = cv2.remap(np.asarray(depth_all[frame], dtype=np.float32), map_x, map_y, cv2.INTER_LINEAR)
            observed_z = median_depth(depth, modal)
            target_u, target_v = centroid(amodal)
            low_amodal = resize_mask(amodal, low_shape)
            low_modal = resize_mask(modal, low_shape)
            low_depth = cv2.resize(depth, (low_shape[1], low_shape[0]), interpolation=cv2.INTER_AREA)

            if local_index:
                previous_rotation = rotation.copy()
                z_guess = center_position[2] + (observed_z - float(previous_observed_z))
                center_guess = np.asarray(((target_u - k[0, 2]) * z_guess / k[0, 0], (target_v - k[1, 2]) * z_guess / k[1, 1], z_guess))

                def decode(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    candidate_rotation = Rotation.from_rotvec(parameters[:3]).as_matrix() @ previous_rotation
                    u = target_u + float(parameters[3])
                    v = target_v + float(parameters[4])
                    z = z_guess + float(parameters[5])
                    candidate_center = np.asarray(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
                    return candidate_rotation, candidate_center, make_sim3(scale, candidate_rotation, candidate_center, canonical_center=center)

                def objective(parameters: np.ndarray) -> float:
                    try:
                        candidate_rotation, _candidate_center, matrix = decode(parameters)
                        _, rendered_depth, rendered_mask = renderer.render(matrix)
                        loss, _ = score_pose(low_amodal, low_modal, low_depth, rendered_mask, rendered_depth, float(np.linalg.norm(parameters[:3])), math.hypot(float(parameters[3]), float(parameters[4])))
                        return loss
                    except Exception:
                        return 10.0

                angle = math.radians(12.0)
                result = minimize(objective, np.zeros(6), method="Powell", bounds=[(-angle, angle)] * 3 + [(-14.0, 14.0), (-14.0, 14.0), (-0.035, 0.035)], options={"maxfev": args.maxfev, "xtol": 1e-3, "ftol": 2e-4})
                candidate_rotation, candidate_center, candidate_matrix = decode(result.x)
                previous_matrix = make_sim3(scale, previous_rotation, center_guess, canonical_center=center)
                candidates = [(candidate_matrix, candidate_rotation, candidate_center, bool(result.success), result.x), (previous_matrix, previous_rotation, center_guess, False, np.zeros(6))]
                evaluated = []
                for matrix, rot, pos, success, parameters in candidates:
                    _, rz, rm = renderer.render(matrix)
                    loss, met = score_pose(low_amodal, low_modal, low_depth, rm, rz, float(np.linalg.norm(parameters[:3])), math.hypot(float(parameters[3]), float(parameters[4])))
                    evaluated.append((loss, matrix, rot, pos, success, met))
                loss, poses[local_index], rotation, center_position, optimizer_success, low_metrics = min(evaluated, key=lambda item: item[0])
            else:
                _, rz, rm = renderer.render(initial_matrix)
                loss, low_metrics = score_pose(low_amodal, low_modal, low_depth, rm, rz, 0.0, 0.0)
                optimizer_success = True

            _, full_depth, full_mask = full_renderer.render(poses[local_index])
            front_depth[local_index] = full_depth.astype(np.float16)
            silhouettes[local_index] = full_mask
            full_loss, full_metrics = score_pose(amodal, modal, depth, full_mask, full_depth, 0.0, 0.0)
            record = {"frame": int(frame), "loss": float(full_loss), "optimizer_success": bool(optimizer_success), **full_metrics, "observed_depth_proxy_m": observed_z, "center_position_proxy_m": center_position.tolist()}
            metrics_list.append(record)
            writer.write(draw_pose_overlay(image, amodal, full_mask, int(frame), full_metrics))
            previous_observed_z = observed_z
            print(f"frame={frame} iou={full_metrics['iou']:.3f} depth={full_metrics['depth_error_proxy_m']:.3f} z={center_position[2]:.3f}", flush=True)
    finally:
        # EGL contexts must be released in reverse creation order.  Some
        # drivers otherwise fail while switching back to the older context.
        for active_renderer in (full_renderer, renderer):
            try:
                active_renderer.close()
            except Exception as exc:
                print(f"warning: renderer cleanup failed after completed rendering: {exc}", file=sys.stderr)
        writer.release()

    np.save(args.output / "frame_indices.npy", frames)
    np.save(args.output / "pose_canonical_to_mh_camera_proxy.npy", poses)
    np.save(args.output / "tracked_front_depth_proxy_m.npy", front_depth)
    np.save(args.output / "tracked_silhouette.npy", silhouettes)
    ious = np.asarray([float(item["iou"]) for item in metrics_list])
    report = {
        "schema_version": 1,
        "kind": "spar3d_mesh_framewise_pose_track",
        "status": "approximate_proxy_not_metric_ground_truth",
        "method": "fixed-scale sequential 6DoF silhouette/depth-proxy tracking",
        "frame_indices": frames.tolist(),
        "coordinate_system": "undistorted MH OpenCV camera coordinates; +Z forward",
        "scale": {"fixed_proxy_m_per_canonical_unit": scale, "metric_verified": False},
        "evidence": {"silhouette": "HACO-assisted amodal completion", "trusted_depth_region": "modal/observed-clean object mask", "depth": "Depth Anything V2 scaled by HaWoR anchors", "sh_role": "SAM2 auxiliary validation only; stereo translation lacks physical checker-square size"},
        "mesh": mesh_stats,
        "quality": {"mean_iou": float(ious.mean()), "min_iou": float(ious.min()), "frames_below_iou_0_5": frames[ious < 0.5].tolist()},
        "per_frame": metrics_list,
        "warnings": ["SPAR3D mesh is learned single-view geometry and non-watertight.", "Depth and pose scale are proxy values, not verified metric measurements.", "This track is suitable for overlay/occlusion experiments, not physical collision guarantees."],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["quality"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
