#!/usr/bin/env python3
"""Track Cup/Snack/Lock/Sweep with a true MH+SH joint silhouette loss."""

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
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1"
DATASET = ROOT / "data/cube_dataset/26.08.05_stereo_calibrated/1"
PROC = DATASET / "camera_2/inpainting/processed/view/0"
sys.path.insert(0, str(ROOT / "scripts"))
from register_spar3d_mesh_pilot import MeshRenderer, load_canonical_mesh, make_sim3, scaled_camera_matrix  # noqa: E402
from track_choco_mesh_pose import centroid, decompose_sim3, median_depth, resize_mask  # noqa: E402
from track_choco_mesh_pose_dual import amodalize_box_mask, extrinsic_matrix, render_sequence, silhouette_score  # noqa: E402

SPECS = {
    "cup": ("Cup", 44, 92, 58),
    "snack": ("Snack", 120, 159, 144),
    "choco": ("Choco", 187, 238, 187),
    "lock": ("Lock", 267, 307, 289),
    "sweep": ("Sweep", 341, 518, 462),
}
# SPAR3D has no shared metric scale across independently reconstructed objects.
# These per-object proxy factors align that arbitrary scale to the same stereo
# translation direction; they are estimated by reference-frame SH silhouette
# search and must not be interpreted as different physical camera baselines.
CHECKER_SCALE_PROXY_M = {"cup": 0.0290, "snack": 0.0295, "choco": 0.0245, "lock": 0.0295, "sweep": 0.0235}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("spar3d", "vggt_omega"), default="spar3d")
    args = parser.parse_args()
    stereo = json.loads((DATASET / "stereo_manifest.json").read_text(encoding="utf-8"))
    mh_cal, sh_cal = stereo["calibration"]["intrinsics_by_view"]["MH"], stereo["calibration"]["intrinsics_by_view"]["SH"]
    kmh, ksh = np.asarray(mh_cal["camera_matrix"], float), np.asarray(sh_cal["camera_matrix"], float)
    dmh, dsh = np.asarray(mh_cal["distortion_k1_k2_p1_p2_k3"], float), np.asarray(sh_cal["distortion_k1_k2_p1_p2_k3"], float)
    mh_x, mh_y = cv2.initUndistortRectifyMap(kmh, dmh, np.eye(3), kmh, (1280, 720), cv2.CV_32FC1)
    sh_x, sh_y = cv2.initUndistortRectifyMap(ksh, dsh, np.eye(3), ksh, (1280, 720), cv2.CV_32FC1)
    rel = np.asarray(stereo["calibration"]["relative_extrinsics"]["T_camera2_from_camera1"], float)
    amodal_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_amodal.npy", mmap_mode="r")
    modal_all = np.load(PROC / "object_layer/object_mask_modal.npy", mmap_mode="r")
    depth_all = np.load(PROC / "depth_processor/depth_aligned_metric.npy", mmap_mode="r")
    summary = {}

    for key, (label, start, end, reference) in SPECS.items():
        print(f"=== {label} {start}-{end} reference={reference} ===", flush=True)
        frames = np.arange(start, end + 1, dtype=np.int32)
        sh_frames = frames + 5
        root = PILOT / key
        output = root / "object_pose_tracking" / ("mh_sh_joint_vggt" if args.geometry == "vggt_omega" else "mh_sh_joint")
        output.mkdir(parents=True, exist_ok=True)
        checker_scale = CHECKER_SCALE_PROXY_M[key]
        extrinsic = extrinsic_matrix(rel[:3, :3], rel[:3, 3], checker_scale)
        geometry_root = root / ("vggt_omega_surface" if args.geometry == "vggt_omega" else "spar3d")
        registration_root = root / ("vggt_omega_surface" if args.geometry == "vggt_omega" else "spar3d_registered_mh")
        mesh, mesh_stats = load_canonical_mesh(geometry_root / "mesh.glb")
        initial = np.load(registration_root / "registration_transform.npz")["sim3_canonical_to_mh_camera"]
        center = np.asarray(mesh.bounds, float).mean(axis=0)
        fixed_scale, initial_rotation, initial_center = decompose_sim3(initial, center)

        sh_mask_path = root / "object_pose_tracking/sh_sam2/object_mask_sam2.npy"
        if not sh_mask_path.exists():
            sh_mask_path = root / "object_pose_tracking/sh_sam2/sh_choco_mask_sam2.npy"
        sh_raw = np.load(sh_mask_path)
        mh_targets = np.stack([cv2.remap(np.asarray(amodal_all[f], np.uint8), mh_x, mh_y, cv2.INTER_NEAREST).astype(bool) for f in frames])
        mh_modal = np.stack([cv2.remap(np.asarray(modal_all[f], np.uint8), mh_x, mh_y, cv2.INTER_NEAREST).astype(bool) for f in frames])
        sh_modal = np.stack([cv2.remap(mask.astype(np.uint8), sh_x, sh_y, cv2.INTER_NEAREST).astype(bool) for mask in sh_raw])
        sh_targets = np.stack([amodalize_box_mask(mask) for mask in sh_modal])
        depths = np.stack([cv2.remap(np.asarray(depth_all[f], np.float32), mh_x, mh_y, cv2.INTER_LINEAR) for f in frames])
        surface_z = np.asarray([median_depth(depth, mask) for depth, mask in zip(depths, mh_modal, strict=True)])
        ref_local = reference - start
        ref_surface = surface_z[ref_local]
        mh_low = np.stack([resize_mask(mask, (180, 320)) for mask in mh_targets])
        sh_low = np.stack([resize_mask(mask, (180, 320)) for mask in sh_targets])
        mh_renderer = MeshRenderer(mesh, width=320, height=180, camera_matrix=scaled_camera_matrix(kmh, (720, 1280), (180, 320)), textured=False)
        sh_renderer = MeshRenderer(mesh, width=320, height=180, camera_matrix=scaled_camera_matrix(ksh, (720, 1280), (180, 320)), textured=False)
        poses = np.zeros((len(frames), 4, 4), dtype=np.float64)
        records: list[dict | None] = [None] * len(frames)

        def track_order(order, starting_rotation):
            previous_rotation = starting_rotation.copy()
            for index in order:
                target_u, target_v = centroid(mh_targets[index])
                z_guess = initial_center[2] + (surface_z[index] - ref_surface)
                center_guess = np.asarray(((target_u - kmh[0, 2]) * z_guess / kmh[0, 0], (target_v - kmh[1, 2]) * z_guess / kmh[1, 1], z_guess))
                base_rotation = initial_rotation if index == ref_local else previous_rotation

                def decode(parameters):
                    rotation = Rotation.from_rotvec(parameters[:3]).as_matrix() @ base_rotation
                    u, v, z = target_u + parameters[3], target_v + parameters[4], z_guess + parameters[5]
                    position = np.asarray(((u - kmh[0, 2]) * z / kmh[0, 0], (v - kmh[1, 2]) * z / kmh[1, 1], z))
                    return rotation, position, make_sim3(fixed_scale, rotation, position, canonical_center=center)

                def evaluate(matrix, parameters):
                    _, _, mr = mh_renderer.render(matrix); _, _, sr = sh_renderer.render(extrinsic @ matrix)
                    ml, mm = silhouette_score(mh_low[index], mr); sl, sm = silhouette_score(sh_low[index], sr)
                    reg = .025 * min(float(np.linalg.norm(parameters[:3])) / math.radians(18), 2) + .01 * min(math.hypot(float(parameters[3]), float(parameters[4])) / 20, 2)
                    return .53 * ml + .43 * sl + reg, mm, sm

                def objective(parameters):
                    try: return evaluate(decode(parameters)[2], parameters)[0]
                    except Exception: return 10.0

                angle = math.radians(22)
                result = minimize(objective, np.zeros(6), method="Powell", bounds=[(-angle, angle)] * 3 + [(-28, 28), (-28, 28), (-.065, .065)], options={"maxfev": 125, "xtol": 1e-3, "ftol": 2e-4})
                rotation, _, optimized = decode(result.x)
                fallback = make_sim3(fixed_scale, base_rotation, center_guess, canonical_center=center)
                candidates = [(optimized, result.x, rotation, "joint"), (fallback, np.zeros(6), base_rotation, "fallback")]
                scored = [(evaluate(matrix, params), matrix, rot, stage) for matrix, params, rot, stage in candidates]
                (loss, mm, sm), poses[index], previous_rotation, stage = min(scored, key=lambda value: value[0][0])
                records[index] = {"mh_frame": int(frames[index]), "sh_frame": int(sh_frames[index]), "stage": stage, "joint_loss": float(loss), "mh_iou": mm["iou"], "sh_iou": sm["iou"]}
                print(f"{label} MH={frames[index]} MH-IoU={mm['iou']:.3f} SH-IoU={sm['iou']:.3f}", flush=True)

        try:
            track_order(range(ref_local, len(frames)), initial_rotation)
            if ref_local:
                track_order(range(ref_local - 1, -1, -1), initial_rotation)
        finally:
            for renderer in (sh_renderer, mh_renderer):
                try: renderer.close()
                except Exception: pass

        mh_rendered, mh_depth = render_sequence(mesh, kmh, poses)
        sh_rendered, _ = render_sequence(mesh, ksh, poses, extrinsic=extrinsic)
        np.save(output / "frame_indices_mh.npy", frames); np.save(output / "frame_indices_sh.npy", sh_frames)
        np.save(output / "pose_canonical_to_mh_camera_joint_proxy.npy", poses)
        np.save(output / "mh_front_depth_proxy_m.npy", mh_depth); np.save(output / "mh_silhouette.npy", mh_rendered); np.save(output / "sh_reprojected_silhouette.npy", sh_rendered)
        quality = {"mean_mh_iou": float(np.mean([r["mh_iou"] for r in records])), "mean_sh_iou": float(np.mean([r["sh_iou"] for r in records])), "minimum_mh_iou": float(np.min([r["mh_iou"] for r in records])), "minimum_sh_iou": float(np.min([r["sh_iou"] for r in records]))}
        report = {"schema_version": 1, "kind": "all_objects_mh_sh_joint_pose", "label": label, "geometry_source": args.geometry, "actual_dual_camera_optimization": True, "frame_mapping": "SH=MH+5", "mh_interval": [start, end], "reference_mh_frame": reference, "checker_square_scale_proxy_m": checker_scale, "scale_semantics": "per-object relative-geometry scale compensation, not a physical camera-baseline estimate", "metric_verified": False, "mesh": mesh_stats, "quality": quality, "per_frame": records}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary[key] = quality
        print(json.dumps({key: quality}), flush=True)
    summary_name = "all_objects_joint_pose_vggt_summary.json" if args.geometry == "vggt_omega" else "all_objects_joint_pose_summary.json"
    (PILOT / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
