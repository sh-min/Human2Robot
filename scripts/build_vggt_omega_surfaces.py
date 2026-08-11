#!/usr/bin/env python3
"""Build conservative tracked-surface proxies from dual-view VGGT-Omega points."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1"
DATASET = ROOT / "data/cube_dataset/26.08.05_stereo_calibrated/1"
DEPTH = DATASET / "camera_2/inpainting/processed/view/0/depth_processor/depth_aligned_metric.npy"
sys.path.insert(0, str(ROOT / "scripts"))
from register_spar3d_mesh_pilot import (  # noqa: E402
    MeshRenderer,
    fit_approximate_sim3,
    load_image_and_mask,
    load_pilot_manifest,
    parse_mh_calibration,
    robust_object_surface,
    summarize_observation,
    undistort_observation,
)

OBJECTS = ("cup", "snack", "choco", "lock", "sweep")


def filtered_points(evidence: Path) -> tuple[np.ndarray, dict[str, int]]:
    with np.load(evidence, allow_pickle=False) as payload:
        points = np.asarray(payload["points_relative"], np.float64)
        views = np.asarray(payload["view_indices"], np.int16)
    selected = []
    counts: dict[str, int] = {}
    for view in (0, 1):
        current = points[views == view]
        before = len(current)
        if before >= 100:
            lo, hi = np.percentile(current, [0.5, 99.5], axis=0)
            current = current[np.all((current >= lo) & (current <= hi), axis=1)]
        if len(current):
            selected.append(current)
        counts[f"view_{view}_before"] = before
        counts[f"view_{view}_after"] = len(current)
    merged = np.concatenate(selected, axis=0)
    if len(merged) < 16 or not np.isfinite(merged).all():
        raise ValueError(f"invalid VGGT object points: {evidence}")
    return merged, counts


def main() -> int:
    depths = np.load(DEPTH, mmap_mode="r")
    for key in OBJECTS:
        root = PILOT / key
        manifest_path = root / "inputs/manifest.json"
        _, manifest = load_pilot_manifest(manifest_path)
        frame = int(manifest["selection"]["mh_frame_index"])
        image_path = Path(manifest["outputs"]["mh_image"]["path"])
        mask_path = Path(manifest["outputs"]["modal_mask"]["path"])
        _, _, image, mask = load_image_and_mask(image_path, mask_path)
        camera = parse_mh_calibration(manifest, image.shape[:2])
        depth = np.asarray(depths[frame], np.float32)
        image_u, mask_u, depth_u, camera_matrix = undistort_observation(
            image, mask, depth, camera
        )
        surface_u, surface_filter = robust_object_surface(depth_u, mask_u)
        observation = summarize_observation(mask_u, surface_u, camera_matrix)

        evidence = root / "vggt_omega/object_dual_mask_filtered_evidence.npz"
        points, counts = filtered_points(evidence)
        hull = ConvexHull(points, qhull_options="QJ")
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=np.asarray(hull.simplices, np.int64),
            process=True,
        )
        mesh.fix_normals()
        if len(mesh.faces) < 12 or not mesh.is_watertight:
            raise ValueError(f"{key}: failed to build watertight conservative hull")

        fit = fit_approximate_sim3(
            mesh,
            mask_u,
            surface_u,
            camera_matrix,
            observation,
            optimization_height=180,
            max_function_evaluations=240,
        )
        output = root / "vggt_omega_surface"
        output.mkdir(parents=True, exist_ok=True)
        mesh_path = output / "mesh.glb"
        mesh.export(mesh_path, include_normals=True)
        matrix = np.asarray(fit["final"]["matrix"], np.float64)
        np.savez_compressed(
            output / "registration_transform.npz",
            sim3_canonical_to_mh_camera=matrix,
            reference_mh_frame=np.int32(frame),
        )
        renderer = MeshRenderer(mesh, width=1280, height=720, camera_matrix=camera_matrix, textured=False)
        try:
            _, front_depth, silhouette = renderer.render(matrix)
        finally:
            renderer.close()
        np.save(output / "reference_front_depth_proxy_m.npy", front_depth.astype(np.float32))
        cv2.imwrite(str(output / "reference_silhouette.png"), silhouette.astype(np.uint8) * 255)
        intersection = int(np.logical_and(mask_u, silhouette).sum())
        union = int(np.logical_or(mask_u, silhouette).sum())
        report = {
            "schema_version": 1,
            "status": "complete",
            "kind": "vggt_omega_dual_view_conservative_convex_surface_proxy",
            "object_label": manifest["selection"]["object_label"],
            "reference_mh_frame": frame,
            "source_model": "facebook/VGGT-Omega VGGT-Omega-1B-512",
            "source_geometry": str(evidence.resolve()),
            "dual_view_points": counts,
            "surface_method": "per-view robust trim then 3D convex hull",
            "watertight": bool(mesh.is_watertight),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "reference_mh_iou": intersection / max(union, 1),
            "metric_verified": False,
            "collision_ready": False,
            "use": "conservative visual occlusion proxy only",
            "surface_filter": surface_filter,
            "outputs": {
                "mesh": str(mesh_path.resolve()),
                "registration": str((output / "registration_transform.npz").resolve()),
            },
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: report["reference_mh_iou"], "vertices": len(mesh.vertices), "faces": len(mesh.faces)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
