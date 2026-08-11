#!/usr/bin/env python3
"""Render a full 553-frame comparison across every labelled object segment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1"
DATASET = ROOT / "data/kitchen_dataset/26.08.05_stereo_calibrated/1"
PROC = DATASET / "camera_2/inpainting/processed/view/0"
OUTPUT = PILOT / "all_objects_comparison"
sys.path.insert(0, str(ROOT / "scripts"))
from compare_spar3d_xhand_occlusion_pilot import _remap_image, _remap_labels, _remap_mask, build_static_occlusion_masks, undistortion_maps, weighted_remap_depth  # noqa: E402
from compare_tracked_mesh_xhand_video import DepthPairRenderer, composite, title_panel  # noqa: E402
from register_spar3d_mesh_pilot import load_canonical_mesh  # noqa: E402
from track_choco_mesh_pose_dual import amodalize_box_mask  # noqa: E402
sys.path.insert(0, str(ROOT / "src/inpainting"))
from composite_xhand_object_barrier import resize_overlay_frame, restore_raw_object_pixels  # noqa: E402


OBJECTS = {
    "Cup": {"key": "cup", "start": 44, "end": 92, "track": "mh_sh_joint"},
    "Snack": {"key": "snack", "start": 120, "end": 159, "track": "mh_sh_joint"},
    "Choco": {"key": "choco", "start": 187, "end": 238, "track": "mh_sh_joint_mesh_track"},
    "Lock": {"key": "lock", "start": 267, "end": 307, "track": "mh_sh_joint"},
    "Sweep": {"key": "sweep", "start": 341, "end": 518, "track": "mh_sh_joint"},
}


def load_tracks(geometry: str):
    tracks = {}
    for label, spec in OBJECTS.items():
        root = PILOT / spec["key"]
        track_name = "mh_sh_joint_vggt" if geometry == "vggt_omega" else spec["track"]
        track = root / "object_pose_tracking" / track_name
        if label == "Choco" and geometry != "vggt_omega":
            frames = np.load(track / "frame_indices_mh.npy") if (track / "frame_indices_mh.npy").exists() else np.arange(187, 239)
            poses = np.load(track / "pose_canonical_to_mh_camera_joint_proxy.npy")
            sh_projected = np.load(track / "sh_reprojected_silhouette.npy", mmap_mode="r")
            sh_masks = np.load(root / "object_pose_tracking/sh_sam2/sh_choco_mask_sam2.npy", mmap_mode="r")
        else:
            frames = np.load(track / "frame_indices_mh.npy")
            poses = np.load(track / "pose_canonical_to_mh_camera_joint_proxy.npy")
            sh_projected = np.load(track / "sh_reprojected_silhouette.npy", mmap_mode="r")
            sh_mask_path = root / "object_pose_tracking/sh_sam2/object_mask_sam2.npy"
            if not sh_mask_path.exists():
                sh_mask_path = root / "object_pose_tracking/sh_sam2/sh_choco_mask_sam2.npy"
            sh_masks = np.load(sh_mask_path, mmap_mode="r")
        if frames.tolist() != list(range(spec["start"], spec["end"] + 1)):
            raise ValueError(f"{label} track does not cover its full label interval")
        mesh_path = root / ("vggt_omega_surface/mesh.glb" if geometry == "vggt_omega" else "spar3d/mesh.glb")
        mesh, _ = load_canonical_mesh(mesh_path)
        tracks[label] = {**spec, "frames": frames, "poses": poses, "sh_projected": sh_projected, "sh_masks": sh_masks, "mesh": mesh}
    return tracks


def label_for_frame(frame, segments):
    for segment in segments:
        if segment["start_frame"] <= frame <= segment["end_frame"]:
            return segment["label"]
    raise ValueError(f"frame {frame} is not covered by annotations")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("spar3d", "vggt_omega"), default="spar3d")
    args = parser.parse_args()
    output_dir = PILOT / ("all_objects_vggt_omega_comparison" if args.geometry == "vggt_omega" else "all_objects_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation = json.loads((ROOT / "8-5/data/annotations/1/gt_labels.json").read_text(encoding="utf-8"))
    stereo = json.loads((DATASET / "stereo_manifest.json").read_text(encoding="utf-8"))
    tracks = load_tracks(args.geometry)
    mh_cal, sh_cal = stereo["calibration"]["intrinsics_by_view"]["MH"], stereo["calibration"]["intrinsics_by_view"]["SH"]
    kmh, ksh = np.asarray(mh_cal["camera_matrix"], float), np.asarray(sh_cal["camera_matrix"], float)
    mh_x, mh_y = undistortion_maps(kmh, np.asarray(mh_cal["distortion_k1_k2_p1_p2_k3"]), width=1280, height=720)
    sh_x, sh_y = undistortion_maps(ksh, np.asarray(sh_cal["distortion_k1_k2_p1_p2_k3"]), width=1280, height=720)
    overlay = PROC / "overlay_processor"
    robot_rgb_all = np.load(overlay / "robot_rgb.npy", mmap_mode="r"); robot_depth_all = np.load(overlay / "robot_depth.npy", mmap_mode="r")
    robot_mask_all = np.load(overlay / "robot_mask.npy", mmap_mode="r"); hand_mask_all = np.load(overlay / "robot_hand_mask.npy", mmap_mode="r"); labels_all = np.load(overlay / "robot_finger_labels.npy", mmap_mode="r")
    support_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_amodal.npy", mmap_mode="r")
    restore_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_observed_clean.npy", mmap_mode="r")
    current_all = np.load(PROC / "overlay_best_inpaint_barrier/occluded_hand_mask.npy", mmap_mode="r")
    haco_all = np.load(PROC / "overlay_haco_dual/occluded_finger_mask.npy", mmap_mode="r")
    background_cap = cv2.VideoCapture(str(PROC / "object_completion_dual_haco_e2fgvi/video_object_completed.mp4"))
    output_name = "episode1_all_objects_vggt_omega_dual_camera_xhand_comparison.mp4" if args.geometry == "vggt_omega" else "episode1_all_objects_dual_camera_xhand_comparison.mp4"
    output_path = output_dir / output_name
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 832))
    if not writer.isOpened(): raise RuntimeError("failed to create full comparison video")
    renderer = None; renderer_label = None; per_segment = {label: {"frames": 0, "hidden_px": 0} for label in OBJECTS}; transition_frames = 0

    try:
        for frame in range(annotation["num_frames"]):
            label = label_for_frame(frame, annotation["segments"])
            raw = cv2.imread(str(DATASET / f"camera_2/rgb/rgb_frame{frame:06d}.jpg")); raw_u = _remap_image(raw, mh_x, mh_y)
            background_cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, bg = background_cap.read()
            if not ok: raise RuntimeError(f"failed background frame {frame}")
            base = restore_raw_object_pixels(_remap_image(bg, mh_x, mh_y), raw_u, _remap_mask(restore_all[frame], mh_x, mh_y))
            resized = resize_overlay_frame(robot_rgb_all[frame], robot_depth_all[frame], robot_mask_all[frame], hand_mask_all[frame], labels_all[frame], width=1280, height=720)
            rr, rd, rm, hm, fl = resized
            rr = _remap_image(rr, mh_x, mh_y); rd = weighted_remap_depth(rd, mh_x, mh_y); rm = _remap_mask(rm, mh_x, mh_y); hm = _remap_mask(hm, mh_x, mh_y); fl = _remap_labels(fl, mh_x, mh_y)
            current = _remap_mask(current_all[frame], mh_x, mh_y)
            existing = composite(base, rr, rm, hm, current)
            # The prepared common dataset ends at 552; fail-open tail policy
            # holds the final SH frame once the +5 lookup leaves that range.
            sh_frame = min(frame + 5, 552)
            sh_image_raw = cv2.imread(str(DATASET / f"camera_1/rgb/rgb_frame{sh_frame:06d}.jpg"))
            sh_image = _remap_image(sh_image_raw, sh_x, sh_y)

            if label in tracks:
                track = tracks[label]; local = frame - track["start"]
                if renderer_label != label:
                    if renderer is not None:
                        try: renderer.close()
                        except Exception: pass
                    renderer = DepthPairRenderer(track["mesh"], kmh, 1280, 720); renderer_label = label
                front, back, mesh_mask, _ = renderer.render(track["poses"][local])
                support = _remap_mask(support_all[frame], mh_x, mh_y); shared = support & mesh_mask
                haco = _remap_mask(haco_all[frame], mh_x, mh_y)
                masks = build_static_occlusion_masks(hand_mask=hm, finger_labels=fl, robot_depth=rd, object_support_mask=support, mesh_mask=shared, front_depth=np.where(shared, front, 0), back_depth=np.where(shared, back, 0), current_mask=current, contact_baseline_mask=haco, thumb_shell_m=.01958, finger_shell_m=.01465, palm_shell_m=.015, spatial_close_radius_px=3, spatial_front_slack_m=.003)
                joint = composite(base, rr, rm, hm, masks["spar_volume_filter"])
                target_contours, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); mesh_contours, _ = cv2.findContours(mesh_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mh_constraint = raw_u.copy(); cv2.drawContours(mh_constraint, target_contours, -1, (0, 0, 255), 3); cv2.drawContours(mh_constraint, mesh_contours, -1, (0, 255, 255), 3)
                sh_target = amodalize_box_mask(_remap_mask(track["sh_masks"][local], sh_x, sh_y)); sh_mesh = np.asarray(track["sh_projected"][local], bool)
                tc, _ = cv2.findContours(sh_target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); sc, _ = cv2.findContours(sh_mesh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(sh_image, tc, -1, (0, 0, 255), 3); cv2.drawContours(sh_image, sc, -1, (0, 255, 255), 3)
                geometry_title = "VGGT-Omega hull" if args.geometry == "vggt_omega" else "SPAR3D mesh"
                panels = [title_panel(existing, "1 Existing 2.5D barrier", f"frame {frame} | {label}"), title_panel(mh_constraint, f"2 MH {geometry_title} fit", "red=target, yellow=joint-pose surface"), title_panel(joint, f"3 MH+SH {geometry_title} volume", "front/back + XHand thickness + HaCo filter"), title_panel(sh_image, f"4 SH {geometry_title} reprojection", f"SH {sh_frame} | USED in pose loss")]
                per_segment[label]["frames"] += 1; per_segment[label]["hidden_px"] += int(masks["spar_volume_filter"].sum())
            else:
                transition_frames += 1
                panels = [title_panel(existing, "1 Existing 2.5D barrier", f"frame {frame} | Trans"), title_panel(raw_u, "2 MH transition", "no single active object label"), title_panel(existing, "3 Transition passthrough", "object mesh filter intentionally inactive"), title_panel(sh_image, "4 SH transition", f"SH {sh_frame} | no object pose update")]
            writer.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
            if frame % 25 == 0: print(f"frame={frame}/552 label={label}", flush=True)
    finally:
        writer.release(); background_cap.release()
        if renderer is not None:
            try: renderer.close()
            except Exception: pass

    report = {"schema_version": 1, "kind": "episode1_all_objects_actual_dual_camera_comparison", "geometry_source": args.geometry, "actual_vggt_omega_inference": args.geometry == "vggt_omega", "actual_dual_camera_pose_optimization": True, "frames": 553, "fps": 10, "duration_seconds": 55.3, "object_segments": per_segment, "transition_frames": transition_frames, "panels": ["existing 2.5D", f"MH target/{args.geometry} surface", f"MH+SH {args.geometry} volume XHand filter", f"SH target/{args.geometry} reprojection"], "output": str(output_path.resolve()), "limitations": ["Camera baseline scale is inferred because checker-square physical length is missing.", "VGGT-Omega outputs relative-scale points; the watertight convex hull is a conservative visual proxy, not physical geometry." if args.geometry == "vggt_omega" else "SPAR3D meshes and SH masks are model estimates.", "Low-IoU cross-view frames remain visible and are recorded in each object report."]}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__": raise SystemExit(main())
