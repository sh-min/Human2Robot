#!/usr/bin/env python3
"""Build a six-panel XHand/object-pose comparison for MH frames 187--238."""

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
DATASET = ROOT / "data/cube_dataset/26.08.05_stereo_calibrated/1"
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1/choco"
PROC = DATASET / "camera_2/inpainting/processed/view/0"
sys.path.insert(0, str(ROOT / "scripts"))
from compare_spar3d_xhand_occlusion_pilot import (  # noqa: E402
    T_CV_TO_GL,
    _remap_image,
    _remap_labels,
    _remap_mask,
    build_static_occlusion_masks,
    orient_depth_pair,
    undistortion_maps,
    weighted_remap_depth,
)
sys.path.insert(0, str(ROOT / "src/inpainting"))
from composite_rb5_contact_occlusion import composite_frame  # noqa: E402
from composite_xhand_object_barrier import resize_overlay_frame, restore_raw_object_pixels  # noqa: E402
from register_spar3d_mesh_pilot import load_canonical_mesh  # noqa: E402
from track_choco_mesh_pose_dual import amodalize_box_mask  # noqa: E402


class DepthPairRenderer:
    def __init__(self, mesh, camera_matrix: np.ndarray, width: int, height: int):
        import pyrender
        import trimesh

        self.pyrender = pyrender
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        self.scenes = []
        self.nodes = []
        for reverse in (False, True):
            use_faces = faces[:, ::-1] if reverse else faces
            plain = trimesh.Trimesh(vertices=vertices, faces=use_faces, process=False)
            scene = pyrender.Scene(bg_color=(0, 0, 0, 0))
            scene.add(pyrender.IntrinsicsCamera(fx=float(camera_matrix[0, 0]), fy=float(camera_matrix[1, 1]), cx=float(camera_matrix[0, 2]), cy=float(camera_matrix[1, 2]), znear=0.01, zfar=5.0), pose=np.eye(4))
            node = scene.add(pyrender.Mesh.from_trimesh(plain, smooth=False), pose=np.eye(4))
            self.scenes.append(scene)
            self.nodes.append(node)
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def render(self, matrix: np.ndarray):
        depths = []
        pose = T_CV_TO_GL @ matrix
        for scene, node in zip(self.scenes, self.nodes, strict=True):
            scene.set_pose(node, pose)
            depths.append(np.asarray(self.renderer.render(scene, flags=self.pyrender.RenderFlags.DEPTH_ONLY), dtype=np.float32))
        return orient_depth_pair(depths[0], depths[1])

    def close(self):
        self.renderer.delete()


def read_video_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"failed to read video frame {frame}")
    return image


def title_panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    # Keep labels completely outside the image so that no object/hand pixels
    # are hidden by the comparison UI.
    panel = np.zeros((416, 640, 3), dtype=np.uint8)
    panel[56:] = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 220, 255), 1, cv2.LINE_AA)
    return panel


def composite(background, robot_rgb, robot_mask, hand_mask, hidden):
    final, _, _ = composite_frame(background, robot_rgb, robot_mask, hand_mask, hidden, robot_edge_sigma_px=0.6, occlusion_edge_sigma_px=0.0)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PILOT / "object_pose_tracking/comparison")
    args = parser.parse_args()
    track_root = PILOT / "object_pose_tracking/mh_mesh_track"
    frames = np.load(track_root / "frame_indices.npy")
    mono_poses = np.load(track_root / "pose_canonical_to_mh_camera_proxy.npy")
    joint_root = PILOT / "object_pose_tracking/mh_sh_joint_mesh_track"
    dual_poses = np.load(joint_root / "pose_canonical_to_mh_camera_joint_proxy.npy")
    sh_reprojected = np.load(joint_root / "sh_reprojected_silhouette.npy", mmap_mode="r")
    sh_frames = np.load(PILOT / "object_pose_tracking/sh_sam2/frame_indices.npy")
    sh_masks = np.load(PILOT / "object_pose_tracking/sh_sam2/sh_choco_mask_sam2.npy", mmap_mode="r")
    if not np.array_equal(sh_frames - 5, frames):
        raise ValueError("MH/SH frame mapping differs from the verified +5 offset")

    manifest = json.loads((PILOT / "inputs/manifest.json").read_text(encoding="utf-8"))
    mh = manifest["calibration"]["intrinsics_by_view"]["MH"]
    k = np.asarray(mh["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(mh["distortion_k1_k2_p1_p2_k3"], dtype=np.float64)
    height, width = 720, 1280
    map_x, map_y = undistortion_maps(k, distortion, width=width, height=height)

    mesh, _ = load_canonical_mesh(PILOT / "spar3d/mesh.glb")
    renderer = DepthPairRenderer(mesh, k, width, height)

    overlay_root = PROC / "overlay_processor"
    robot_rgb_all = np.load(overlay_root / "robot_rgb.npy", mmap_mode="r")
    robot_depth_all = np.load(overlay_root / "robot_depth.npy", mmap_mode="r")
    robot_mask_all = np.load(overlay_root / "robot_mask.npy", mmap_mode="r")
    hand_mask_all = np.load(overlay_root / "robot_hand_mask.npy", mmap_mode="r")
    labels_all = np.load(overlay_root / "robot_finger_labels.npy", mmap_mode="r")
    support_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_amodal.npy", mmap_mode="r")
    restore_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_observed_clean.npy", mmap_mode="r")
    current_all = np.load(PROC / "overlay_best_inpaint_barrier/occluded_hand_mask.npy", mmap_mode="r")
    haco_dual_all = np.load(PROC / "overlay_haco_dual/occluded_finger_mask.npy", mmap_mode="r")
    background_cap = cv2.VideoCapture(str(PROC / "object_completion_dual_haco_e2fgvi/video_object_completed.mp4"))

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "choco_pose_mesh_xhand_dual_camera_comparison.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1920, 832))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create {out_path}")

    counts = []
    try:
        for local, (mh_frame, sh_frame, mono_pose, dual_pose) in enumerate(zip(frames, sh_frames, mono_poses, dual_poses, strict=True)):
            frame = int(mh_frame)
            raw = cv2.imread(str(DATASET / f"camera_2/rgb/rgb_frame{frame:06d}.jpg"), cv2.IMREAD_COLOR)
            raw_u = _remap_image(raw, map_x, map_y)
            background = read_video_frame(background_cap, frame)
            background_u = _remap_image(background, map_x, map_y)
            support = _remap_mask(support_all[frame], map_x, map_y)
            restore = _remap_mask(restore_all[frame], map_x, map_y)
            base = restore_raw_object_pixels(background_u, raw_u, restore)

            resized = resize_overlay_frame(robot_rgb_all[frame], robot_depth_all[frame], robot_mask_all[frame], hand_mask_all[frame], labels_all[frame], width=width, height=height)
            robot_rgb, robot_depth, robot_mask, hand_mask, labels = resized
            robot_rgb = _remap_image(robot_rgb, map_x, map_y)
            robot_depth = weighted_remap_depth(robot_depth, map_x, map_y)
            robot_mask = _remap_mask(robot_mask, map_x, map_y)
            hand_mask = _remap_mask(hand_mask, map_x, map_y)
            labels = _remap_labels(labels, map_x, map_y)
            current = _remap_mask(current_all[frame], map_x, map_y)
            haco_dual = _remap_mask(haco_dual_all[frame], map_x, map_y)

            mono_front, mono_back, mono_mesh, _ = renderer.render(mono_pose)
            dual_front, dual_back, dual_mesh, _ = renderer.render(dual_pose)
            mono_support = mono_mesh & support
            dual_support = dual_mesh & support
            mono_masks = build_static_occlusion_masks(hand_mask=hand_mask, finger_labels=labels, robot_depth=robot_depth, object_support_mask=support, mesh_mask=mono_support, front_depth=np.where(mono_support, mono_front, 0), back_depth=np.where(mono_support, mono_back, 0), current_mask=current, contact_baseline_mask=haco_dual, thumb_shell_m=0.01958, finger_shell_m=0.01465, palm_shell_m=0.015, spatial_close_radius_px=3, spatial_front_slack_m=0.003)
            dual_masks = build_static_occlusion_masks(hand_mask=hand_mask, finger_labels=labels, robot_depth=robot_depth, object_support_mask=support, mesh_mask=dual_support, front_depth=np.where(dual_support, dual_front, 0), back_depth=np.where(dual_support, dual_back, 0), current_mask=current, contact_baseline_mask=haco_dual, thumb_shell_m=0.01958, finger_shell_m=0.01465, palm_shell_m=0.015, spatial_close_radius_px=3, spatial_front_slack_m=0.003)
            dual_increment = int((dual_masks["spar_volume_filter"] & ~mono_masks["spar_volume_filter"]).sum())
            dual_removed = int((mono_masks["spar_volume_filter"] & ~dual_masks["spar_volume_filter"]).sum())

            images = [
                composite(base, robot_rgb, robot_mask, hand_mask, current),
                composite(base, robot_rgb, robot_mask, hand_mask, mono_masks["spar_front"]),
                composite(base, robot_rgb, robot_mask, hand_mask, dual_masks["spar_front"]),
                composite(base, robot_rgb, robot_mask, hand_mask, mono_masks["spar_volume_filter"]),
                composite(base, robot_rgb, robot_mask, hand_mask, dual_masks["spar_volume_filter"]),
            ]
            sh_image = cv2.imread(str(DATASET / f"camera_1/rgb/rgb_frame{int(sh_frame):06d}.jpg"), cv2.IMREAD_COLOR)
            sh_mask = np.asarray(sh_masks[local], dtype=bool)
            sh_target = amodalize_box_mask(sh_mask)
            target_contours, _ = cv2.findContours(sh_target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            projected_contours, _ = cv2.findContours(np.asarray(sh_reprojected[local], np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sh_image, target_contours, -1, (0, 0, 255), 3)
            cv2.drawContours(sh_image, projected_contours, -1, (0, 255, 255), 3)
            panels = [
                title_panel(images[0], "1 Existing 2.5D barrier", f"MH {frame} | existing HaCo/inpainting result"),
                title_panel(images[1], "2 MH-only pose: front-Z", "single-camera mesh pose"),
                title_panel(images[2], "3 MH+SH joint pose: front-Z", "both silhouettes used in pose loss"),
                title_panel(images[3], "4 MH-only pose: volume", "front/back + XHand thickness + filter"),
                title_panel(images[4], "5 MH+SH joint pose: volume", f"vs MH-only: +{dual_increment} / -{dual_removed} hidden px"),
                title_panel(sh_image, "6 SH joint constraint", f"SH {int(sh_frame)} | red=target, yellow=joint reprojection"),
            ]
            grid = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
            writer.write(grid)
            counts.append({"mh_frame": frame, "sh_frame": int(sh_frame), "current_hidden_px": int(current.sum()), "mono_pose_hidden_px": int(mono_masks['spar_volume_filter'].sum()), "joint_pose_hidden_px": int(dual_masks['spar_volume_filter'].sum()), "joint_added_hidden_px": dual_increment, "joint_removed_hidden_px": dual_removed})
            print(f"frame={frame} joint_added={dual_increment} joint_removed={dual_removed}", flush=True)
    finally:
        writer.release()
        background_cap.release()
        try:
            renderer.close()
        except Exception as exc:
            print(f"warning: EGL cleanup after completed render: {exc}", file=sys.stderr)

    report = {
        "schema_version": 1,
        "kind": "actual_mh_sh_joint_pose_xhand_comparison",
        "actual_dual_camera_optimization": True,
        "output": str(out_path.resolve()),
        "frames": [int(v) for v in frames],
        "panels": ["existing 2.5D barrier", "MH-only pose front-Z", "MH+SH joint pose front-Z", "MH-only pose front/back volume + XHand shell", "MH+SH joint pose front/back volume + XHand shell", "SH target and joint mesh reprojection"],
        "limitations": ["SH is used in the joint pose loss, but the checker-square length is missing, so the baseline scale was inferred from silhouette agreement and is not metric ground truth.", "SPAR3D geometry and HaWoR-anchored monocular depth are proxy estimates.", "The volume rule suppresses visually emerging robot pixels; it does not solve physical mesh-mesh collision."],
        "dual_camera_result": {"total_joint_added_hidden_pixels": int(sum(item["joint_added_hidden_px"] for item in counts)), "total_joint_removed_hidden_pixels": int(sum(item["joint_removed_hidden_px"] for item in counts)), "interpretation": "SH silhouette is included directly in pose optimization; changed pixels are caused by the joint pose rather than merely loading an SH-labelled mask."},
        "per_frame": counts,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
