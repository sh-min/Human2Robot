#!/usr/bin/env python3
"""Pilot the CVPR 2019 ObMan joint hand/object model on selected video frames.

The official AtlasNet object and MANO hand are inferred jointly.  The predicted
hand is then similarity-aligned to the existing camera-space HaWoR MANO mesh,
which transfers the jointly predicted object into the current camera frame.
Object contact surface vertices are defined by the paper's nearest hand/object
mesh distance rule; hand vertices inside the closed object are reported as
penetration separately.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
OBMAN = REPO / "third_party/obman_train"
MANOPTH = REPO / "third_party/manopth"


def similarity_align(source: np.ndarray, target: np.ndarray):
    """Return scale, row-vector rotation, translation for source -> target."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    rotated = source_zero @ rotation
    scale = float(
        np.sum(rotated * target_zero) / np.sum(source_zero * source_zero)
    )
    translation = target_center - scale * source_center @ rotation
    return scale, rotation, translation


def project(points: np.ndarray, focal: float, width: int, height: int):
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-4)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    uv[valid, 0] = focal * points[valid, 0] / points[valid, 2] + width / 2
    uv[valid, 1] = focal * points[valid, 1] / points[valid, 2] + height / 2
    return uv, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--video_output", type=Path, default=None)
    parser.add_argument("--save_frame_assets", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OBMAN / "release_models/obman/checkpoint.pth.tar",
    )
    parser.add_argument(
        "--contact_threshold_mm",
        type=float,
        default=10.0,
        help="Official checkpoint training threshold (default: 10mm).",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=2.0,
        help="Square crop side relative to the projected HaWoR hand extent.",
    )
    args = parser.parse_args()

    episode = args.episode.resolve()
    output_dir = args.output_dir.resolve()
    video_output = (
        args.video_output.resolve() if args.video_output is not None else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths = sorted((episode / "rgb").glob("*.jpg"))
    if args.all_frames:
        if args.frames is not None:
            parser.error("choose either --all_frames or --frames")
        frame_indices = list(range(len(rgb_paths)))
    elif args.frames is not None:
        frame_indices = args.frames
    else:
        parser.error("one of --all_frames or --frames is required")
    retarget_path = episode / "rgb_hawor/retarget_input.npz"
    if not args.checkpoint.is_file() or not retarget_path.is_file():
        raise FileNotFoundError("official checkpoint or HaWoR input is missing")

    sys.path[:0] = [str(MANOPTH), str(OBMAN)]
    previous_cwd = Path.cwd()
    os.chdir(OBMAN)
    try:
        from handobjectdatasets.queries import BaseQueries, TransQueries
        from mano_train.demo.preprocess import prepare_input
        from mano_train.netscripts.reload import reload_model

        with (args.checkpoint.parent / "opt.pkl").open("rb") as handle:
            options = pickle.load(handle)
        model = reload_model(
            str(args.checkpoint),
            options,
            mano_root=str(OBMAN / "misc/mano"),
        )
        model.eval()

        reports = []
        sequence_vertices = []
        sequence_contact = []
        sequence_distances = []
        sequence_penetration = []
        sequence_indices = []
        video_writer = None
        with np.load(retarget_path) as retarget, torch.inference_mode():
            focal = float(retarget["img_focal"])
            hand_all = np.asarray(retarget["verts_left"], dtype=np.float64)
            valid_all = np.asarray(retarget["valid"][0], dtype=bool)
            for frame_index in frame_indices:
                if not 0 <= frame_index < len(rgb_paths):
                    raise IndexError(frame_index)
                if not valid_all[frame_index]:
                    print(f"[skip] frame {frame_index}: no valid left hand")
                    continue
                bgr = cv2.imread(str(rgb_paths[frame_index]))
                if bgr is None:
                    raise RuntimeError(f"could not read {rgb_paths[frame_index]}")
                height, width = bgr.shape[:2]
                target_hand = hand_all[frame_index]
                hand_uv, hand_uv_valid = project(
                    target_hand, focal, width, height
                )
                visible_uv = hand_uv[hand_uv_valid]
                if len(visible_uv) < 50:
                    print(f"[skip] frame {frame_index}: insufficient hand projection")
                    continue
                uv_min = visible_uv.min(axis=0)
                uv_max = visible_uv.max(axis=0)
                crop_center = 0.5 * (uv_min + uv_max)
                crop_side = max(64, int(np.ceil(
                    args.crop_scale * float(np.max(uv_max - uv_min))
                )))
                # Official demo expects a roughly centered, large hand.  A
                # HaWoR-guided crop preserves that assumption while leaving a
                # one-hand-width margin for the manipulated object.
                square = cv2.getRectSubPix(
                    bgr, (crop_side, crop_side), tuple(crop_center)
                )
                square = cv2.resize(square, (256, 256))
                image_tensor = prepare_input(square, flip_left_right=False)
                sample = {
                    TransQueries.images: image_tensor,
                    BaseQueries.sides: ["left"],
                    TransQueries.joints3d: image_tensor.new_ones((1, 21, 3)),
                    TransQueries.objpoints3d: image_tensor.new_ones((1, 600, 3)),
                    "root": "wrist",
                }
                _, result, _ = model.forward(sample, no_loss=True)
                pred_hand = result["verts"][0].detach().cpu().numpy().astype(np.float64)
                pred_object = (
                    result["objpoints3d"][0].detach().cpu().numpy().astype(np.float64)
                )
                object_faces = np.asarray(result["objfaces"], dtype=np.int64)

                scale, rotation, translation = similarity_align(pred_hand, target_hand)
                aligned_hand = scale * pred_hand @ rotation + translation
                aligned_object = scale * pred_object @ rotation + translation
                alignment_rmse = float(
                    np.sqrt(np.mean(np.sum((aligned_hand - target_hand) ** 2, axis=1)))
                )

                hand_tree = cKDTree(target_hand)
                object_distance, _ = hand_tree.query(aligned_object, k=1)
                threshold_m = args.contact_threshold_mm / 1000.0
                object_contact = object_distance <= threshold_m
                object_mesh = trimesh.Trimesh(
                    vertices=aligned_object,
                    faces=object_faces,
                    process=False,
                )
                penetration = np.zeros(len(target_hand), dtype=bool)
                try:
                    penetration = object_mesh.contains(target_hand)
                except (ModuleNotFoundError, ImportError):
                    pass

                uv_object, valid_object = project(
                    aligned_object, focal, width, height
                )
                visualization = bgr.copy()
                for point in uv_object[valid_object & ~object_contact][::3]:
                    cv2.circle(
                        visualization,
                        tuple(np.rint(point).astype(int)),
                        1, (180, 180, 180), -1, cv2.LINE_AA,
                    )
                for point in uv_object[valid_object & object_contact]:
                    cv2.circle(
                        visualization,
                        tuple(np.rint(point).astype(int)),
                        4, (0, 0, 255), -1, cv2.LINE_AA,
                    )
                cv2.putText(
                    visualization,
                    "ObMan object mesh: gray | contact surface: red",
                    (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    visualization,
                    "ObMan object mesh: gray | contact surface: red",
                    (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (25, 25, 25), 1, cv2.LINE_AA,
                )

                stem = f"frame_{frame_index:06d}"
                if args.save_frame_assets:
                    cv2.imwrite(
                        str(output_dir / f"{stem}_contact.png"), visualization
                    )
                    object_mesh.export(
                        output_dir / f"{stem}_object_aligned.obj"
                    )
                    np.savez_compressed(
                        output_dir / f"{stem}_contact.npz",
                        object_vertices=aligned_object.astype(np.float32),
                        object_faces=object_faces.astype(np.int32),
                        object_contact=object_contact,
                        object_hand_distance_m=object_distance.astype(np.float32),
                        hand_vertices=target_hand.astype(np.float32),
                        hand_penetration=penetration,
                    )
                if video_output is not None:
                    if video_writer is None:
                        video_output.parent.mkdir(parents=True, exist_ok=True)
                        video_writer = cv2.VideoWriter(
                            str(video_output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            30.0,
                            (width, height),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(
                                f"could not open {video_output}"
                            )
                    video_writer.write(visualization)
                sequence_indices.append(frame_index)
                sequence_vertices.append(aligned_object.astype(np.float32))
                sequence_contact.append(object_contact)
                sequence_distances.append(object_distance.astype(np.float32))
                sequence_penetration.append(penetration)
                frame_report = {
                    "frame_index": frame_index,
                    "source": str(rgb_paths[frame_index]),
                    "alignment_rmse_mm": alignment_rmse * 1000.0,
                    "object_contact_vertices": int(object_contact.sum()),
                    "object_vertices": int(len(aligned_object)),
                    "hand_penetrating_vertices": int(penetration.sum()),
                    "contact_threshold_mm": args.contact_threshold_mm,
                    "crop_side_px": crop_side,
                }
                reports.append(frame_report)
                if len(frame_indices) <= 20 or (len(reports) % 50 == 0):
                    print(f"[ok] {frame_report}")
        if video_writer is not None:
            video_writer.release()
    finally:
        os.chdir(previous_cwd)

    (output_dir / "report.json").write_text(json.dumps({
        "method": "Hasson_et_al_CVPR_2019_joint_reconstruction_contact_pilot",
        "official_repository": "https://github.com/hassony2/obman_train",
        "official_checkpoint": str(args.checkpoint.resolve()),
        "contact_definition": "object vertices within threshold of aligned hand mesh",
        "frames": reports,
        "limitations": [
            "The 2019 AtlasNet object topology is sphere-like and category agnostic.",
            "A per-frame similarity alignment transfers the official joint prediction to HaWoR camera coordinates.",
            "This pilot does not yet temporally track or fuse the inferred object mesh.",
        ],
    }, indent=2))
    if sequence_indices:
        np.savez_compressed(
            output_dir / "contact_sequence.npz",
            frame_indices=np.asarray(sequence_indices, dtype=np.int32),
            object_vertices=np.stack(sequence_vertices),
            object_faces=object_faces.astype(np.int32),
            object_contact=np.stack(sequence_contact),
            object_hand_distance_m=np.stack(sequence_distances),
            hand_penetration=np.stack(sequence_penetration),
        )


if __name__ == "__main__":
    main()
