#!/usr/bin/env python3
"""Transfer dense MANO visibility anchors to corresponding XHand pixels.

The 778 MANO vertices are visibility anchors.  Within each finger, human
vertices and rendered robot pixels are expressed in normalized longitudinal
and lateral coordinates.  A local k-NN field transfers object-occlusion state
without relying on direct screen-space overlap or a bounding-box warp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
PART_GROUPS = ((13, 14, 15), (1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9))


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def vertex_finger_labels(parts: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(parts), dtype=np.uint8)
    for label, group in enumerate(PART_GROUPS, start=1):
        labels[np.isin(parts, group)] = label
    return labels


def project(vertices: np.ndarray, focal: float, width: int, height: int):
    valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 1e-5)
    uv = np.full((len(vertices), 2), np.nan, np.float32)
    uv[valid, 0] = focal * vertices[valid, 0] / vertices[valid, 2] + width / 2
    uv[valid, 1] = focal * vertices[valid, 1] / vertices[valid, 2] + height / 2
    return uv, valid


def canonical_human(
    points: np.ndarray, parts: np.ndarray, group: tuple[int, int, int]
) -> np.ndarray | None:
    proximal = points[parts == group[0]]
    distal = points[parts == group[-1]]
    if len(proximal) < 2 or len(distal) < 2:
        return None
    base = np.nanmean(proximal, axis=0)
    tip = np.nanmean(distal, axis=0)
    axis = tip - base
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length < 1.0:
        return None
    axis /= length
    lateral_axis = np.array([-axis[1], axis[0]], np.float32)
    along = (points - base) @ axis
    lo, hi = np.percentile(along, (2, 98))
    scale = max(float(hi - lo), 1.0)
    s = (along - lo) / scale
    lateral = (points - base) @ lateral_axis
    lateral_scale = max(float(np.percentile(np.abs(lateral), 95)), 1.0)
    return np.stack((s, lateral / lateral_scale), axis=1).astype(np.float32)


def canonical_robot(
    coordinates_yx: np.ndarray, robot_center_xy: np.ndarray
) -> np.ndarray | None:
    if len(coordinates_yx) < 8:
        return None
    points = coordinates_yx[:, ::-1].astype(np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))].astype(np.float32)
    projection = centered @ axis
    endpoint_a = center + axis * float(projection.min())
    endpoint_b = center + axis * float(projection.max())
    # The endpoint closer to the full robot centroid is the finger base.
    if np.linalg.norm(endpoint_b - robot_center_xy) < np.linalg.norm(endpoint_a - robot_center_xy):
        axis = -axis
        projection = -projection
    lo, hi = np.percentile(projection, (1, 99))
    scale = max(float(hi - lo), 1.0)
    s = (projection - lo) / scale
    lateral_axis = np.array([-axis[1], axis[0]], np.float32)
    lateral = centered @ lateral_axis
    lateral_scale = max(float(np.percentile(np.abs(lateral), 95)), 1.0)
    return np.stack((s, lateral / lateral_scale), axis=1).astype(np.float32)


def transfer_knn(
    source_coordinates: np.ndarray,
    source_occluded: np.ndarray,
    target_coordinates: np.ndarray,
    neighbors: int,
    sigma: float,
) -> np.ndarray:
    distance = ((target_coordinates[:, None, :] - source_coordinates[None, :, :]) ** 2).sum(axis=2)
    k = min(neighbors, distance.shape[1])
    nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
    selected_distance = np.take_along_axis(distance, nearest, axis=1)
    weights = np.exp(-selected_distance / max(2.0 * sigma * sigma, 1e-6))
    values = source_occluded[nearest].astype(np.float32)
    return (weights * values).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean_plate", type=Path, required=True)
    parser.add_argument("--visible_human_mask", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument(
        "--observed_object_mask", type=Path, required=True,
        help="trusted modal object pixels that override nearby SAM hand support",
    )
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--finger_parts", type=Path, required=True)
    parser.add_argument("--robot_rgb", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument("--robot_finger_labels", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--visible_probe_px", type=int, default=3)
    parser.add_argument("--object_dilate_px", type=int, default=2)
    parser.add_argument("--visible_threshold", type=float, default=0.30)
    parser.add_argument("--transfer_threshold", type=float, default=0.50)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--knn_sigma", type=float, default=0.22)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visible_all = np.load(args.visible_human_mask, mmap_mode="r")
    object_all = np.load(args.object_mask, mmap_mode="r")
    observed_object_all = np.load(args.observed_object_mask, mmap_mode="r")
    robot_rgb = np.load(args.robot_rgb, mmap_mode="r")
    robot_mask = np.load(args.robot_mask, mmap_mode="r")
    robot_labels = np.load(args.robot_finger_labels, mmap_mode="r")
    parts = np.asarray(np.load(args.finger_parts), dtype=np.int32)
    vertex_labels = vertex_finger_labels(parts)
    with np.load(args.hawor_npz) as hawor:
        vertices_all = np.asarray(hawor[f"verts_{args.side}"], dtype=np.float32)
        focal_full = float(np.asarray(hawor["img_focal"]).item())
        valid_raw = np.asarray(hawor["valid"], dtype=bool)
        side_index = 0 if args.side == "left" else 1
        valid_all = valid_raw[side_index] if valid_raw.shape[0] == 2 else valid_raw[:, side_index]

    cap = cv2.VideoCapture(str(args.clean_plate))
    if not cap.isOpened():
        raise FileNotFoundError(args.clean_plate)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not (
        len(visible_all) == len(object_all) == len(observed_object_all)
        == len(vertices_all) == len(valid_all)
        == len(robot_rgb) == len(robot_mask) == len(robot_labels) == frames
    ):
        raise ValueError("frame count mismatch")
    low_h, low_w = robot_mask.shape[1:]
    focal_low = focal_full * (low_w / width)
    probe_kernel = (args.visible_probe_px * 2 + 1,) * 2
    object_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (args.object_dilate_px * 2 + 1,) * 2
    )
    hidden_all = np.zeros((frames, low_h, low_w), dtype=bool)
    hidden_counts = np.zeros(frames, np.int64)
    anchor_counts = np.zeros((frames, 5, 2), np.int32)

    final_writer = open_writer(
        args.output_dir / "video_dense_anchor_robot_visibility.mp4v.mp4",
        fps, (width, height),
    )
    panel_w, panel_h, header = width // 3, height // 3, 68
    compare_writer = open_writer(
        args.output_dir / "video_compare_dense_anchor_mapping.mp4v.mp4",
        fps, (panel_w * 3, panel_h + header),
    )

    try:
        for index in range(frames):
            ok, clean = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {index}")
            rmask_low = np.asarray(robot_mask[index], dtype=bool)
            labels_low = np.asarray(robot_labels[index], dtype=np.uint8)
            rrgb_low = np.asarray(robot_rgb[index])
            robot_yx = np.argwhere(rmask_low)
            robot_center_xy = (
                robot_yx[:, ::-1].mean(axis=0).astype(np.float32)
                if len(robot_yx) else np.array([low_w / 2, low_h / 2], np.float32)
            )
            visible_low = resize_mask(visible_all[index], low_w, low_h)
            object_low = resize_mask(object_all[index], low_w, low_h)
            object_low = cv2.dilate(object_low.astype(np.uint8), object_kernel).astype(bool)
            observed_object_low = resize_mask(
                observed_object_all[index], low_w, low_h
            )
            observed_object_low = cv2.dilate(
                observed_object_low.astype(np.uint8), object_kernel
            ).astype(bool)
            visible_fraction_map = cv2.boxFilter(
                visible_low.astype(np.float32), -1, probe_kernel,
                normalize=True, borderType=cv2.BORDER_CONSTANT,
            )
            hidden = np.zeros((low_h, low_w), dtype=bool)
            anchor_view = cv2.resize(clean, (low_w, low_h))

            if valid_all[index]:
                uv, valid_vertices = project(
                    vertices_all[index], focal_low, low_w, low_h
                )
                for finger_index, group in enumerate(PART_GROUPS):
                    label = finger_index + 1
                    selected = (
                        valid_vertices & (vertex_labels == label)
                        & (uv[:, 0] >= 0) & (uv[:, 0] < low_w)
                        & (uv[:, 1] >= 0) & (uv[:, 1] < low_h)
                    )
                    source_points = uv[selected]
                    source_parts = parts[selected]
                    target_yx = np.argwhere(labels_low == label)
                    if len(source_points) < 8 or len(target_yx) < 8:
                        continue
                    rounded = np.rint(source_points).astype(np.int32)
                    rounded[:, 0] = np.clip(rounded[:, 0], 0, low_w - 1)
                    rounded[:, 1] = np.clip(rounded[:, 1], 0, low_h - 1)
                    support = visible_fraction_map[rounded[:, 1], rounded[:, 0]]
                    on_object = object_low[rounded[:, 1], rounded[:, 0]]
                    on_observed_object = observed_object_low[
                        rounded[:, 1], rounded[:, 0]
                    ]
                    source_occluded = on_observed_object | (
                        (support < args.visible_threshold) & on_object
                    )
                    source_canonical = canonical_human(
                        source_points, source_parts, group
                    )
                    target_canonical = canonical_robot(target_yx, robot_center_xy)
                    if source_canonical is None or target_canonical is None:
                        continue
                    scores = transfer_knn(
                        source_canonical, source_occluded, target_canonical,
                        args.knn, args.knn_sigma,
                    )
                    selected_target = scores >= args.transfer_threshold
                    hidden[target_yx[selected_target, 0], target_yx[selected_target, 1]] = True
                    anchor_counts[index, finger_index] = (
                        len(source_points), int(source_occluded.sum())
                    )
                    for point, is_hidden in zip(rounded, source_occluded):
                        cv2.circle(
                            anchor_view, tuple(point), 1,
                            (0, 0, 255) if is_hidden else (0, 255, 0),
                            -1, cv2.LINE_8,
                        )

            hidden &= labels_low > 0
            hidden_all[index] = hidden
            hidden_counts[index] = int(hidden.sum())
            alpha_low = rmask_low.astype(np.float32)
            alpha_low[hidden] = 0.0
            alpha = cv2.resize(alpha_low, (width, height), interpolation=cv2.INTER_NEAREST)
            rrgb = cv2.resize(rrgb_low, (width, height), interpolation=cv2.INTER_LINEAR)
            simple_alpha = cv2.resize(
                rmask_low.astype(np.float32), (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            simple = np.clip(
                clean.astype(np.float32) * (1 - simple_alpha[..., None])
                + rrgb.astype(np.float32) * simple_alpha[..., None], 0, 255,
            ).astype(np.uint8)
            final = np.clip(
                clean.astype(np.float32) * (1 - alpha[..., None])
                + rrgb.astype(np.float32) * alpha[..., None], 0, 255,
            ).astype(np.uint8)
            final_writer.write(final)

            evidence = simple.copy()
            hidden_hi = cv2.resize(
                hidden.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            evidence[hidden_hi] = (
                evidence[hidden_hi].astype(np.float32) * 0.2
                + np.array([0, 0, 255], np.float32) * 0.8
            ).astype(np.uint8)
            panels = (simple, evidence, final)
            canvas = np.full((panel_h + header, panel_w * 3, 3), 22, np.uint8)
            for column, panel in enumerate(panels):
                canvas[header:, column * panel_w:(column + 1) * panel_w] = cv2.resize(
                    panel, (panel_w, panel_h)
                )
            names = ("1 Full robot hand", "2 Dense mapped invisible pixels", "3 Invisible parts removed")
            for column, name in enumerate(names):
                cv2.putText(
                    canvas, name, (column * panel_w + 12, 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (245, 245, 245), 2,
                    cv2.LINE_AA,
                )
            compare_writer.write(canvas)
            if (index + 1) % 100 == 0:
                print(f"[dense-map] {index + 1}/{frames}", flush=True)
    finally:
        cap.release()
        final_writer.release()
        compare_writer.release()

    np.save(args.output_dir / "dense_mapped_invisible_robot_mask.npy", hidden_all)
    np.save(args.output_dir / "dense_anchor_counts.npy", anchor_counts)
    report = {
        "schema_version": 1,
        "method": "778_MANO_visibility_anchors_to_XHand_finger_pixels",
        "frames": frames,
        "frames_with_removed_robot_pixels": int((hidden_counts > 0).sum()),
        "removed_robot_pixels_lowres": int(hidden_counts.sum()),
        "visible_threshold": args.visible_threshold,
        "transfer_threshold": args.transfer_threshold,
        "knn": args.knn,
        "knn_sigma": args.knn_sigma,
        "correspondence": "finger identity + normalized longitudinal/lateral coordinates",
        "occluded_anchor_rule": (
            "trusted observed object overrides nearby SAM hand support; otherwise "
            "not-supported-by-SAM AND inside completed object"
        ),
        "robot_arm_removed": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
