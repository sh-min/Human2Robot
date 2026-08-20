"""Complete object pixels hidden by the source human before background inpainting.

The visible SAM mask of a grasped object has finger-shaped holes.  Sending
those holes to the background inpainter makes the object look hollow.  This
stage builds a conservative amodal silhouette from the visible object, limits
new pixels to the source-human occlusion, and fills them from the nearest real
object texture in the same frame.  Consequently it cannot create a second
object on exposed tabletop.

The sponge can be supplied as a separate SAM2 track seeded by a clean box at
five seconds.  Other objects are separated from the merged interaction track
using their configured seed component.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _component(mask: np.ndarray, previous: np.ndarray | None,
               seed: tuple[float, float] | None = None) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    score = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    score[0] = -np.inf
    if previous is not None and previous.any():
        ids, overlap = np.unique(labels[previous], return_counts=True)
        for label, value in zip(ids, overlap):
            if label:
                score[label] += 1000.0 * float(value)
    if seed is not None:
        x, y = seed
        label = labels[
            int(np.clip(round(y), 0, labels.shape[0] - 1)),
            int(np.clip(round(x), 0, labels.shape[1] - 1)),
        ]
        if label:
            score[label] += 1e12
    return labels == int(np.argmax(score))


def _compact_track(merged: np.ndarray, segment: dict) -> tuple[int, np.ndarray]:
    """Return only the configured interval instead of allocating a full T mask."""
    start, end, seed = (int(segment[key]) for key in
                        ("start_frame", "end_frame", "seed_frame"))
    track = np.zeros((end - start + 1,) + merged.shape[1:], dtype=bool)
    points = segment.get("positive_points") or []
    seed_point = tuple(points[0]) if points else None
    current = _component(merged[seed], None, seed_point)
    track[seed - start] = current
    previous = current
    for frame_index in range(seed + 1, end + 1):
        current = _component(merged[frame_index], previous)
        track[frame_index - start] = current
        if current.any():
            previous = current
    previous = track[seed - start]
    for frame_index in range(seed - 1, start - 1, -1):
        current = _component(merged[frame_index], previous)
        track[frame_index - start] = current
        if current.any():
            previous = current
    return start, track


def _amodal_hull(visible: np.ndarray, close_radius: int,
                 max_hull_ratio: float) -> np.ndarray:
    current = np.asarray(visible, dtype=np.uint8)
    if close_radius > 0:
        current = cv2.morphologyEx(
            current, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2
            ),
        )
    points = cv2.findNonZero(current)
    if points is None or len(points) < 3:
        return current.astype(bool)
    hull_points = cv2.convexHull(points)
    hull = np.zeros_like(current)
    cv2.fillConvexPoly(hull, hull_points, 1)
    visible_area = max(1, int(current.sum()))
    if int(hull.sum()) > max_hull_ratio * visible_area:
        # A leaked component can span a very large polygon.  Falling back to a
        # closed modal mask is safer than painting an object onto the table.
        return current.astype(bool)
    return hull.astype(bool)


def _nearest_texture(frame: np.ndarray, visible: np.ndarray,
                     missing: np.ndarray) -> np.ndarray:
    """Predict missing RGB from real object pixels without sampling skin/table."""
    result = np.zeros_like(frame)
    support = visible | missing
    ys, xs = np.nonzero(support)
    if not len(xs) or not visible.any():
        return result
    padding = 4
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(frame.shape[1], int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(frame.shape[0], int(ys.max()) + padding + 1)
    visible_crop = visible[y0:y1, x0:x1]
    missing_crop = missing[y0:y1, x0:x1]
    _, indices = distance_transform_edt(
        ~visible_crop, return_distances=True, return_indices=True
    )
    source = frame[y0:y1, x0:x1]
    nearest = source[indices[0], indices[1]]
    # A light spatial blend removes nearest-neighbour streaks but retains the
    # source object's colour/print rather than inventing tabletop pixels.
    smooth = cv2.GaussianBlur(nearest, (0, 0), 0.75)
    predicted = cv2.addWeighted(nearest, 0.82, smooth, 0.18, 0.0)
    crop = result[y0:y1, x0:x1]
    crop[missing_crop] = predicted[missing_crop]
    return result


def _outline(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2
    )
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)


def _project_joint(joints: np.ndarray | None, valid: np.ndarray | None,
                   frame_idx: int, focal: float | None,
                   width: int, height: int) -> tuple[float, float] | None:
    if (joints is None or valid is None or focal is None or
            frame_idx >= len(joints) or not valid[frame_idx]):
        return None
    xyz = np.asarray(joints[frame_idx, 0], dtype=np.float32)
    if xyz[2] <= 1e-3:
        return None
    return (float(focal * xyz[0] / xyz[2] + width / 2.0),
            float(focal * xyz[1] / xyz[2] + height / 2.0))


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)


def _shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate reference RGB with the same motion used for its mask."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame, matrix, (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"video decode stopped at reference frame {frame_index}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--merged_mask", type=Path, required=True)
    parser.add_argument("--base_object_mask", type=Path, required=True)
    parser.add_argument("--human_mask", type=Path, required=True)
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--sponge_mask", type=Path, default=None)
    parser.add_argument(
        "--only_object", choices=("navy_mug", "red_snack_box",
                                   "green_snack_box", "mint_container",
                                   "sponge"), default=None,
        help="Restrict completion to one object, useful for a conservative "
             "second pass over an otherwise finished object layer.",
    )
    parser.add_argument("--robot_mask", type=Path, default=None)
    parser.add_argument("--human_dilate", type=int, default=2)
    parser.add_argument("--robot_erode", type=int, default=5)
    parser.add_argument("--close_radius", type=int, default=5)
    parser.add_argument(
        "--sponge_dilate", type=int, default=0,
        help="Expand the sponge amodal support by this many pixels, but only "
             "inside the source-human occlusion. This conservatively restores "
             "a little more sponge at close hand contact without painting it "
             "onto exposed tabletop.",
    )
    parser.add_argument("--max_hull_ratio", type=float, default=3.2)
    parser.add_argument("--min_visible_pixels", type=int, default=80)
    parser.add_argument("--output_video", type=Path, required=True)
    parser.add_argument("--output_mask", type=Path, required=True)
    parser.add_argument("--prediction_mask", type=Path, default=None)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--hawor_npz", type=Path, default=None)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--temporal_reference_ratio", type=float, default=0.55,
                        help="When visible area falls below this fraction of a "
                             "nearby clean frame, use that frame as amodal "
                             "support inside the human occlusion only.")
    parser.add_argument("--temporal_search_frames", type=int, default=20)
    args = parser.parse_args()

    merged = np.load(args.merged_mask, mmap_mode="r")
    base_object = np.load(args.base_object_mask, mmap_mode="r")
    human = np.load(args.human_mask, mmap_mode="r")
    sponge = (np.load(args.sponge_mask, mmap_mode="r")
              if args.sponge_mask is not None else None)
    robot = (np.load(args.robot_mask, mmap_mode="r")
             if args.robot_mask is not None else None)
    hand_joints = hand_valid = None
    hand_focal = None
    if args.hawor_npz is not None:
        hawor = np.load(args.hawor_npz)
        side_idx = 0 if args.side == "left" else 1
        hand_joints = hawor[f"joints_{args.side}"]
        hand_valid = hawor["valid"][side_idx]
        hand_focal = float(hawor["img_focal"])
    payload = json.loads(args.segments_json.read_text(encoding="utf-8"))
    segments = payload["segments"]
    if args.only_object is not None:
        segments = [item for item in segments
                    if item["name"] == args.only_object]

    tracks: list[dict] = []
    for segment in segments:
        name = segment["name"]
        if name == "sponge" and sponge is not None:
            tracks.append({
                "name": name, "start": 0,
                "end": min(len(sponge), len(merged)) - 1,
                "track": sponge,
            })
            print("[track] sponge: separate 5-second box-prompt SAM2 mask")
        else:
            start, track = _compact_track(merged, segment)
            tracks.append({
                "name": name, "start": start,
                "end": start + len(track) - 1, "track": track,
                "area": track.sum(axis=(1, 2)),
            })
            print(
                f"[track] {name}: {start}..{start + len(track) - 1}, "
                f"pixels={int(track.sum())}"
            )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    reference_cap = cv2.VideoCapture(str(args.video))
    if not reference_cap.isOpened():
        raise RuntimeError(args.video)
    reference_cache: dict[int, np.ndarray] = {}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_count = min(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(merged),
        len(base_object), len(human),
    )
    if robot is not None:
        frame_count = min(frame_count, len(robot))

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"FFV1"), fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output_video}")
    preview_writer = None
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview_writer = cv2.VideoWriter(
            str(args.preview), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )

    output_mask = np.lib.format.open_memmap(
        args.output_mask, mode="w+", dtype=bool,
        shape=(frame_count, height, width),
    )
    predicted_output = None
    if args.prediction_mask is not None:
        args.prediction_mask.parent.mkdir(parents=True, exist_ok=True)
        predicted_output = np.lib.format.open_memmap(
            args.prediction_mask, mode="w+", dtype=bool,
            shape=(frame_count, height, width),
        )

    human_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(0, args.human_dilate) + 1,) * 2
    )
    robot_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(0, args.robot_erode) + 1,) * 2
    )
    sponge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(0, args.sponge_dilate) + 1,) * 2
    )
    per_object_pixels = {layer["name"]: 0 for layer in tracks}
    per_object_temporal = {layer["name"]: 0 for layer in tracks}
    remaining_background_pixels = 0

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"video decode stopped at frame {frame_index}")
        completed = frame.copy()
        current_object = np.asarray(base_object[frame_index], dtype=bool).copy()
        current_prediction = np.zeros((height, width), dtype=bool)
        current_visible_all = np.zeros((height, width), dtype=bool)
        occlusion = cv2.dilate(
            np.asarray(human[frame_index], dtype=np.uint8),
            human_kernel, iterations=1,
        ).astype(bool)

        for layer in tracks:
            if not layer["start"] <= frame_index <= layer["end"]:
                continue
            visible = np.asarray(
                layer["track"][frame_index - layer["start"]], dtype=bool
            )
            if int(visible.sum()) < args.min_visible_pixels:
                continue
            support = visible
            texture_frame = frame
            clean_visible = visible & ~occlusion
            texture_visible = clean_visible if clean_visible.any() else visible
            temporal_texture = False
            local = frame_index - layer["start"]
            areas = layer.get("area")
            if (areas is not None and args.temporal_reference_ratio > 0 and
                    args.temporal_search_frames > 0):
                lo = max(0, local - args.temporal_search_frames)
                hi = min(len(areas), local + args.temporal_search_frames + 1)
                reference_local = lo + int(np.argmax(areas[lo:hi]))
                reference_area = int(areas[reference_local])
                if (reference_area >= args.min_visible_pixels and
                        int(visible.sum()) <
                        args.temporal_reference_ratio * reference_area):
                    reference_frame = layer["start"] + reference_local
                    current_wrist = _project_joint(
                        hand_joints, hand_valid, frame_index, hand_focal,
                        width, height)
                    reference_wrist = _project_joint(
                        hand_joints, hand_valid, reference_frame, hand_focal,
                        width, height)
                    reference = np.asarray(
                        layer["track"][reference_local], dtype=bool)
                    reference_occlusion = cv2.dilate(
                        np.asarray(human[reference_frame], dtype=np.uint8),
                        human_kernel, iterations=1,
                    ).astype(bool)
                    reference_texture_visible = reference & ~reference_occlusion
                    if reference_frame not in reference_cache:
                        # Nearby collapse frames repeatedly select the same
                        # clean reference. A small cache avoids seeking for
                        # every frame without retaining the whole video.
                        if len(reference_cache) >= 8:
                            reference_cache.pop(next(iter(reference_cache)))
                        reference_cache[reference_frame] = _read_frame_at(
                            reference_cap, reference_frame)
                    reference_rgb = reference_cache[reference_frame]
                    if current_wrist is not None and reference_wrist is not None:
                        dx = current_wrist[0] - reference_wrist[0]
                        dy = current_wrist[1] - reference_wrist[1]
                        reference = _shift_mask(
                            reference, dx, dy)
                        reference_texture_visible = _shift_mask(
                            reference_texture_visible, dx, dy)
                        reference_rgb = _shift_frame(reference_rgb, dx, dy)
                    support = visible | reference
                    # A collapsed current mask is commonly dominated by skin
                    # at contact. Sample texture from the same clean temporal
                    # reference that supplied the amodal silhouette.
                    texture_frame = reference_rgb
                    texture_visible = (
                        reference_texture_visible
                        if reference_texture_visible.any() else reference
                    )
                    temporal_texture = True
                    per_object_temporal[layer["name"]] += 1
            amodal = _amodal_hull(
                support, args.close_radius, args.max_hull_ratio
            )
            if layer["name"] == "sponge" and args.sponge_dilate > 0:
                amodal = cv2.dilate(
                    amodal.astype(np.uint8), sponge_kernel, iterations=1
                ).astype(bool)
            missing = amodal & occlusion & ~visible & ~current_object
            if not missing.any():
                current_visible_all |= visible
                current_object |= visible
                continue
            paint = missing
            if temporal_texture:
                # Pixels already labelled as object can still be skin when
                # SAM absorbs the grasping hand. Replace every object pixel
                # inside the human occlusion from the clean reference too.
                paint = paint | (visible & occlusion)
            texture = _nearest_texture(
                texture_frame, texture_visible, paint)
            completed[paint] = texture[paint]
            # Repainted SAM pixels are synthesized too, even when they were
            # already present in the modal mask. Downstream contact layering
            # must not mistake them for directly observed object texture.
            current_prediction |= paint
            current_visible_all |= visible
            current_object |= visible | missing
            per_object_pixels[layer["name"]] += int(missing.sum())

        output_mask[frame_index] = current_object
        if predicted_output is not None:
            predicted_output[frame_index] = current_prediction
        writer.write(completed)

        if preview_writer is not None:
            preview = completed.copy()
            if robot is not None:
                robot_core = cv2.erode(
                    np.asarray(robot[frame_index], dtype=np.uint8),
                    robot_kernel, iterations=1,
                ).astype(bool)
                residual = (
                    np.asarray(human[frame_index], dtype=bool)
                    & ~robot_core & ~current_object
                )
                remaining_background_pixels += int(residual.sum())
                preview[residual] = (
                    0.38 * preview[residual]
                    + 0.62 * np.array([25, 25, 235])
                ).astype(np.uint8)
            preview[_outline(current_visible_all, 1)] = (20, 225, 20)
            preview[_outline(current_prediction, 2)] = (25, 220, 245)
            preview_writer.write(preview)

        if (frame_index + 1) % 100 == 0:
            print(f"[frame] {frame_index + 1}/{frame_count}", flush=True)

    cap.release()
    reference_cap.release()
    writer.release()
    if preview_writer is not None:
        preview_writer.release()
    output_mask.flush()
    if predicted_output is not None:
        predicted_output.flush()
    for name, pixels in per_object_pixels.items():
        print(f"[completion] {name}: predicted={pixels} px")
        if per_object_temporal[name]:
            print(f"[temporal] {name}: reference used on "
                  f"{per_object_temporal[name]} frames")
    if robot is not None:
        print(
            f"[info] remaining true-background inpaint pixels="
            f"{remaining_background_pixels}"
        )
    print(f"[ok] wrote {args.output_video}")
    print(f"[ok] wrote {args.output_mask}")


if __name__ == "__main__":
    main()
