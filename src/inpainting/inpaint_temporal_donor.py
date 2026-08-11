"""Inpaint a residual human mask from real clean frames of a fixed-camera clip.

For every output frame, candidate donor frames are ranked by how little their
human/object exclusion masks overlap the requested fill.  Pixels are copied
from the best valid donor and only fall back to a supplied plate when no real
clean observation exists.  This preserves tabletop texture more sharply than
large generative fills and avoids the ghost average of a single global median
plate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--exclude_mask", type=Path, action="append", default=[],
        help="Per-frame human/object mask excluded from temporal donors; repeatable.",
    )
    parser.add_argument("--protect_mask", type=Path, default=None)
    parser.add_argument("--fallback_plate", type=Path, required=True)
    parser.add_argument("--donor_stride", type=int, default=4)
    parser.add_argument(
        "--pixel_selection", choices=("coherent", "cleanest"),
        default="coherent",
        help="coherent prefers a small set of whole donor frames; cleanest "
             "selects the brightest neutral valid donor per fill pixel.",
    )
    parser.add_argument(
        "--exclude_skin", action="store_true",
        help="Also reject conservative YCrCb skin-colour pixels from donors.",
    )
    parser.add_argument("--exclude_dilate", type=int, default=2)
    parser.add_argument("--protect_dilate", type=int, default=1)
    parser.add_argument("--feather_sigma", type=float, default=0.9)
    parser.add_argument("--color_match_ring", type=int, default=16)
    parser.add_argument("--max_donors", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    fill_masks = np.load(args.mask, mmap_mode="r")
    excludes = [np.load(path, mmap_mode="r") for path in args.exclude_mask]
    protect = (np.load(args.protect_mask, mmap_mode="r")
               if args.protect_mask is not None else None)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(fill_masks))
    for mask in excludes:
        frame_count = min(frame_count, len(mask))
    if protect is not None:
        frame_count = min(frame_count, len(protect))

    fallback = cv2.imread(str(args.fallback_plate), cv2.IMREAD_COLOR)
    if fallback is None:
        raise RuntimeError(f"cannot read fallback plate: {args.fallback_plate}")
    if fallback.shape[:2] != (height, width):
        fallback = cv2.resize(fallback, (width, height), cv2.INTER_LINEAR)

    donor_indices = list(range(0, frame_count, max(1, args.donor_stride)))
    if donor_indices[-1] != frame_count - 1:
        donor_indices.append(frame_count - 1)
    donor_set = set(donor_indices)
    donor_frames = []
    index = 0
    while index < frame_count:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"video decode stopped at frame {index}")
        if index in donor_set:
            donor_frames.append(frame)
        index += 1
    cap.release()
    donor_frames_array = np.stack(donor_frames)

    exclude_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.exclude_dilate) + 1,) * 2,
    )
    invalid_bank = np.zeros(
        (len(donor_indices), height, width), dtype=bool
    )
    for bank_index, frame_index in enumerate(donor_indices):
        invalid = np.zeros((height, width), dtype=np.uint8)
        for mask in excludes:
            invalid |= np.asarray(mask[frame_index], dtype=np.uint8)
        if args.exclude_skin:
            ycrcb = cv2.cvtColor(
                donor_frames_array[bank_index], cv2.COLOR_BGR2YCrCb
            )
            luminance, cr, cb = cv2.split(ycrcb)
            skin = (
                (luminance >= 45) & (cr >= 130) & (cr <= 182)
                & (cb >= 70) & (cb <= 142)
            )
            invalid |= skin.astype(np.uint8)
        if args.exclude_dilate > 0:
            invalid = cv2.dilate(invalid, exclude_kernel, iterations=1)
        invalid_bank[bank_index] = invalid.astype(bool)
    donor_indices_array = np.asarray(donor_indices, dtype=np.int32)
    print(
        f"[info] donors={len(donor_indices)}, stride={args.donor_stride}, "
        f"frames={frame_count}, {width}x{height}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"FFV1"), fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")
    preview_writer = None
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview_writer = cv2.VideoWriter(
            str(args.preview), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )

    protect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.protect_dilate) + 1,) * 2,
    )
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.color_match_ring) + 1,) * 2,
    )
    cap = cv2.VideoCapture(str(args.video))
    temporal_pixels = 0
    fallback_pixels = 0
    primary_donor_distances = []

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"video decode stopped at frame {frame_index}")
        fill = np.asarray(fill_masks[frame_index], dtype=bool).copy()
        protected = None
        if protect is not None:
            protected = cv2.dilate(
                np.asarray(protect[frame_index], dtype=np.uint8),
                protect_kernel, iterations=1,
            ).astype(bool)
            fill &= ~protected
        if not fill.any():
            writer.write(frame)
            if preview_writer is not None:
                preview_writer.write(frame)
            continue

        ys, xs = np.nonzero(fill)
        # Evaluate candidates only at requested pixels.  A slight time penalty
        # picks a nearby clean frame when several donors have equal overlap.
        overlap = invalid_bank[:, ys, xs].sum(axis=1).astype(np.float64)
        overlap += 0.002 * np.abs(donor_indices_array - frame_index)
        order = np.argsort(overlap)
        primary = int(order[0])
        primary_donor_distances.append(
            abs(int(donor_indices_array[primary]) - frame_index)
        )

        content = fallback.copy()
        unresolved = np.ones(len(xs), dtype=bool)
        used = 0
        if args.pixel_selection == "cleanest":
            values = donor_frames_array[:, ys, xs]
            values_float = values.astype(np.float32)
            luminance = (
                0.114 * values_float[..., 0]
                + 0.587 * values_float[..., 1]
                + 0.299 * values_float[..., 2]
            )
            chroma = values_float.max(axis=2) - values_float.min(axis=2)
            score = luminance - 1.25 * chroma
            score[invalid_bank[:, ys, xs]] = -np.inf
            best = np.argmax(score, axis=0)
            good = np.isfinite(score[best, np.arange(len(xs))])
            if good.any():
                content[ys[good], xs[good]] = values[
                    best[good], np.flatnonzero(good)
                ]
                temporal_pixels += int(good.sum())
                unresolved[good] = False
                used = int(len(np.unique(best[good])))
        else:
            for donor_index in order[:max(1, args.max_donors)]:
                valid = ~invalid_bank[donor_index, ys, xs]
                take = unresolved & valid
                if take.any():
                    content[ys[take], xs[take]] = donor_frames_array[
                        donor_index, ys[take], xs[take]
                    ]
                    temporal_pixels += int(take.sum())
                    unresolved[take] = False
                    used += 1
                if not unresolved.any():
                    break
        fallback_pixels += int(unresolved.sum())

        if args.color_match_ring > 0:
            ring = cv2.dilate(
                fill.astype(np.uint8), ring_kernel, iterations=1
            ).astype(bool) & ~fill
            if protected is not None:
                ring &= ~protected
            if int(ring.sum()) >= 100:
                primary_frame = donor_frames_array[primary]
                offset = np.median(
                    frame[ring].astype(np.float32)
                    - primary_frame[ring].astype(np.float32), axis=0,
                )
                offset = np.clip(offset, -24.0, 24.0)
                content_at_fill = np.clip(
                    content[fill].astype(np.float32) + offset, 0, 255
                ).astype(np.uint8)
                content[fill] = content_at_fill

        alpha = fill.astype(np.float32)
        if args.feather_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.feather_sigma)
        if protected is not None:
            alpha[protected] = 0.0
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        composed = (
            alpha * content.astype(np.float32)
            + (1.0 - alpha) * frame.astype(np.float32)
        )
        output = np.clip(composed, 0, 255).astype(np.uint8)
        writer.write(output)

        if preview_writer is not None:
            preview = output.copy()
            boundary = cv2.morphologyEx(
                fill.astype(np.uint8), cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ).astype(bool)
            preview[boundary] = (30, 220, 245)
            label = (
                f"donor {int(donor_indices_array[primary])} "
                f"({used} sources)"
            )
            cv2.putText(
                preview, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (20, 20, 20), 3, cv2.LINE_AA,
            )
            cv2.putText(
                preview, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (245, 245, 245), 1, cv2.LINE_AA,
            )
            preview_writer.write(preview)
        if (frame_index + 1) % 100 == 0:
            print(f"[frame] {frame_index + 1}/{frame_count}", flush=True)

    cap.release()
    writer.release()
    if preview_writer is not None:
        preview_writer.release()
    mean_distance = (
        float(np.mean(primary_donor_distances))
        if primary_donor_distances else 0.0
    )
    print(
        f"[info] temporal={temporal_pixels} px, fallback={fallback_pixels} px, "
        f"mean primary donor distance={mean_distance:.1f} frames"
    )
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
