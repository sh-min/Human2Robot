"""Clean hand/arm removal for a fixed camera using a temporal background plate.

For a static camera, a masked temporal median is usually sharper and more
stable than generative video inpainting over a large forearm.  This script
samples the clip, excludes pixels marked as human in each sampled frame,
computes a full-resolution background plate in memory-bounded row blocks, and
feather-composites that plate only inside the requested hand/arm mask.

An optional manipulated-object mask is protected from replacement.  This keeps
visible object texture intact at contact while allowing the robot/object layer
compositor to restore the correct depth ordering afterward.
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protect_mask", type=Path, default=None)
    parser.add_argument("--plate_exclude_mask", type=Path, default=None,
                        help="Additional per-frame pixels excluded while estimating "
                             "the plate, typically manipulated objects.")
    parser.add_argument("--plate_exclude_dilate", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=4)
    parser.add_argument("--row_block", type=int, default=48)
    parser.add_argument("--plate_method",
                        choices=("cleanest", "brightest", "median"),
                        default="cleanest")
    parser.add_argument("--feather_sigma", type=float, default=2.0)
    parser.add_argument(
        "--blend_mode", choices=("alpha", "poisson"), default="alpha",
        help="alpha uses the optional Gaussian feather; poisson preserves a "
             "sharp plate texture while solving the boundary illumination.",
    )
    parser.add_argument(
        "--poisson_edge_width", type=int, default=0,
        help="Directly replace this many pixels along the left image edge "
             "where seamlessClone cannot solve a boundary-touching mask.",
    )
    parser.add_argument("--poisson_edge_feather", type=int, default=8)
    parser.add_argument(
        "--color_match_ring", type=int, default=0,
        help="Match the plate's per-channel brightness to a ring around the fill "
             "before compositing. This permits a crisp, narrow feather without "
             "an obvious exposure boundary.",
    )
    parser.add_argument("--protect_dilate", type=int, default=3)
    parser.add_argument("--plate_output", type=Path, default=None)
    parser.add_argument(
        "--plate_only", action="store_true",
        help="Estimate/write the clean plate and stop before video compositing.",
    )
    parser.add_argument("--plate_input", type=Path, default=None,
                        help="Use an existing clean full-resolution background plate "
                             "instead of estimating one from temporal samples.")
    parser.add_argument("--mask_dilate", type=int, default=0,
                        help="Expand the fill mask before feathering so the seam "
                             "falls outside the human silhouette and cast shadow.")
    parser.add_argument("--missing_fill", choices=("fallback", "horizontal"),
                        default="fallback",
                        help="How to fill plate pixels with too few clean samples. "
                             "horizontal interpolates clean tabletop pixels from "
                             "the same image row and avoids human-shaped fallback ghosts.")
    parser.add_argument("--min_valid_samples", type=int, default=1)
    args = parser.parse_args()

    masks = np.load(args.mask, mmap_mode="r")
    protect = (np.load(args.protect_mask, mmap_mode="r")
               if args.protect_mask is not None else None)
    plate_exclude = (np.load(args.plate_exclude_mask, mmap_mode="r")
                     if args.plate_exclude_mask is not None else None)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(masks))
    if protect is not None:
        frame_count = min(frame_count, len(protect))
    if plate_exclude is not None:
        frame_count = min(frame_count, len(plate_exclude))

    if args.plate_input is not None:
        cap.release()
        plate = cv2.imread(str(args.plate_input), cv2.IMREAD_COLOR)
        if plate is None:
            raise RuntimeError(f"cannot read plate: {args.plate_input}")
        if plate.shape[:2] != (height, width):
            plate = cv2.resize(plate, (width, height), interpolation=cv2.INTER_LINEAR)
        print(f"[info] using supplied plate {args.plate_input}, {width}x{height}")
    else:
        sample_frames = []
        sample_masks = []
        frame_idx = 0
        while frame_idx < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(1, args.sample_stride) == 0:
                sample_frames.append(frame)
                sample_mask = np.asarray(masks[frame_idx], dtype=np.uint8)
                if args.mask_dilate > 0:
                    sample_mask = cv2.dilate(
                        sample_mask,
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * args.mask_dilate + 1,) * 2,
                        ),
                        iterations=1,
                    )
                sample_mask = sample_mask.astype(bool)
                if plate_exclude is not None:
                    excluded = np.asarray(
                        plate_exclude[frame_idx], dtype=np.uint8
                    )
                    if args.plate_exclude_dilate > 0:
                        excluded = cv2.dilate(
                            excluded,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (2 * args.plate_exclude_dilate + 1,) * 2,
                            ),
                            iterations=1,
                        )
                    sample_mask |= excluded.astype(bool)
                sample_masks.append(sample_mask)
            frame_idx += 1
        cap.release()
        frames = np.stack(sample_frames)
        human = np.stack(sample_masks)
        print(f"[info] background samples={len(frames)}, {width}x{height}")

        plate = np.empty((height, width, 3), dtype=np.uint8)
        low_support_plate = np.zeros((height, width), dtype=bool)
        for row_start in range(0, height, max(1, args.row_block)):
            row_end = min(height, row_start + max(1, args.row_block))
            values = frames[:, row_start:row_end].astype(np.float32)
            invalid = human[:, row_start:row_end, :, None]
            low_support = ((~invalid[..., 0]).sum(axis=0) <
                           max(1, args.min_valid_samples))
            if args.plate_method == "median":
                values[invalid.repeat(3, axis=3)] = np.nan
                with np.errstate(all="ignore"):
                    block = np.nanmedian(values, axis=0)
                missing = ~np.isfinite(block)
                if missing.any():
                    fallback = np.median(frames[:, row_start:row_end], axis=0)
                    block[missing] = fallback[missing]
            else:
                luminance = (0.114 * values[..., 0] +
                             0.587 * values[..., 1] +
                             0.299 * values[..., 2])
                if args.plate_method == "cleanest":
                    chroma = values.max(axis=3) - values.min(axis=3)
                    score = luminance - 1.25 * chroma
                else:
                    score = luminance
                score[invalid[..., 0]] = -np.inf
                best = np.argmax(score, axis=0)
                block = np.take_along_axis(
                    values, best[None, ..., None], axis=0
                )[0]
                missing = ~np.isfinite(score).any(axis=0)
                if missing.any():
                    fallback = np.median(frames[:, row_start:row_end], axis=0)
                    block[missing] = fallback[missing]
            plate[row_start:row_end] = np.clip(block, 0, 255).astype(np.uint8)
            low_support_plate[row_start:row_end] = low_support
            print(f"[plate] rows {row_start}:{row_end}", flush=True)

        if args.missing_fill == "horizontal" and low_support_plate.any():
            x_all = np.arange(width, dtype=np.float32)
            repaired = plate.astype(np.float32)
            for y in range(height):
                missing_row = low_support_plate[y]
                if not missing_row.any():
                    continue
                good_x = np.flatnonzero(~missing_row)
                if not len(good_x):
                    continue
                for channel in range(3):
                    repaired[y, missing_row, channel] = np.interp(
                        x_all[missing_row], good_x,
                        repaired[y, good_x, channel],
                    )
            plate = np.clip(repaired, 0, 255).astype(np.uint8)
            print(f"[plate] horizontally repaired "
                  f"{int(low_support_plate.sum())} low-support pixels")

    plate_path = args.plate_output or args.output.with_name(
        f"{args.output.stem}_background_plate.jpg"
    )
    plate_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(plate_path), plate, [cv2.IMWRITE_JPEG_QUALITY, 95])

    if args.plate_only:
        print(f"[ok] wrote {plate_path}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"FFV1"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {args.output}")
    cap = cv2.VideoCapture(str(args.video))
    protect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.protect_dilate) + 1,
         2 * max(0, args.protect_dilate) + 1),
    )
    fill_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.mask_dilate) + 1,
         2 * max(0, args.mask_dilate) + 1),
    )
    match_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(0, args.color_match_ring) + 1,
         2 * max(0, args.color_match_ring) + 1),
    )
    plate_float = plate.astype(np.float32)
    written = 0
    for frame_idx in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        fill = np.asarray(masks[frame_idx], dtype=np.uint8)
        if args.mask_dilate > 0:
            fill = cv2.dilate(fill, fill_kernel, iterations=1)
        fill = fill.astype(bool)
        protected = None
        if protect is not None and frame_idx < len(protect):
            protected = cv2.dilate(
                np.asarray(protect[frame_idx], dtype=np.uint8),
                protect_kernel, iterations=1,
            ).astype(bool)
            if args.blend_mode == "poisson":
                fill &= ~protected
        matched_plate = plate_float
        if args.color_match_ring > 0 and fill.any():
            ring = cv2.dilate(
                fill.astype(np.uint8), match_kernel, iterations=1
            ).astype(bool) & ~fill
            if int(ring.sum()) >= 100:
                difference = frame.astype(np.float32) - plate_float
                offset = np.median(difference[ring], axis=0)
                # A robust local exposure correction is sufficient here; clipping
                # prevents a nearby coloured object from tinting the whole fill.
                offset = np.clip(offset, -35.0, 35.0)
                matched_plate = np.clip(plate_float + offset, 0.0, 255.0)
        if args.blend_mode == "poisson" and fill.any():
            ys, xs = np.nonzero(fill)
            padding = 3
            x0 = max(0, int(xs.min()) - padding)
            x1 = min(width, int(xs.max()) + padding + 1)
            y0 = max(0, int(ys.min()) - padding)
            y1 = min(height, int(ys.max()) + padding + 1)
            source = np.clip(
                matched_plate[y0:y1, x0:x1], 0, 255
            ).astype(np.uint8)
            clone_mask = (fill[y0:y1, x0:x1].astype(np.uint8) * 255)
            center = (x0 + source.shape[1] // 2,
                      y0 + source.shape[0] // 2)
            composed = cv2.seamlessClone(
                source, frame, clone_mask, center, cv2.NORMAL_CLONE
            )
            if args.poisson_edge_width > 0:
                edge_width = min(width, args.poisson_edge_width)
                feather = max(1, args.poisson_edge_feather)
                x = np.arange(width, dtype=np.float32)
                edge_alpha = np.clip(
                    (float(edge_width) - x) / float(feather), 0.0, 1.0
                )[None, :]
                edge_alpha = edge_alpha * fill.astype(np.float32)
                edge_alpha = edge_alpha[..., None]
                composed = (
                    edge_alpha * matched_plate
                    + (1.0 - edge_alpha) * composed.astype(np.float32)
                )
        else:
            alpha = fill.astype(np.float32)
            if args.feather_sigma > 0:
                alpha = cv2.GaussianBlur(alpha, (0, 0), args.feather_sigma)
            # Apply protection after feathering. Subtracting the object first
            # creates a low-alpha halo that leaves human skin around contact.
            if protected is not None:
                alpha[protected] = 0.0
            alpha = np.clip(alpha, 0.0, 1.0)[..., None]
            composed = alpha * matched_plate + (1.0 - alpha) * frame
        writer.write(np.clip(composed, 0, 255).astype(np.uint8))
        written += 1
        if written % 100 == 0:
            print(f"[frame] {written}/{frame_count}", flush=True)
    cap.release()
    writer.release()
    print(f"[ok] wrote {args.output}")
    print(f"[ok] wrote {plate_path}")


if __name__ == "__main__":
    main()
