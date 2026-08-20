"""Apply a precomputed contact-occlusion mask to a full-resolution overlay.

This preserves the original high-resolution robot render and only restores the
inpainted background at pixels selected by the HaCo compositor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def video_meta(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    result = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
    )
    cap.release()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--occlusion_mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--edge_sigma_px", type=float, default=1.2)
    args = parser.parse_args()

    overlay_meta = video_meta(args.overlay)
    background_meta = video_meta(args.background)
    if overlay_meta != background_meta:
        raise ValueError(
            f"overlay/background mismatch: {overlay_meta} != {background_meta}"
        )
    width, height, frame_count, fps = overlay_meta
    masks = np.load(args.occlusion_mask, mmap_mode="r")
    if masks.shape != (frame_count, height, width):
        raise ValueError(
            f"mask shape {masks.shape} != {(frame_count, height, width)}"
        )
    if args.edge_sigma_px < 0:
        parser.error("--edge_sigma_px must be non-negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay_cap = cv2.VideoCapture(str(args.overlay))
    background_cap = cv2.VideoCapture(str(args.background))
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output: {args.output}")

    hidden_per_frame = np.zeros(frame_count, dtype=np.int64)
    try:
        for frame_index in range(frame_count):
            ok_overlay, overlay = overlay_cap.read()
            ok_background, background = background_cap.read()
            if not ok_overlay or not ok_background:
                raise RuntimeError(f"video read failed at frame {frame_index}")
            hidden = np.asarray(masks[frame_index], dtype=np.float32)
            hidden_per_frame[frame_index] = int(np.count_nonzero(hidden))
            if args.edge_sigma_px > 0 and np.any(hidden):
                hidden = cv2.GaussianBlur(
                    hidden, (0, 0), args.edge_sigma_px
                )
            alpha = np.clip(hidden, 0.0, 1.0)[..., None]
            frame = np.clip(
                overlay.astype(np.float32) * (1.0 - alpha)
                + background.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)
            writer.write(frame)
    finally:
        overlay_cap.release()
        background_cap.release()
        writer.release()

    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(json.dumps({
        "method": "full_resolution_haco_masked_arm_stabilized_overlay",
        "overlay": str(args.overlay.resolve()),
        "background": str(args.background.resolve()),
        "occlusion_mask": str(args.occlusion_mask.resolve()),
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "edge_sigma_px": args.edge_sigma_px,
        "occluded_pixels": int(hidden_per_frame.sum()),
        "occluded_frames": int(np.count_nonzero(hidden_per_frame)),
        "invariants": {
            "arm_stabilized_source_preserved": True,
            "only_haco_selected_pixels_changed": True,
            "inpainting_background_preserved": True,
        },
    }, indent=2))
    print(f"[ok] wrote {args.output}")
    print(
        f"[info] hidden pixels={hidden_per_frame.sum()}, "
        f"frames={np.count_nonzero(hidden_per_frame)}/{frame_count}"
    )


if __name__ == "__main__":
    main()
