"""Compare HaCo-only, strict HaCo+Depth, and sensor-Depth RB5 overlays."""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from composite_rb5_contact_occlusion import (
    FINGER_NAMES,
    _open_writer,
    _true_runs,
    _video_metadata,
)


MODE_NAMES = ("haco", "ensemble", "depth")
PAIR_NAMES = (
    ("haco", "ensemble"),
    ("haco", "depth"),
    ("ensemble", "depth"),
)


def _run_statistics(track: np.ndarray) -> dict[str, float | int]:
    runs = _true_runs(track)
    lengths = [end - start + 1 for start, end in runs]
    return {
        "run_count": len(runs),
        "median_run_frames": (
            float(np.median(lengths)) if lengths else 0.0
        ),
        "max_run_frames": max(lengths, default=0),
        "single_frame_runs": sum(length == 1 for length in lengths),
    }


def _label_frame(frame: np.ndarray, text: str, frame_index: int) -> np.ndarray:
    panel = np.asarray(frame, dtype=np.uint8).copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"frame {frame_index:04d}",
        (panel.shape[1] - 155, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--haco_dir", type=Path, required=True)
    parser.add_argument("--ensemble_dir", type=Path, required=True)
    parser.add_argument("--depth_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    processed = args.processed_demo.resolve()
    source_dirs = {
        "haco": args.haco_dir.resolve(),
        "ensemble": args.ensemble_dir.resolve(),
        "depth": args.depth_dir.resolve(),
    }
    output_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else processed / "occlusion_comparison"
    )
    video_paths = {
        "haco": source_dirs["haco"] / "video_overlay_contact.mp4",
        "ensemble": (
            source_dirs["ensemble"] / "video_overlay_contact.mp4"
        ),
        "depth": source_dirs["depth"] / "video_overlay_depth.mp4",
    }
    mask_paths = {
        mode: source_dirs[mode] / "occluded_finger_mask.npy"
        for mode in MODE_NAMES
    }

    metadata = {
        mode: _video_metadata(video_paths[mode])
        for mode in MODE_NAMES
    }
    width, height, frame_count, fps = metadata["haco"]
    for mode in MODE_NAMES[1:]:
        if metadata[mode][:3] != (width, height, frame_count):
            raise ValueError(
                f"{mode} video geometry mismatch: {metadata[mode]} "
                f"vs {metadata['haco']}"
            )
        if not np.isclose(metadata[mode][3], fps, atol=0.1):
            raise ValueError(f"{mode} video fps mismatch")

    masks = {
        mode: np.load(mask_paths[mode], mmap_mode="r")
        for mode in MODE_NAMES
    }
    expected_mask_shape = (frame_count, height, width)
    for mode, mask in masks.items():
        if mask.shape != expected_mask_shape:
            raise ValueError(
                f"{mode} mask shape mismatch: {mask.shape} != "
                f"{expected_mask_shape}"
            )

    finger_labels = np.load(
        processed / "overlay_processor" / "robot_finger_labels.npy",
        mmap_mode="r",
    )
    if len(finger_labels) != frame_count:
        raise ValueError("robot finger labels are not frame-aligned")

    pixel_count = {mode: 0 for mode in MODE_NAMES}
    frame_tracks = {
        mode: np.zeros(frame_count, dtype=bool)
        for mode in MODE_NAMES
    }
    finger_pixel_count = {
        mode: np.zeros(len(FINGER_NAMES), dtype=np.int64)
        for mode in MODE_NAMES
    }
    finger_frame_tracks = {
        mode: np.zeros(
            (frame_count, len(FINGER_NAMES)),
            dtype=bool,
        )
        for mode in MODE_NAMES
    }
    pair_counts = {
        f"{first}_{second}": {
            "intersection_pixels": 0,
            "union_pixels": 0,
            f"{first}_only_pixels": 0,
            f"{second}_only_pixels": 0,
        }
        for first, second in PAIR_NAMES
    }
    subset_violations = {
        "ensemble_not_in_haco_pixels": 0,
        "ensemble_not_in_depth_pixels": 0,
    }
    non_finger_pixels = {mode: 0 for mode in MODE_NAMES}

    for frame_index in range(frame_count):
        frame_masks = {
            mode: np.asarray(masks[mode][frame_index], dtype=bool)
            for mode in MODE_NAMES
        }
        labels = np.asarray(
            finger_labels[frame_index],
            dtype=np.uint8,
        )
        if labels.shape != (height, width):
            labels = cv2.resize(
                labels,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        rendered_fingers = labels > 0
        for mode in MODE_NAMES:
            mask = frame_masks[mode]
            pixels = int(mask.sum())
            pixel_count[mode] += pixels
            frame_tracks[mode][frame_index] = pixels > 0
            non_finger_pixels[mode] += int(
                np.logical_and(mask, ~rendered_fingers).sum()
            )
            for finger_index in range(len(FINGER_NAMES)):
                finger_pixels = int(
                    np.logical_and(
                        mask,
                        labels == finger_index + 1,
                    ).sum()
                )
                finger_pixel_count[mode][finger_index] += finger_pixels
                finger_frame_tracks[mode][
                    frame_index,
                    finger_index,
                ] = finger_pixels > 0
        for first, second in PAIR_NAMES:
            first_mask = frame_masks[first]
            second_mask = frame_masks[second]
            key = f"{first}_{second}"
            pair_counts[key]["intersection_pixels"] += int(
                np.logical_and(first_mask, second_mask).sum()
            )
            pair_counts[key]["union_pixels"] += int(
                np.logical_or(first_mask, second_mask).sum()
            )
            pair_counts[key][f"{first}_only_pixels"] += int(
                np.logical_and(first_mask, ~second_mask).sum()
            )
            pair_counts[key][f"{second}_only_pixels"] += int(
                np.logical_and(second_mask, ~first_mask).sum()
            )
        subset_violations["ensemble_not_in_haco_pixels"] += int(
            np.logical_and(
                frame_masks["ensemble"],
                ~frame_masks["haco"],
            ).sum()
        )
        subset_violations["ensemble_not_in_depth_pixels"] += int(
            np.logical_and(
                frame_masks["ensemble"],
                ~frame_masks["depth"],
            ).sum()
        )

    for values in pair_counts.values():
        union = values["union_pixels"]
        values["iou"] = (
            float(values["intersection_pixels"] / union)
            if union
            else 1.0
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".occlusion_comparison.",
            dir=output_dir.parent,
        )
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    captures = {
        mode: cv2.VideoCapture(str(video_paths[mode]))
        for mode in MODE_NAMES
    }
    if not all(capture.isOpened() for capture in captures.values()):
        raise RuntimeError("could not open all comparison videos")
    writer = _open_writer(
        staging / "video_compare_haco_ensemble_depth.mp4",
        fps,
        (width * 3, height),
    )
    labels = {
        "haco": "HaCo only",
        "ensemble": "HaCo + Sensor Depth",
        "depth": "Sensor Depth only",
    }
    try:
        for frame_index in range(frame_count):
            panels = []
            for mode in MODE_NAMES:
                ok, frame = captures[mode].read()
                if not ok:
                    raise RuntimeError(
                        f"{mode} video read failed at frame {frame_index}"
                    )
                panels.append(
                    _label_frame(frame, labels[mode], frame_index)
                )
            writer.write(np.concatenate(panels, axis=1))
    finally:
        for capture in captures.values():
            capture.release()
        writer.release()

    mode_report = {}
    for mode in MODE_NAMES:
        per_finger = {}
        for finger_index, finger in enumerate(FINGER_NAMES):
            per_finger[finger] = {
                "pixels": int(
                    finger_pixel_count[mode][finger_index]
                ),
                "frames": int(
                    finger_frame_tracks[mode][
                        :,
                        finger_index,
                    ].sum()
                ),
                **_run_statistics(
                    finger_frame_tracks[mode][:, finger_index]
                ),
            }
        mode_report[mode] = {
            "pixels": int(pixel_count[mode]),
            "frames": int(frame_tracks[mode].sum()),
            **_run_statistics(frame_tracks[mode]),
            "per_finger": per_finger,
        }

    report = {
        "schema_version": 1,
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "definitions": {
            "haco": (
                "HaCo contact/hidden evidence and HaCo contact-surface depth"
            ),
            "depth": (
                "sensor object depth and depth-coherent object mask; "
                "no HaCo decision"
            ),
            "ensemble": (
                "strict pixelwise intersection of HaCo and sensor-depth "
                "occlusion decisions"
            ),
        },
        "sources": {
            mode: {
                "directory": str(source_dirs[mode]),
                "video": str(video_paths[mode]),
                "mask": str(mask_paths[mode]),
            }
            for mode in MODE_NAMES
        },
        "modes": mode_report,
        "pairwise": pair_counts,
        "invariants": {
            "ensemble_subset_of_haco": (
                subset_violations["ensemble_not_in_haco_pixels"] == 0
            ),
            "ensemble_subset_of_depth": (
                subset_violations["ensemble_not_in_depth_pixels"] == 0
            ),
            "all_masks_are_finger_only": all(
                count == 0 for count in non_finger_pixels.values()
            ),
            "subset_violation_pixels": subset_violations,
            "non_finger_pixels": non_finger_pixels,
        },
        "note": (
            "Pairwise IoU measures predictor agreement, not ground-truth "
            "occlusion accuracy."
        ),
    }
    (staging / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] occlusion comparison: {output_dir}", flush=True)
    print(
        "[info] "
        + ", ".join(
            f"{mode}={pixel_count[mode]}px/"
            f"{int(frame_tracks[mode].sum())}f"
            for mode in MODE_NAMES
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
