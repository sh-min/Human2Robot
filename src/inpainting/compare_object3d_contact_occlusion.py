"""Render a synchronized 2x2 comparison of object-depth contact strategies.

Layout::

    current HaCo proxy       | per-frame scalar object depth
    dense object surface     | dense surface + HaCo contact registration

The source compositor outputs are read-only.  Reports, masks, video metadata,
and pairwise differences are validated before the comparison is published.
"""

from __future__ import annotations

import argparse
import atexit
import itertools
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from atomic_directory_publish import publish_directory


MODE_SPECS = (
    ("haco_proxy", "1 Current: HaCo contact-Z"),
    ("scalar_object_depth", "2 Legacy: HaCo AND scalar object-Z"),
    ("dense_surface", "3 3D: dense object surface"),
    ("contact_aligned_surface", "4 Proposed: surface + HaCo register"),
)
VIDEO_NAME = "video_overlay_contact.mp4"
MASK_NAME = "occluded_finger_mask.npy"
REPORT_NAME = "report.json"
OUTPUT_VIDEO_NAME = "video_compare_object3d_contact_2x2.mp4"


def _video_metadata(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    metadata: dict[str, float | int] = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    capture.release()
    if (
        int(metadata["width"]) <= 0
        or int(metadata["height"]) <= 0
        or int(metadata["frames"]) <= 0
        or float(metadata["fps"]) <= 0.0
    ):
        raise ValueError(f"invalid video metadata: {path}")
    return metadata


def _label_panel(frame: np.ndarray, label: str) -> np.ndarray:
    image = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    header = np.zeros((40, 640, 3), dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((header, image))


def _load_source(directory: Path, mode: str) -> dict[str, object]:
    root = directory.resolve()
    video_path = root / VIDEO_NAME
    mask_path = root / MASK_NAME
    report_path = root / REPORT_NAME
    for path in (video_path, mask_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"{mode} source is incomplete: {path}")
    report = json.loads(report_path.read_text())
    mask = np.load(mask_path, mmap_mode="r")
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError(f"{mode} mask must be bool (T,H,W), got {mask.shape}")
    metadata = _video_metadata(video_path)
    if len(mask) != metadata["frames"]:
        raise ValueError(f"{mode} mask/video frame mismatch")
    if int(report.get("frames", -1)) != len(mask):
        raise ValueError(f"{mode} report/mask frame mismatch")
    if int(report.get("occluded_pixels_total", -1)) != int(mask.sum()):
        raise ValueError(f"{mode} report/mask pixel-count mismatch")
    return {
        "root": root,
        "video": video_path,
        "mask": mask,
        "report": report,
        "metadata": metadata,
    }


def _validate_contract(sources: dict[str, dict[str, object]]) -> None:
    expected_modes = {
        "haco_proxy": "haco",
        "scalar_object_depth": "ensemble",
        "dense_surface": "object3d",
        "contact_aligned_surface": "object3d",
    }
    reference = sources["haco_proxy"]["metadata"]
    reference_report = sources["haco_proxy"]["report"]
    assert isinstance(reference, dict)
    assert isinstance(reference_report, dict)
    common_report_fields = (
        "frames",
        "width",
        "height",
        "fps",
        "side",
        "finger_names",
        "config",
        "aux_frame_offset",
        "aux_side",
        "contact_score_fused",
        "hidden_fraction",
        "active_runs",
    )
    common_source_fields = (
        "processed_demo",
        "episode_dir",
        "background",
        "raw_video",
        "hawor_npz",
        "contact_dir",
        "aux_contact_dir",
        "overlay_dir",
        "object_mask",
    )
    for mode, source in sources.items():
        report = source["report"]
        metadata = source["metadata"]
        mask = source["mask"]
        assert isinstance(report, dict)
        assert isinstance(metadata, dict)
        assert isinstance(mask, np.ndarray)
        if report.get("occlusion_mode") != expected_modes[mode]:
            raise ValueError(
                f"{mode} occlusion mode {report.get('occlusion_mode')!r} "
                f"!= {expected_modes[mode]!r}"
            )
        if metadata != reference:
            raise ValueError(f"{mode} video metadata differs from baseline")
        baseline_mask = sources["haco_proxy"]["mask"]
        assert isinstance(baseline_mask, np.ndarray)
        if mask.shape != baseline_mask.shape:
            raise ValueError(f"{mode} mask shape differs from baseline")
        for field in common_report_fields:
            if report.get(field) != reference_report.get(field):
                raise ValueError(
                    f"{mode} report field {field!r} differs from baseline"
                )
        report_sources = report.get("sources", {})
        reference_sources = reference_report.get("sources", {})
        for field in common_source_fields:
            if report_sources.get(field) != reference_sources.get(field):
                raise ValueError(
                    f"{mode} source {field!r} differs from baseline"
                )
    dense_report = sources["dense_surface"]["report"]
    aligned_report = sources["contact_aligned_surface"]["report"]
    assert isinstance(dense_report, dict) and isinstance(aligned_report, dict)
    if dense_report.get("object_surface_3d", {}).get("alignment") != "none":
        raise ValueError("dense surface source is not the unaligned ablation")
    if aligned_report.get("object_surface_3d", {}).get("alignment") != "contact":
        raise ValueError("contact-aligned source has the wrong alignment mode")
    dense_surface_path = dense_report.get("sources", {}).get(
        "object_surface_depth"
    )
    aligned_surface_path = aligned_report.get("sources", {}).get(
        "object_surface_depth"
    )
    if not dense_surface_path or dense_surface_path != aligned_surface_path:
        raise ValueError(
            "dense and contact-registered variants use different surfaces"
        )
    for mode, report in (
        ("dense_surface", dense_report),
        ("contact_aligned_surface", aligned_report),
    ):
        if not report.get("invariants", {}).get(
            "object3d_haco_is_selector_only"
        ):
            raise ValueError(f"{mode} does not use selector-only HaCo")
    scalar_report = sources["scalar_object_depth"]["report"]
    assert isinstance(scalar_report, dict)
    if not scalar_report.get("sources", {}).get("scene_depth"):
        raise ValueError("scalar object-depth source is missing scene depth")


def _statistics(sources: dict[str, dict[str, object]]) -> dict[str, object]:
    masks = {mode: source["mask"] for mode, source in sources.items()}
    for mask in masks.values():
        assert isinstance(mask, np.ndarray)
    frame_count = len(masks["haco_proxy"])
    mode_stats = {
        mode: {
            "pixels": int(mask.sum()),
            "frames": int(np.any(mask, axis=(1, 2)).sum()),
        }
        for mode, mask in masks.items()
    }
    pairs: dict[str, dict[str, int]] = {}
    for first_mode, second_mode in itertools.combinations(masks, 2):
        first_mask = masks[first_mode]
        second_mask = masks[second_mode]
        added = removed = changed_frames = 0
        for frame_index in range(frame_count):
            first = np.asarray(first_mask[frame_index], dtype=bool)
            second = np.asarray(second_mask[frame_index], dtype=bool)
            frame_added = int((second & ~first).sum())
            frame_removed = int((first & ~second).sum())
            added += frame_added
            removed += frame_removed
            changed_frames += int(frame_added + frame_removed > 0)
        pairs[f"{first_mode}_vs_{second_mode}"] = {
            "added_pixels": added,
            "removed_pixels": removed,
            "changed_frames": changed_frames,
        }
    return {"modes": mode_stats, "comparisons": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--scalar_dir", type=Path, required=True)
    parser.add_argument("--surface_dir", type=Path, required=True)
    parser.add_argument("--contact_aligned_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    directories = {
        "haco_proxy": args.baseline_dir,
        "scalar_object_depth": args.scalar_dir,
        "dense_surface": args.surface_dir,
        "contact_aligned_surface": args.contact_aligned_dir,
    }
    sources = {
        mode: _load_source(directory, mode)
        for mode, directory in directories.items()
    }
    _validate_contract(sources)
    statistics = _statistics(sources)
    reference = sources["haco_proxy"]["metadata"]
    assert isinstance(reference, dict)

    out_dir = args.out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".object3d_compare.", dir=out_dir.parent))
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    captures = {
        mode: cv2.VideoCapture(str(source["video"]))
        for mode, source in sources.items()
    }
    if not all(capture.isOpened() for capture in captures.values()):
        raise RuntimeError("could not open all comparison source videos")
    writer = cv2.VideoWriter(
        str(staging / OUTPUT_VIDEO_NAME),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(reference["fps"]),
        (1280, 800),
    )
    if not writer.isOpened():
        raise RuntimeError("could not open object3d comparison writer")
    try:
        for frame_index in range(int(reference["frames"])):
            panels = []
            for mode, label in MODE_SPECS:
                ok, frame = captures[mode].read()
                if not ok:
                    raise RuntimeError(
                        f"{mode} video read failed at frame {frame_index}"
                    )
                panels.append(_label_panel(frame, label))
            writer.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
    finally:
        writer.release()
        for capture in captures.values():
            capture.release()

    report = {
        "schema_version": 1,
        "layout": [
            [MODE_SPECS[0][0], MODE_SPECS[1][0]],
            [MODE_SPECS[2][0], MODE_SPECS[3][0]],
        ],
        "mode_definitions": {
            "haco_proxy": "legacy HaCo contact-Z depth gate",
            "scalar_object_depth": (
                "legacy ensemble intersection of HaCo contact-Z and one "
                "robust object depth per frame"
            ),
            "dense_surface": (
                "HaCo selects finger/local support; dense object surface "
                "owns depth ordering"
            ),
            "contact_aligned_surface": (
                "dense surface depth ordering after bounded local HaCo "
                "contact registration"
            ),
        },
        "video": OUTPUT_VIDEO_NAME,
        "frames": int(reference["frames"]),
        "fps": float(reference["fps"]),
        "sources": {
            mode: str(source["root"])
            for mode, source in sources.items()
        },
        **statistics,
    }
    (staging / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    publish_directory(staging, out_dir)
    print(f"[ok] {out_dir / OUTPUT_VIDEO_NAME}", flush=True)


if __name__ == "__main__":
    main()
