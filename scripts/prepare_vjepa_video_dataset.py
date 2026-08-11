#!/usr/bin/env python3
"""Build a lightweight V-JEPA dataset that links directly to raw videos."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


VIDEO_SUFFIXES = frozenset({".mov", ".mp4", ".avi", ".mkv"})


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def probe_frames(video: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def validate_annotation(path: Path, video_frames: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_count = int(payload["num_frames"])
    if frame_count > video_frames:
        raise ValueError(
            f"{path}: annotation has {frame_count} frames but video has only "
            f"{video_frames}"
        )
    coverage = [0] * frame_count
    for segment in payload["segments"]:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if not 0 <= start <= end < frame_count:
            raise ValueError(f"{path}: invalid segment {start}-{end}")
        for frame in range(start, end + 1):
            coverage[frame] += 1
    if any(value != 1 for value in coverage):
        raise ValueError(f"{path}: labels must cover every frame exactly once")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--annotation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--exclude_label", action="append", default=[])
    parser.add_argument("--validation_count", type=int, default=4)
    args = parser.parse_args()

    video_dir = args.video_dir.resolve()
    annotation_dir = args.annotation_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    if args.validation_count <= 0:
        raise ValueError("validation_count must be positive")

    videos = {
        path.stem: path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    }
    annotations = {
        path.parent.name: path
        for path in annotation_dir.glob("*/gt_labels.json")
    }
    if set(videos) != set(annotations):
        raise ValueError(
            "video/annotation episode mismatch: "
            f"video_only={sorted(set(videos) - set(annotations), key=natural_key)}, "
            f"annotation_only={sorted(set(annotations) - set(videos), key=natural_key)}"
        )

    excluded_labels = {label.casefold() for label in args.exclude_label}
    included: list[dict] = []
    excluded: list[dict] = []
    for episode in sorted(videos, key=natural_key):
        video = videos[episode]
        annotation = annotations[episode]
        video_frames = probe_frames(video)
        payload = validate_annotation(annotation, video_frames)
        labels = sorted(
            {str(segment["label"]) for segment in payload["segments"]},
            key=str.casefold,
        )
        matched = sorted(
            {label for label in labels if label.casefold() in excluded_labels},
            key=str.casefold,
        )
        item = {
            "episode": episode,
            "video": str(video),
            "annotation": str(annotation),
            "frames": int(payload["num_frames"]),
            "source_video_frames": video_frames,
            "tail_frames_dropped": video_frames - int(payload["num_frames"]),
            "fps": float(payload["fps"]),
            "labels": labels,
        }
        if matched:
            item["matched_excluded_labels"] = matched
            excluded.append(item)
        else:
            included.append(item)

    if len(included) <= args.validation_count:
        raise ValueError(
            f"need more than {args.validation_count} included episodes, got {len(included)}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for item in included:
            episode_dir = staging / item["episode"]
            episode_dir.mkdir()
            relative_video = os.path.relpath(item["video"], episode_dir)
            (episode_dir / "rgb").symlink_to(relative_video)
            shutil.copy2(item["annotation"], episode_dir / "gt_labels.json")

        names = [item["episode"] for item in included]
        train = names[:-args.validation_count]
        validation = names[-args.validation_count:]
        manifest = {
            "schema_version": 1,
            "method": "raw-video symlink V-JEPA dataset",
            "video_dir": str(video_dir),
            "annotation_dir": str(annotation_dir),
            "excluded_labels": sorted(excluded_labels),
            "exclusion_match": "case-insensitive exact label",
            "included": included,
            "excluded": excluded,
            "split": {"train": train, "validation": validation},
            "totals": {
                "included_episodes": len(included),
                "excluded_episodes": len(excluded),
                "included_frames": sum(item["frames"] for item in included),
            },
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(manifest["totals"], indent=2))
    print(f"train={','.join(train)}")
    print(f"validation={','.join(validation)}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
