#!/usr/bin/env python3
"""Prepare the 08_04 MH-primary / SH-auxiliary stereo training dataset.

The annotation frame count is authoritative.  Each source MOV is decoded by
frame index without FPS resampling and truncated to that shared count.  The
resulting layout is accepted by the existing V-JEPA, HaWoR, HaCo, and stereo
occlusion tools while keeping MH as the legacy/default recording view.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "08_04"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "cube_dataset" / "26.08.04_stereo"
LABELS = {"Cup", "Lock", "Milk", "Snack", "Sweep", "Trans"}
# A multi-cue motion-correlation audit found a fixed one-frame capture phase
# difference only in episodes 16 and 18.  MH/GT remains immutable; this offset
# is consumed solely when SH evidence is fused onto the MH output time axis.
CAMERA1_FRAME_OFFSETS = {"16": -1, "18": -1}


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def probe_frame_count(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if value and value != "N/A":
        return int(value)
    fallback = command.copy()
    fallback.insert(fallback.index("-show_entries"), "-count_frames")
    fallback[fallback.index("stream=nb_frames")] = "stream=nb_read_frames"
    result = subprocess.run(fallback, check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def load_and_validate_gt(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"episode", "num_frames", "fps", "segments"}
    if set(payload) != required:
        raise ValueError(f"{path}: expected exactly {sorted(required)}, got {sorted(payload)}")
    frame_count = int(payload["num_frames"])
    fps = float(payload["fps"])
    if frame_count <= 0 or abs(fps - 24.0) > 1.0e-6:
        raise ValueError(f"{path}: expected positive frame count at 24 FPS")
    coverage = [0] * frame_count
    previous_end = -1
    for index, segment in enumerate(payload["segments"], 1):
        label = segment.get("label")
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        if label not in LABELS:
            raise ValueError(f"{path}: segment {index} has unknown label {label!r}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{path}: segment {index} bounds must be integers")
        if not 0 <= start <= end < frame_count:
            raise ValueError(f"{path}: segment {index} is out of bounds: {start}-{end}")
        if start <= previous_end:
            raise ValueError(f"{path}: segment {index} overlaps the previous segment")
        for frame in range(start, end + 1):
            coverage[frame] += 1
        previous_end = end
    missing = [index for index, count in enumerate(coverage) if count == 0]
    duplicate = [index for index, count in enumerate(coverage) if count > 1]
    if missing or duplicate:
        raise ValueError(
            f"{path}: labels must cover every frame once; "
            f"missing={missing[:5]}, duplicate={duplicate[:5]}"
        )
    return payload


def discover_episodes(source_root: Path) -> list[str]:
    mh = {path.stem for path in (source_root / "mh").glob("*.mov")}
    sh = {path.stem for path in (source_root / "sh").glob("*.mov")}
    gt = {
        path.parent.name
        for path in (source_root / "annotations").glob("*/gt_labels.json")
    }
    common = sorted(mh & sh & gt, key=natural_key)
    if not common:
        raise ValueError(f"no complete MH/SH/GT episodes under {source_root}")
    if mh != sh or mh != gt:
        raise ValueError(
            "source sets do not match: "
            f"mh_only={sorted(mh-sh, key=natural_key)}, "
            f"sh_only={sorted(sh-mh, key=natural_key)}, "
            f"without_gt={sorted((mh | sh)-gt, key=natural_key)}"
        )
    return common


def image_count(directory: Path) -> int:
    return sum(
        1
        for path in directory.glob("rgb_frame*.jpg")
        if path.is_file()
    )


def extract_frames(source: Path, destination: Path, expected: int) -> None:
    if destination.is_dir():
        count = image_count(destination)
        if count == expected:
            return
        raise ValueError(
            f"existing frame directory is incomplete: {destination} "
            f"({count} != {expected}); move it aside before retrying"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(expected),
            "-start_number",
            "0",
            "-q:v",
            "2",
            str(staging / "rgb_frame%06d.jpg"),
        ]
        subprocess.run(command, check=True)
        count = image_count(staging)
        if count != expected:
            raise RuntimeError(
                f"decoded frame mismatch for {source}: {count} != {expected}"
            )
        staging.chmod(0o755)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def ensure_relative_symlink(link: Path, target: Path) -> None:
    relative = os.path.relpath(target, start=link.parent)
    if link.is_symlink():
        if os.readlink(link) != relative:
            raise ValueError(f"existing symlink has a different target: {link}")
        return
    if link.exists():
        raise ValueError(f"refusing to replace existing path: {link}")
    link.symlink_to(relative, target_is_directory=target.is_dir())


def prepare_episode(source_root: Path, output_root: Path, name: str) -> dict[str, object]:
    gt_source = source_root / "annotations" / name / "gt_labels.json"
    gt = load_and_validate_gt(gt_source)
    if str(gt["episode"]) != name:
        raise ValueError(f"{gt_source}: episode field {gt['episode']!r} != {name!r}")
    expected = int(gt["num_frames"])
    mh_source = source_root / "mh" / f"{name}.mov"
    sh_source = source_root / "sh" / f"{name}.mov"
    mh_raw_frames = probe_frame_count(mh_source)
    sh_raw_frames = probe_frame_count(sh_source)
    if expected != min(mh_raw_frames, sh_raw_frames):
        raise ValueError(
            f"{name}: GT frames {expected} != common raw frames "
            f"min({mh_raw_frames}, {sh_raw_frames})"
        )

    episode = output_root / name
    camera1 = episode / "camera_1"  # SH auxiliary evidence
    camera2 = episode / "camera_2"  # MH primary/final view
    episode.mkdir(parents=True, exist_ok=True)
    extract_frames(sh_source, camera1 / "rgb", expected)
    extract_frames(mh_source, camera2 / "rgb", expected)
    for directory in (
        camera1 / "rgb_hawor",
        camera1 / "contact",
        camera2 / "rgb_hawor",
        camera2 / "contact",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(gt_source, episode / "gt_labels.json")
    ensure_relative_symlink(episode / "rgb", camera2 / "rgb")
    ensure_relative_symlink(episode / "rgb_hawor", camera2 / "rgb_hawor")
    ensure_relative_symlink(episode / "contact", camera2 / "contact")
    ensure_relative_symlink(camera1 / "source.mov", sh_source)
    ensure_relative_symlink(camera2 / "source.mov", mh_source)
    camera1_frame_offset = CAMERA1_FRAME_OFFSETS.get(name, 0)

    manifest = {
        "schema_version": 2,
        "episode": name,
        "fps": 24.0,
        "common_frames": expected,
        "primary_view": "MH",
        "auxiliary_view": "SH",
        "stereo_code_mapping": {"camera_1": "SH", "camera_2": "MH"},
        "training_view": "MH",
        "robot_overlay_view": "MH",
        "sources": {
            "MH": str(mh_source.resolve()),
            "SH": str(sh_source.resolve()),
            "gt_labels": str(gt_source.resolve()),
        },
        "raw_frame_counts": {"MH": mh_raw_frames, "SH": sh_raw_frames},
        "tail_frames_dropped": {
            "MH": mh_raw_frames - expected,
            "SH": sh_raw_frames - expected,
        },
        "temporal_alignment": {
            "reference_view": "camera_2/MH/GT",
            "camera1_frame_offset": camera1_frame_offset,
            "camera1_lookup": (
                "camera1/SH source index = camera2/MH frame k + "
                f"({camera1_frame_offset})"
            ),
            "apply_only_during_dual_view_fusion": True,
            "source_frames_reordered": False,
            "out_of_range_policy": "fail_open",
            "audit_method": (
                "six-cue motion correlation plus high-motion visual review"
            ),
        },
        "intrinsics": {
            "status": "pending_actual_phone_checkerboard_calibration",
            "source_lens_tags": {
                "SH": "iPhone 13 26mm",
                "MH": "iPhone 17 26mm",
            },
            "pixel_focal_px": {"SH": None, "MH": None},
            "rejected_source": "calibration-20260802T125601Z-1-001.zip (RealSense)",
        },
        "frame_mapping": "output frame k equals decoded source frame k",
    }
    (episode / "stereo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="Optional episode names; default prepares every complete pair",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    available = discover_episodes(source_root)
    episodes = args.episodes or available
    unknown = sorted(set(episodes) - set(available), key=natural_key)
    if unknown:
        raise SystemExit(f"unknown/incomplete episodes: {unknown}")
    output_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for index, name in enumerate(sorted(episodes, key=natural_key), 1):
        manifest = prepare_episode(source_root, output_root, name)
        total_frames += int(manifest["common_frames"])
        dropped = manifest["tail_frames_dropped"]
        print(
            f"[{index:02d}/{len(episodes):02d}] {name}: "
            f"{manifest['common_frames']} frames "
            f"(drop MH={dropped['MH']}, SH={dropped['SH']})",
            flush=True,
        )
    print(
        f"Prepared {len(episodes)} episodes / {total_frames} common frames at "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
