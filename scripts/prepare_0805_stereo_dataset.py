#!/usr/bin/env python3
"""Prepare the 08-05 MH-primary / SH-auxiliary stereo dataset.

Unlike the 08-04 recordings, the MH and SH filenames do not share episode
stems.  This script therefore pairs the naturally sorted MH files, SH files,
and annotation directories by position and records that decision explicitly
in every episode manifest.  The annotation frame count is authoritative; raw
videos are decoded by frame index, without FPS resampling, and truncated to
the common annotated length.

The pipeline camera convention is unchanged from 08-04:

* ``camera_1`` is SH (auxiliary evidence)
* ``camera_2`` is MH (primary/training/final view)

The manual calibration picker used the opposite numbering for this capture
(``camera_1`` = MH, ``camera_2`` = SH).  When ``--calibration_json`` is given,
the manifest records both namespaces, their exact mapping, calibration source
videos, and the per-view intrinsics so consumers cannot silently swap them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "8-5" / "data"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "cube_dataset" / "26.08.05_stereo"
LABELS = ("Cup", "Lock", "Choco", "Snack", "Sweep", "Trans")
LABEL_SET = set(LABELS)

# This is the output layout consumed by the existing 08-04 runners.
PIPELINE_CAMERA_MAPPING = {"camera_1": "SH", "camera_2": "MH"}
# This is the namespace used while saving 8-5/cali/manual_pairs.
DEFAULT_CALIBRATION_CAMERA_MAPPING = {"camera_1": "MH", "camera_2": "SH"}

# Reuse the already exercised frame probing/extraction and symlink safety
# helpers without changing the 08-04 script or its label vocabulary.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from prepare_0804_stereo_dataset import (  # noqa: E402
    ensure_relative_symlink,
    extract_frames,
    natural_key,
    probe_frame_count,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing video directory: {directory}")
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() == ".mov"
        ),
        key=lambda path: natural_key(path.name),
    )


def load_and_validate_gt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"episode", "num_frames", "fps", "segments"}
    if set(payload) != required:
        raise ValueError(
            f"{path}: expected exactly {sorted(required)}, got {sorted(payload)}"
        )

    frame_count = int(payload["num_frames"])
    fps = float(payload["fps"])
    if frame_count <= 0 or abs(fps - 24.0) > 1.0e-6:
        raise ValueError(f"{path}: expected positive frame count at 24 FPS")
    if not isinstance(payload["segments"], list) or not payload["segments"]:
        raise ValueError(f"{path}: expected at least one labeled segment")

    coverage = [0] * frame_count
    previous_end = -1
    for index, segment in enumerate(payload["segments"], 1):
        label = segment.get("label")
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        if label not in LABEL_SET:
            raise ValueError(f"{path}: segment {index} has unknown label {label!r}")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"{path}: segment {index} start_frame must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f"{path}: segment {index} end_frame must be an integer")
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


def discover_episode_sources(source_root: Path) -> list[dict[str, Any]]:
    """Pair annotations, MH videos, and SH videos by natural-order position."""

    source_root = source_root.resolve()
    mh_videos = _video_files(source_root / "mh")
    sh_videos = _video_files(source_root / "sh")
    annotations = sorted(
        (
            path
            for path in (source_root / "annotations").glob("*/gt_labels.json")
            if path.is_file()
        ),
        key=lambda path: natural_key(path.parent.name),
    )
    counts = {
        "annotations": len(annotations),
        "MH": len(mh_videos),
        "SH": len(sh_videos),
    }
    if not annotations:
        raise ValueError(f"no annotation episodes under {source_root / 'annotations'}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"cannot pair unequal natural-order source sets: {counts}")

    paired: list[dict[str, Any]] = []
    episode_names: set[str] = set()
    for order_index, (gt_path, mh_path, sh_path) in enumerate(
        zip(annotations, mh_videos, sh_videos, strict=True),
        start=1,
    ):
        episode = gt_path.parent.name
        if episode in episode_names:
            raise ValueError(f"duplicate annotation episode {episode!r}")
        episode_names.add(episode)
        gt = load_and_validate_gt(gt_path)
        if str(gt["episode"]) != episode:
            raise ValueError(
                f"{gt_path}: episode field {gt['episode']!r} != directory {episode!r}"
            )
        paired.append(
            {
                "episode": episode,
                "order_index": order_index,
                "MH": mh_path.resolve(),
                "SH": sh_path.resolve(),
                "gt_labels": gt_path.resolve(),
            }
        )
    return paired


def _decode_motion_energy(
    video: Path,
    frame_limit: int,
    *,
    width: int = 32,
    height: int = 18,
) -> list[float]:
    """Decode a tiny grayscale stream and return frame-to-frame motion energy."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={width}:{height}:flags=area,format=gray",
        "-frames:v",
        str(frame_limit),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    pixels_per_frame = width * height
    if len(result.stdout) % pixels_per_frame:
        raise RuntimeError(
            f"unexpected rawvideo byte count for {video}: {len(result.stdout)}"
        )
    frame_count = len(result.stdout) // pixels_per_frame
    if frame_count < 3:
        raise RuntimeError(f"too few decoded frames for motion audit: {video}")
    raw = memoryview(result.stdout)
    trace: list[float] = []
    for frame in range(1, frame_count):
        previous = raw[(frame - 1) * pixels_per_frame : frame * pixels_per_frame]
        current = raw[frame * pixels_per_frame : (frame + 1) * pixels_per_frame]
        trace.append(
            sum(abs(int(a) - int(b)) for a, b in zip(previous, current))
            / pixels_per_frame
        )
    return trace


def _pearson_at_offset(mh_trace: list[float], sh_trace: list[float], offset: int) -> float:
    """Score ``SH index = MH index + offset`` for two motion traces."""

    start = max(0, -offset)
    stop = min(len(mh_trace), len(sh_trace) - offset)
    if stop - start < 3:
        return float("nan")
    mh = mh_trace[start:stop]
    sh = sh_trace[start + offset : stop + offset]
    mh_mean = sum(mh) / len(mh)
    sh_mean = sum(sh) / len(sh)
    numerator = sum((a - mh_mean) * (b - sh_mean) for a, b in zip(mh, sh))
    mh_energy = sum((value - mh_mean) ** 2 for value in mh)
    sh_energy = sum((value - sh_mean) ** 2 for value in sh)
    denominator = math.sqrt(mh_energy * sh_energy)
    if denominator <= 1.0e-12:
        return float("nan")
    return numerator / denominator


def estimate_motion_offset(
    mh_trace: list[float],
    sh_trace: list[float],
    *,
    max_offset: int = 12,
    min_correlation: float = 0.50,
    min_peak_prominence: float = 0.04,
) -> dict[str, Any]:
    """Estimate an integer SH offset, falling open to zero when ambiguous."""

    if max_offset < 1:
        raise ValueError("max_offset must be at least 1")
    scores = {
        offset: _pearson_at_offset(mh_trace, sh_trace, offset)
        for offset in range(-max_offset, max_offset + 1)
    }
    finite_scores = {
        offset: score for offset, score in scores.items() if math.isfinite(score)
    }
    if not finite_scores:
        return {
            "status": "ambiguous_fail_open",
            "estimated_camera1_frame_offset": None,
            "selected_camera1_frame_offset": 0,
            "reason": "no_finite_correlation_scores",
            "scores_by_offset": {str(key): None for key in scores},
        }

    best_offset = max(finite_scores, key=finite_scores.get)
    best_score = finite_scores[best_offset]
    outside_peak = [
        score
        for offset, score in finite_scores.items()
        if abs(offset - best_offset) > 1
    ]
    prominence_reference = max(outside_peak) if outside_peak else float("-inf")
    peak_prominence = best_score - prominence_reference
    accepted = (
        abs(best_offset) < max_offset
        and best_score >= min_correlation
        and peak_prominence >= min_peak_prominence
    )
    reasons: list[str] = []
    if abs(best_offset) == max_offset:
        reasons.append("best_peak_on_search_boundary")
    if best_score < min_correlation:
        reasons.append("correlation_below_threshold")
    if peak_prominence < min_peak_prominence:
        reasons.append("peak_not_distinct_from_non_neighbor_offsets")
    return {
        "status": "accepted" if accepted else "ambiguous_fail_open",
        "estimated_camera1_frame_offset": best_offset,
        "selected_camera1_frame_offset": best_offset if accepted else 0,
        "reason": "clear_motion_peak" if accepted else ",".join(reasons),
        "best_correlation": best_score,
        "peak_prominence_excluding_adjacent_offsets": peak_prominence,
        "thresholds": {
            "min_correlation": min_correlation,
            "min_peak_prominence": min_peak_prominence,
            "max_abs_offset": max_offset,
        },
        "scores_by_offset": {
            str(offset): score if math.isfinite(score) else None
            for offset, score in scores.items()
        },
    }


def audit_temporal_alignment(
    mh_video: Path,
    sh_video: Path,
    frame_limit: int,
    *,
    max_offset: int = 12,
) -> dict[str, Any]:
    """Run a lightweight cross-view motion audit with a fail-open policy."""

    base: dict[str, Any] = {
        "method": "32x18_grayscale_frame_difference_pearson",
        "offset_convention": "camera1/SH source index = camera2/MH frame k + offset",
        "frame_limit": frame_limit,
        "MH_source": str(mh_video.resolve()),
        "SH_source": str(sh_video.resolve()),
        "out_of_range_policy": "fail_open",
    }
    try:
        mh_trace = _decode_motion_energy(mh_video, frame_limit)
        sh_trace = _decode_motion_energy(sh_video, frame_limit)
        estimate = estimate_motion_offset(
            mh_trace,
            sh_trace,
            max_offset=max_offset,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        estimate = {
            "status": "audit_failed_fail_open",
            "estimated_camera1_frame_offset": None,
            "selected_camera1_frame_offset": 0,
            "reason": f"{type(error).__name__}: {error}",
            "scores_by_offset": {},
        }
    return {**base, **estimate}


def _validate_camera_mapping(mapping: dict[str, str]) -> None:
    if set(mapping) != {"camera_1", "camera_2"}:
        raise ValueError("calibration camera mapping must define camera_1 and camera_2")
    if set(mapping.values()) != {"MH", "SH"}:
        raise ValueError("calibration camera mapping must assign MH and SH exactly once")


def _matrix_intrinsics(payload: dict[str, Any], camera: str) -> dict[str, Any]:
    camera_payload = payload.get(camera)
    if not isinstance(camera_payload, dict):
        raise ValueError(f"calibration JSON is missing {camera}")
    matrix = camera_payload.get("camera_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError(f"calibration JSON {camera}.camera_matrix must be 3x3")
    try:
        fx = float(matrix[0][0])
        fy = float(matrix[1][1])
        cx = float(matrix[0][2])
        cy = float(matrix[1][2])
    except (TypeError, ValueError) as error:
        raise ValueError(f"calibration JSON {camera}.camera_matrix is not numeric") from error
    if min(fx, fy) <= 0:
        raise ValueError(f"calibration JSON {camera} has non-positive focal length")
    return {
        "camera_matrix": matrix,
        "distortion_k1_k2_p1_p2_k3": camera_payload.get(
            "distortion_k1_k2_p1_p2_k3"
        ),
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
        "rms_reprojection_px": camera_payload.get("rms_reprojection_px"),
    }


def _calibration_source_videos(payload: dict[str, Any]) -> tuple[str | None, dict[str, str]]:
    source = payload.get("source", {})
    source_path = source.get("path") if isinstance(source, dict) else None
    if not isinstance(source_path, str):
        return None, {}
    pairs_path = Path(source_path) / "pairs.json"
    if not pairs_path.is_file():
        return str(pairs_path.resolve()), {}
    pairs_payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    pairs = pairs_payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{pairs_path}: expected a non-empty pairs list")
    videos: dict[str, str] = {}
    for camera in ("camera_1", "camera_2"):
        values = {
            pair.get(camera, {}).get("video")
            for pair in pairs
            if isinstance(pair, dict) and isinstance(pair.get(camera), dict)
        }
        values.discard(None)
        if len(values) != 1:
            raise ValueError(
                f"{pairs_path}: expected one source video for {camera}, got {sorted(values)}"
            )
        videos[camera] = str(Path(values.pop()).resolve())
    return str(pairs_path.resolve()), videos


def load_calibration_metadata(
    calibration_json: Path | None,
    camera_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load calibration and make its camera namespace explicit."""

    mapping = dict(camera_mapping or DEFAULT_CALIBRATION_CAMERA_MAPPING)
    _validate_camera_mapping(mapping)
    pipeline_to_calibration = {
        pipeline_camera: next(
            calibration_camera
            for calibration_camera, view in mapping.items()
            if view == pipeline_view
        )
        for pipeline_camera, pipeline_view in PIPELINE_CAMERA_MAPPING.items()
    }
    base: dict[str, Any] = {
        "status": "not_provided",
        "reference_json": None,
        "reference_sha256": None,
        "calibration_camera_mapping": mapping,
        "pipeline_camera_mapping": dict(PIPELINE_CAMERA_MAPPING),
        "pipeline_to_calibration_camera": pipeline_to_calibration,
    }
    if calibration_json is None:
        return base

    path = calibration_json.resolve()
    if not path.is_file():
        raise ValueError(f"calibration JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration JSON root must be an object: {path}")
    intrinsics_by_calibration_camera = {
        camera: _matrix_intrinsics(payload, camera)
        for camera in ("camera_1", "camera_2")
    }
    intrinsics_by_view = {
        view: {
            "calibration_camera": camera,
            **intrinsics_by_calibration_camera[camera],
        }
        for camera, view in mapping.items()
    }
    pairs_manifest, calibration_source_videos = _calibration_source_videos(payload)
    source_videos_by_view = {
        mapping[camera]: video
        for camera, video in calibration_source_videos.items()
    }
    checkerboard = payload.get("checkerboard", {})
    quality = payload.get("quality", {})
    stereo = payload.get("stereo", {})
    base.update(
        {
            "status": "provided",
            "reference_json": str(path),
            "reference_sha256": _sha256(path),
            "schema_version": payload.get("schema_version"),
            "created_utc": payload.get("created_utc"),
            "quality_status": quality.get("status")
            if isinstance(quality, dict)
            else None,
            "quality_limitations": quality.get("limitations", [])
            if isinstance(quality, dict)
            else [],
            "image_size_wh": payload.get("source", {}).get("image_size_wh"),
            "checkerboard": {
                "square_size_mm": checkerboard.get("square_size_mm")
                if isinstance(checkerboard, dict)
                else None,
                "length_unit": checkerboard.get("length_unit")
                if isinstance(checkerboard, dict)
                else None,
                "metric_scale_verified": bool(
                    checkerboard.get("metric_scale_verified", False)
                )
                if isinstance(checkerboard, dict)
                else False,
            },
            "calibration_pair_manifest": pairs_manifest,
            "calibration_source_videos": source_videos_by_view,
            "intrinsics_by_view": intrinsics_by_view,
            "relative_extrinsics": {
                "from_calibration_camera": "camera_1",
                "to_calibration_camera": "camera_2",
                "from_view": mapping["camera_1"],
                "to_view": mapping["camera_2"],
                "T_camera2_from_camera1": stereo.get("T_camera2_from_camera1")
                if isinstance(stereo, dict)
                else None,
                "translation_unit": stereo.get("translation_unit")
                if isinstance(stereo, dict)
                else None,
            },
        }
    )
    return base


def _copy_gt_safely(source: Path, destination: Path) -> None:
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"refusing to replace different GT file: {destination}")
        return
    if destination.exists():
        raise ValueError(f"refusing to replace existing path: {destination}")
    shutil.copy2(source, destination)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare_episode(
    output_root: Path,
    source_pair: dict[str, Any],
    calibration: dict[str, Any],
    temporal_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(source_pair["episode"])
    gt_source = Path(source_pair["gt_labels"])
    mh_source = Path(source_pair["MH"])
    sh_source = Path(source_pair["SH"])
    gt = load_and_validate_gt(gt_source)
    if str(gt["episode"]) != name:
        raise ValueError(f"{gt_source}: episode field {gt['episode']!r} != {name!r}")
    expected = int(gt["num_frames"])
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

    _copy_gt_safely(gt_source, episode / "gt_labels.json")
    ensure_relative_symlink(episode / "rgb", camera2 / "rgb")
    ensure_relative_symlink(episode / "rgb_hawor", camera2 / "rgb_hawor")
    ensure_relative_symlink(episode / "contact", camera2 / "contact")
    ensure_relative_symlink(camera1 / "source.mov", sh_source)
    ensure_relative_symlink(camera2 / "source.mov", mh_source)

    intrinsics_by_view = calibration.get("intrinsics_by_view", {})
    pixel_focal = {
        view: intrinsics_by_view.get(view, {}).get("fx_px")
        for view in ("SH", "MH")
    }
    if temporal_audit is None:
        temporal_audit = {
            "status": "not_run_fail_open",
            "estimated_camera1_frame_offset": None,
            "selected_camera1_frame_offset": 0,
            "reason": "no temporal audit was supplied to prepare_episode",
            "out_of_range_policy": "fail_open",
        }
    camera1_frame_offset = int(
        temporal_audit.get("selected_camera1_frame_offset", 0)
    )
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "episode": name,
        "fps": 24.0,
        "common_frames": expected,
        "label_vocabulary": list(LABELS),
        "primary_view": "MH",
        "auxiliary_view": "SH",
        "stereo_code_mapping": dict(PIPELINE_CAMERA_MAPPING),
        "training_view": "MH",
        "robot_overlay_view": "MH",
        "source_pairing": {
            "method": "natural_order_by_annotation_MH_SH",
            "order_index_1_based": int(source_pair["order_index"]),
            "annotation_episode": name,
            "MH_filename": mh_source.name,
            "SH_filename": sh_source.name,
        },
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
            "source_frames_reordered": False,
            "pairing_basis": "same decoded frame index after natural-order video pairing",
            "apply_offset_only_during_dual_view_fusion": True,
            "motion_correlation_audit": temporal_audit,
            "out_of_range_policy": "fail_open",
        },
        "calibration": calibration,
        "intrinsics": {
            "status": calibration.get("status", "not_provided"),
            "pixel_focal_px": pixel_focal,
            "calibration_reference": calibration.get("reference_json"),
            "camera_namespace_note": (
                "pipeline camera_1=SH/camera_2=MH; calibration "
                "camera_1=MH/camera_2=SH unless overridden explicitly"
            ),
        },
        "frame_mapping": "output frame k equals decoded source frame k",
    }
    _write_json_atomic(episode / "stereo_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", "--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_root", "--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--calibration_json",
        "--calibration-json",
        type=Path,
        help="Optional stereo calibration JSON to reference in every manifest",
    )
    parser.add_argument(
        "--calibration_camera_1_view",
        "--calibration-camera-1-view",
        choices=("MH", "SH"),
        default="MH",
        help="Physical view represented by calibration camera_1 (default: MH)",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="Optional annotation episode names; default prepares every pair",
    )
    parser.add_argument(
        "--motion_max_offset",
        "--motion-max-offset",
        type=int,
        default=12,
        help="Maximum absolute SH-vs-MH frame offset checked before extraction",
    )
    parser.add_argument(
        "--motion_audit_only",
        "--motion-audit-only",
        action="store_true",
        help="Write preparation_audit.json after validation without extracting frames",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    available_pairs = discover_episode_sources(source_root)
    available = {str(pair["episode"]): pair for pair in available_pairs}
    episode_names = args.episodes or list(available)
    unknown = sorted(set(episode_names) - set(available), key=natural_key)
    if unknown:
        raise SystemExit(f"unknown/incomplete episodes: {unknown}")

    calibration_camera_1_view = args.calibration_camera_1_view
    calibration_camera_2_view = "SH" if calibration_camera_1_view == "MH" else "MH"
    calibration = load_calibration_metadata(
        args.calibration_json,
        {
            "camera_1": calibration_camera_1_view,
            "camera_2": calibration_camera_2_view,
        },
    )
    output_root.mkdir(parents=True, exist_ok=True)

    ordered_names = sorted(episode_names, key=natural_key)
    temporal_audits: dict[str, dict[str, Any]] = {}
    audit_episodes: list[dict[str, Any]] = []
    for name in ordered_names:
        source_pair = available[name]
        gt = load_and_validate_gt(Path(source_pair["gt_labels"]))
        expected = int(gt["num_frames"])
        mh_raw_frames = probe_frame_count(Path(source_pair["MH"]))
        sh_raw_frames = probe_frame_count(Path(source_pair["SH"]))
        if expected != min(mh_raw_frames, sh_raw_frames):
            raise ValueError(
                f"{name}: GT frames {expected} != common raw frames "
                f"min({mh_raw_frames}, {sh_raw_frames})"
            )
        temporal = audit_temporal_alignment(
            Path(source_pair["MH"]),
            Path(source_pair["SH"]),
            expected,
            max_offset=args.motion_max_offset,
        )
        temporal_audits[name] = temporal
        audit_episodes.append(
            {
                "episode": name,
                "source_pairing": {
                    "order_index_1_based": source_pair["order_index"],
                    "MH": str(source_pair["MH"]),
                    "SH": str(source_pair["SH"]),
                    "gt_labels": str(source_pair["gt_labels"]),
                },
                "common_frames": expected,
                "raw_frame_counts": {"MH": mh_raw_frames, "SH": sh_raw_frames},
                "temporal_alignment": temporal,
            }
        )
        print(
            f"[motion-audit] {name}: status={temporal['status']} "
            f"estimated={temporal.get('estimated_camera1_frame_offset')} "
            f"selected={temporal['selected_camera1_frame_offset']}",
            flush=True,
        )
    preparation_audit = {
        "schema_version": 1,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "label_vocabulary": list(LABELS),
        "pipeline_camera_mapping": dict(PIPELINE_CAMERA_MAPPING),
        "calibration_reference": calibration.get("reference_json"),
        "calibration_camera_mapping": calibration.get("calibration_camera_mapping"),
        "episodes": audit_episodes,
    }
    _write_json_atomic(output_root / "preparation_audit.json", preparation_audit)
    if args.motion_audit_only:
        print(
            f"Validated {len(ordered_names)} pairs and wrote motion audit to "
            f"{output_root / 'preparation_audit.json'}"
        )
        return

    total_frames = 0
    for index, name in enumerate(ordered_names, 1):
        manifest = prepare_episode(
            output_root,
            available[name],
            calibration,
            temporal_audits[name],
        )
        total_frames += int(manifest["common_frames"])
        dropped = manifest["tail_frames_dropped"]
        print(
            f"[{index:02d}/{len(ordered_names):02d}] {name}: "
            f"{manifest['common_frames']} frames "
            f"(drop MH={dropped['MH']}, SH={dropped['SH']})",
            flush=True,
        )
    print(
        f"Prepared {len(ordered_names)} episodes / {total_frames} common frames at "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()
