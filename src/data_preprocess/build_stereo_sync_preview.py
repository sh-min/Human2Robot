"""Build a side-by-side stereo annotation preview and synchronization QA.

The input episode is produced by ``extract_stereo_realsense_db3.py`` and must
contain synchronized ``camera_1/rgb``, ``camera_2/rgb``, and
``stereo_pairs.csv`` entries.  The output video uses one common frame index,
shows both source frame numbers and their capture-time difference, and is
therefore suitable for one shared temporal annotation.

In addition to timestamp statistics, the script compares per-frame motion
energy between the two views.  A correlation peak at zero frames is useful as
an independent visual-content check that timestamps were paired in the right
order.  It is not a substitute for hardware synchronization or a strobe-based
calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"no stereo pairs in {path}")
    parsed: list[dict[str, Any]] = []
    for expected_index, row in enumerate(rows):
        output_index = int(row["output_index"])
        if output_index != expected_index:
            raise ValueError(
                f"non-contiguous output index in {path}: "
                f"expected {expected_index}, got {output_index}"
            )
        parsed.append(
            {
                "output_index": output_index,
                "camera_1_source_index": int(row["camera_1_source_index"]),
                "camera_1_frame_number": int(row["camera_1_frame_number"]),
                "camera_1_timestamp_s": float(row["camera_1_timestamp_s"]),
                "camera_2_source_index": int(row["camera_2_source_index"]),
                "camera_2_frame_number": int(row["camera_2_frame_number"]),
                "camera_2_timestamp_s": float(row["camera_2_timestamp_s"]),
                "camera_2_minus_camera_1_ms": float(
                    row["camera_2_minus_camera_1_ms"]
                ),
            }
        )
    return parsed


def _rgb_paths(episode: Path, camera_id: int) -> list[Path]:
    paths = sorted((episode / f"camera_{camera_id}" / "rgb").glob("*.png"))
    if not paths:
        raise FileNotFoundError(
            f"no RGB PNGs in {episode / f'camera_{camera_id}' / 'rgb'}"
        )
    return paths


def _motion_energy(paths: list[Path]) -> np.ndarray:
    """Return robust global motion energy for every inter-frame interval."""

    energies: list[float] = []
    previous = None
    top_count = int(160 * 90 * 0.05)
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot read {path}")
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if previous is not None:
            difference = cv2.absdiff(gray, previous).astype(np.float32).ravel()
            strongest = np.partition(difference, -top_count)[-top_count:]
            energies.append(float(strongest.mean()))
        previous = gray
    return np.asarray(energies, dtype=np.float64)


def _standardize(signal: np.ndarray) -> np.ndarray:
    centered = signal - np.median(signal)
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError("motion signal has no usable variation")
    return centered / scale


def _lagged_signals(
    camera_1: np.ndarray, camera_2: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Align signals; positive lag means camera 2 is shifted later."""

    if lag < 0:
        return camera_1[-lag:], camera_2[: len(camera_2) + lag]
    if lag > 0:
        return camera_1[:-lag], camera_2[lag:]
    return camera_1, camera_2


def analyze_motion_sync(
    camera_1_paths: list[Path],
    camera_2_paths: list[Path],
    max_lag_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(camera_1_paths) != len(camera_2_paths):
        raise ValueError(
            f"camera frame-count mismatch: {len(camera_1_paths)} != {len(camera_2_paths)}"
        )
    motion_1 = _standardize(_motion_energy(camera_1_paths))
    motion_2 = _standardize(_motion_energy(camera_2_paths))
    lags = list(range(-max_lag_frames, max_lag_frames + 1))
    correlations: list[float] = []
    for lag in lags:
        first, second = _lagged_signals(motion_1, motion_2, lag)
        if len(first) < 3:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(first, second)[0, 1])
        correlations.append(correlation)
    best_position = int(np.nanargmax(correlations))
    best_lag = lags[best_position]
    subframe_lag = float(best_lag)
    if 0 < best_position < len(correlations) - 1:
        left = correlations[best_position - 1]
        center = correlations[best_position]
        right = correlations[best_position + 1]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1.0e-12:
            subframe_lag += 0.5 * (left - right) / denominator
    analysis = {
        "lag_convention": "positive means camera_2 motion is shifted later",
        "lags_frames": lags,
        "correlations": correlations,
        "best_integer_lag_frames": best_lag,
        "parabolic_subframe_lag_frames": subframe_lag,
        "best_correlation": correlations[best_position],
        "zero_lag_correlation": correlations[lags.index(0)],
    }
    return motion_1, motion_2, analysis


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR image, got {image.shape}")
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _label_panel(
    panel: np.ndarray,
    title: str,
    source_frame_number: int,
    timestamp_s: float,
    color: tuple[int, int, int],
) -> None:
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, panel, 0.38, 0.0, panel)
    cv2.putText(
        panel,
        title,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
    text = f"source frame {source_frame_number}"
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    cv2.putText(
        panel,
        text,
        (panel.shape[1] - size[0] - 14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"global t={timestamp_s:.6f} s",
        (14, panel.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def build_preview(
    camera_1_paths: list[Path],
    camera_2_paths: list[Path],
    pairs: list[dict[str, Any]],
    output: Path,
    fps: float,
    panel_width: int,
    panel_height: int,
) -> None:
    frame_count = len(pairs)
    if len(camera_1_paths) != frame_count or len(camera_2_paths) != frame_count:
        raise ValueError(
            "preview input count mismatch: "
            f"pairs={frame_count}, camera_1={len(camera_1_paths)}, "
            f"camera_2={len(camera_2_paths)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.tmp-", suffix=output.suffix, dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    frame_width = panel_width * 2
    info_height = 64
    frame_height = panel_height + info_height
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{frame_width}x{frame_height}",
        "-r",
        f"{fps:.9f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-g",
        "15",
        "-keyint_min",
        "15",
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-frames:v",
        str(frame_count),
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        for frame_index, (camera_1_path, camera_2_path, pair) in enumerate(
            zip(camera_1_paths, camera_2_paths, pairs)
        ):
            camera_1 = cv2.imread(str(camera_1_path), cv2.IMREAD_COLOR)
            camera_2 = cv2.imread(str(camera_2_path), cv2.IMREAD_COLOR)
            if camera_1 is None or camera_2 is None:
                raise ValueError(
                    f"cannot read preview inputs: {camera_1_path}, {camera_2_path}"
                )
            left = _letterbox(camera_1, panel_width, panel_height)
            right = _letterbox(camera_2, panel_width, panel_height)
            _label_panel(
                left,
                "CAMERA 1  |  D455  |  SIDE VIEW",
                pair["camera_1_frame_number"],
                pair["camera_1_timestamp_s"],
                (80, 220, 255),
            )
            _label_panel(
                right,
                "CAMERA 2  |  D435I  |  EGO VIEW",
                pair["camera_2_frame_number"],
                pair["camera_2_timestamp_s"],
                (120, 255, 120),
            )
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            canvas[:panel_height, :panel_width] = left
            canvas[:panel_height, panel_width:] = right
            cv2.line(
                canvas,
                (panel_width, 0),
                (panel_width, panel_height),
                (255, 255, 255),
                2,
            )
            delta_ms = pair["camera_2_minus_camera_1_ms"]
            absolute_delta_ms = abs(delta_ms)
            if absolute_delta_ms <= 1000.0 / fps / 4.0:
                sync_color = (80, 220, 80)
                sync_band = "GREEN <= 1/4 frame"
            elif absolute_delta_ms <= 1000.0 / fps / 2.0:
                sync_color = (0, 190, 255)
                sync_band = "AMBER <= 1/2 frame"
            else:
                sync_color = (60, 60, 255)
                sync_band = "RED > 1/2 frame"
            cv2.rectangle(
                canvas,
                (1, 1),
                (frame_width - 2, panel_height - 2),
                sync_color,
                3,
            )
            info = (
                f"COMMON FRAME {frame_index:04d}/{frame_count - 1:04d}"
                f"    camera2 - camera1 = {delta_ms:+.2f} ms"
                f"    {sync_band}"
            )
            cv2.putText(
                canvas,
                info,
                (20, panel_height + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            elapsed_s = (
                pair["camera_2_timestamp_s"] - pairs[0]["camera_2_timestamp_s"]
            )
            cv2.putText(
                canvas,
                f"elapsed={elapsed_s:7.3f} s    frame period={1000.0 / fps:.2f} ms",
                (20, panel_height + 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            process.stdin.write(canvas.tobytes())
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_count:
                print(f"[preview] {frame_index + 1}/{frame_count}")
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
        os.replace(temporary, output)
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.terminate()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise


def _write_plot(
    output: Path,
    motion_1: np.ndarray,
    motion_2: np.ndarray,
    analysis: dict[str, Any],
    pairs: list[dict[str, Any]],
    fps: float,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].plot(np.arange(1, len(motion_1) + 1), motion_1, label="camera 1")
    axes[0].plot(
        np.arange(1, len(motion_2) + 1), motion_2, label="camera 2", alpha=0.82
    )
    axes[0].set_title("Normalized inter-frame motion energy")
    axes[0].set_xlabel("common frame index")
    axes[0].set_ylabel("normalized motion")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        analysis["lags_frames"], analysis["correlations"], marker="o"
    )
    axes[1].axvline(0, color="black", linewidth=1, alpha=0.5)
    axes[1].scatter(
        [analysis["best_integer_lag_frames"]],
        [analysis["best_correlation"]],
        color="red",
        zorder=3,
        label=(
            f"peak lag={analysis['best_integer_lag_frames']} frame, "
            f"r={analysis['best_correlation']:.3f}"
        ),
    )
    axes[1].set_title("Cross-view motion correlation by temporal lag")
    axes[1].set_xlabel("camera 2 lag relative to camera 1 (frames)")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    elapsed = np.asarray(
        [
            row["camera_2_timestamp_s"] - pairs[0]["camera_2_timestamp_s"]
            for row in pairs
        ]
    )
    offsets = np.asarray([row["camera_2_minus_camera_1_ms"] for row in pairs])
    axes[2].plot(elapsed, offsets, linewidth=1.2, label="camera 2 - camera 1")
    quarter_frame = 250.0 / fps
    half_frame = 500.0 / fps
    for value, color, label in (
        (quarter_frame, "#d97706", "quarter-frame threshold"),
        (-quarter_frame, "#d97706", None),
        (half_frame, "#dc2626", "half-frame threshold"),
        (-half_frame, "#dc2626", None),
    ):
        axes[2].axhline(value, color=color, linestyle="--", linewidth=1, label=label)
    axes[2].set_title("Timestamp offset and clock drift")
    axes[2].set_xlabel("elapsed time (s)")
    axes[2].set_ylabel("camera 2 - camera 1 (ms)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    episode = args.episode.resolve()
    camera_1_paths = _rgb_paths(episode, 1)
    camera_2_paths = _rgb_paths(episode, 2)
    pairs = _read_pairs(episode / "stereo_pairs.csv")
    if len(camera_1_paths) != len(pairs) or len(camera_2_paths) != len(pairs):
        raise ValueError(
            f"frame/pair mismatch: camera_1={len(camera_1_paths)}, "
            f"camera_2={len(camera_2_paths)}, pairs={len(pairs)}"
        )

    motion_1, motion_2, motion = analyze_motion_sync(
        camera_1_paths, camera_2_paths, args.max_lag_frames
    )
    offsets_ms = [row["camera_2_minus_camera_1_ms"] for row in pairs]
    elapsed_s = np.asarray(
        [
            row["camera_2_timestamp_s"] - pairs[0]["camera_2_timestamp_s"]
            for row in pairs
        ],
        dtype=np.float64,
    )
    offset_array = np.asarray(offsets_ms, dtype=np.float64)
    drift_slope_ms_per_s, drift_intercept_ms = np.polyfit(
        elapsed_s, offset_array, 1
    )
    absolute_offsets = np.abs(offset_array)
    half_frame_ms = 500.0 / args.fps
    timestamp_pass = max(abs(value) for value in offsets_ms) <= half_frame_ms
    motion_pass = (
        motion["best_integer_lag_frames"] == 0
        and motion["zero_lag_correlation"] >= args.min_motion_correlation
    )
    report = {
        "schema_version": 1,
        "episode": str(episode),
        "frame_count": len(pairs),
        "fps": args.fps,
        "frame_period_ms": 1000.0 / args.fps,
        "hardware_synchronized": False,
        "timestamp_pairing": {
            "camera_2_minus_camera_1_ms": {
                "min": min(offsets_ms),
                "median": statistics.median(offsets_ms),
                "mean": float(np.mean(offset_array)),
                "p95_abs": float(np.percentile(absolute_offsets, 95)),
                "max": max(offsets_ms),
                "max_abs": max(abs(value) for value in offsets_ms),
            },
            "drift_fit": {
                "intercept_ms": float(drift_intercept_ms),
                "slope_ms_per_s": float(drift_slope_ms_per_s),
                "slope_ppm_approx": float(drift_slope_ms_per_s * 1000.0),
            },
            "quality_bands": {
                "green_le_quarter_frame": int(
                    np.sum(absolute_offsets <= 250.0 / args.fps)
                ),
                "amber_between_quarter_and_half_frame": int(
                    np.sum(
                        (absolute_offsets > 250.0 / args.fps)
                        & (absolute_offsets <= half_frame_ms)
                    )
                ),
                "red_gt_half_frame": int(np.sum(absolute_offsets > half_frame_ms)),
            },
            "criterion": "maximum absolute offset <= half a frame",
            "half_frame_ms": half_frame_ms,
            "pass": timestamp_pass,
        },
        "motion_content_check": motion,
        "motion_lag_estimate_ms": motion["parabolic_subframe_lag_frames"]
        * 1000.0
        / args.fps,
        "motion_criterion": (
            "integer correlation peak at zero frames and zero-lag correlation "
            f">= {args.min_motion_correlation:.3f}"
        ),
        "motion_pass": motion_pass,
        "status": "pass" if timestamp_pass and motion_pass else "review",
        "limitations": [
            "The cameras used global timestamps but hardware sync was disabled.",
            "Motion correlation is a content-based sanity check, not sub-frame strobe validation.",
        ],
    }

    output = (
        args.output.resolve()
        if args.output is not None
        else episode / "_stereo_sync_preview.mp4"
    )
    report_path = (
        args.report.resolve()
        if args.report is not None
        else episode / "stereo_sync_report.json"
    )
    plot_path = (
        args.plot.resolve()
        if args.plot is not None
        else episode / "stereo_sync_motion.png"
    )
    build_preview(
        camera_1_paths,
        camera_2_paths,
        pairs,
        output,
        args.fps,
        args.panel_width,
        args.panel_height,
    )
    _write_plot(plot_path, motion_1, motion_2, motion, pairs, args.fps)
    report["preview_video"] = str(output)
    report["motion_plot"] = str(plot_path)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[ok] preview: {output}")
    print(f"[ok] report: {report_path}")
    print(f"[sync] status={report['status']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-lag-frames", type=int, default=10)
    parser.add_argument("--min-motion-correlation", type=float, default=0.5)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
