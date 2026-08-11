"""Compare finger-only and whole-XHand visual object barriers.

The comparison explicitly distinguishes compositing-only camera-Z barriers
from a physical collision solver.  All four panels use unchanged wrist,
finger, and RB5 state; the final three progressively extend the hidden region
to palm/base, a conservative XHand thickness shell, and a bounded temporal
closing pass.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import itertools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from make_video_comparison_grid import (
    GridLayout,
    NamedVideo,
    VideoMetadata,
    probe_video,
    render_comparison_grid_layout,
)


MODE_SPECS = (
    ("finger_best", "1 Current best: finger barrier"),
    ("whole_hand_zero", "2 Whole XHand: 0 mm"),
    ("whole_hand_shell", "3 Whole XHand: thickness shell"),
    ("whole_hand_shell_temporal", "4 XHand shell + Temporal <= 2f"),
)
GRID = GridLayout(columns=2, rows=2)
FULL_VIDEO_NAME = "video_compare_xhand_object_barrier_2x2.mp4"
ROI_VIDEO_NAME = "video_compare_xhand_object_barrier_roi_2x2.mp4"


def _load_source(directory: Path, mode: str) -> dict[str, object]:
    root = directory.expanduser().resolve()
    if mode == "finger_best":
        video = root / "video_overlay_contact.mp4"
        mask_path = root / "occluded_finger_mask.npy"
    else:
        video = root / "video_overlay_hand_barrier.mp4"
        mask_path = root / "occluded_hand_mask.npy"
    report_path = root / "report.json"
    for path in (video, mask_path, report_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{mode} source is incomplete: {path}")
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError(f"{mode} mask must be bool (T,H,W)")
    report = json.loads(report_path.read_text())
    metadata = probe_video(video)
    if len(mask) != metadata.frame_count:
        raise ValueError(f"{mode} mask/video frame mismatch")
    if int(report.get("frames", -1)) != len(mask):
        raise ValueError(f"{mode} report/mask frame mismatch")
    if mode == "finger_best":
        reported = int(report.get("occluded_pixels_total", -1))
    else:
        reported = int(report.get("counts", {}).get("final_occluded_pixels", -1))
    if reported != int(mask.sum()):
        raise ValueError(f"{mode} report/mask pixel mismatch")
    return {
        "root": root,
        "video": video,
        "mask": mask,
        "report": report,
        "metadata": metadata,
    }


def _metadata_signature(value: VideoMetadata) -> tuple[object, ...]:
    return (
        value.width,
        value.height,
        value.frame_count,
        value.fps,
        value.duration_s,
    )


def validate_lattice(masks: dict[str, np.ndarray]) -> None:
    required = {mode for mode, _label in MODE_SPECS}
    if set(masks) != required:
        raise ValueError("barrier comparison mask modes are incomplete")
    reference_shape = masks["finger_best"].shape
    if any(mask.shape != reference_shape for mask in masks.values()):
        raise ValueError("barrier comparison mask shapes differ")
    subset_pairs = (
        ("finger_best", "whole_hand_zero"),
        ("whole_hand_zero", "whole_hand_shell"),
        ("whole_hand_shell", "whole_hand_shell_temporal"),
    )
    for smaller_name, larger_name in subset_pairs:
        smaller = masks[smaller_name]
        larger = masks[larger_name]
        for frame_index in range(len(smaller)):
            violation = np.asarray(smaller[frame_index]) & ~np.asarray(
                larger[frame_index]
            )
            if np.any(violation):
                raise ValueError(
                    f"{smaller_name} is not a subset of {larger_name} at "
                    f"frame {frame_index}: {int(violation.sum())} pixels"
                )


def _validate_barrier_controls(sources: dict[str, dict[str, object]]) -> None:
    expected = {
        "whole_hand_zero": (0.0, 0.0, 0.0, 0),
        "whole_hand_shell": (0.01958, 0.01465, 0.015, 0),
        "whole_hand_shell_temporal": (0.01958, 0.01465, 0.015, 2),
    }
    baseline_report = sources["finger_best"]["report"]
    if not isinstance(baseline_report, dict):
        raise TypeError("baseline report must be a dictionary")
    baseline_config = baseline_report.get("config", {})
    if not (
        bool(baseline_config.get("object3d_force_surface", False))
        and int(baseline_config.get("object3d_temporal_max_gap_frames", 0)) == 2
    ):
        raise ValueError("finger baseline is not force-surface + temporal 2f")
    for mode, values in expected.items():
        report = sources[mode]["report"]
        if not isinstance(report, dict):
            raise TypeError(f"{mode} report must be a dictionary")
        if report.get("method") != "visual_camera_z_xhand_barrier":
            raise ValueError(f"{mode} is not a whole-XHand camera-Z barrier")
        if report.get("pose_state_modified") is not False:
            raise ValueError(f"{mode} unexpectedly modifies pose state")
        if report.get("metric_collision_guarantee") is not False:
            raise ValueError(f"{mode} overstates metric collision provenance")
        config = report.get("config", {})
        actual = (
            float(config.get("thumb_shell_m", -1.0)),
            float(config.get("finger_shell_m", -1.0)),
            float(config.get("palm_shell_m", -1.0)),
            int(config.get("temporal_max_gap_frames", -1)),
        )
        if not np.allclose(actual[:3], values[:3], atol=1.0e-9) or (
            actual[3] != values[3]
        ):
            raise ValueError(f"{mode} controls {actual} != {values}")
        if int(report.get("counts", {}).get("residual_violation_pixels", -1)) != 0:
            raise ValueError(f"{mode} has residual camera-Z violations")


def _statistics(sources: dict[str, dict[str, object]]) -> dict[str, object]:
    masks = {mode: source["mask"] for mode, source in sources.items()}
    modes = {
        mode: {
            "pixels": int(mask.sum()),
            "frames": int(
                sum(bool(np.any(mask[index])) for index in range(len(mask)))
            ),
        }
        for mode, mask in masks.items()
    }
    comparisons: dict[str, dict[str, int]] = {}
    for left, right in itertools.combinations((mode for mode, _ in MODE_SPECS), 2):
        added = 0
        removed = 0
        changed_frames = 0
        for frame_index in range(len(masks[left])):
            left_frame = np.asarray(masks[left][frame_index], dtype=bool)
            right_frame = np.asarray(masks[right][frame_index], dtype=bool)
            frame_added = int(np.sum(right_frame & ~left_frame))
            frame_removed = int(np.sum(left_frame & ~right_frame))
            added += frame_added
            removed += frame_removed
            changed_frames += int(frame_added > 0 or frame_removed > 0)
        comparisons[f"{left}_vs_{right}"] = {
            "added_pixels": added,
            "removed_pixels": removed,
            "changed_frames": changed_frames,
        }
    return {"modes": modes, "comparisons": comparisons}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_overlay_geometry(
    reference_dir: Path,
    barrier_dir: Path,
) -> dict[str, object]:
    """Verify geometry/semantic arrays are bit-identical after re-render."""
    reference = reference_dir.expanduser().resolve()
    barrier = barrier_dir.expanduser().resolve()
    names = (
        "robot_depth.npy",
        "robot_mask.npy",
        "robot_finger_labels.npy",
        "robot_finger_mask.npy",
    )
    result: dict[str, object] = {}
    for name in names:
        left = np.load(reference / name, mmap_mode="r", allow_pickle=False)
        right = np.load(barrier / name, mmap_mode="r", allow_pickle=False)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"overlay geometry contract differs for {name}")
        different = 0
        for frame_index in range(len(left)):
            different += int(
                np.count_nonzero(
                    np.asarray(left[frame_index]) != np.asarray(right[frame_index])
                )
            )
        if different:
            raise ValueError(f"overlay geometry changed in {name}: {different} values")
        result[name] = {"equal": True, "different_values": 0}

    hand = np.load(barrier / "robot_hand_mask.npy", mmap_mode="r", allow_pickle=False)
    robot = np.load(barrier / "robot_mask.npy", mmap_mode="r", allow_pickle=False)
    finger = np.load(
        barrier / "robot_finger_mask.npy", mmap_mode="r", allow_pickle=False
    )
    hand_not_robot = 0
    finger_not_hand = 0
    palm_base_pixels = 0
    for frame_index in range(len(hand)):
        hand_frame = np.asarray(hand[frame_index], dtype=bool)
        robot_frame = np.asarray(robot[frame_index], dtype=bool)
        finger_frame = np.asarray(finger[frame_index], dtype=bool)
        hand_not_robot += int(np.sum(hand_frame & ~robot_frame))
        finger_not_hand += int(np.sum(finger_frame & ~hand_frame))
        palm_base_pixels += int(np.sum(hand_frame & ~finger_frame))
    if hand_not_robot or finger_not_hand:
        raise ValueError("whole-XHand semantic mask violates renderer subsets")
    result["robot_hand_mask.npy"] = {
        "hand_not_robot_pixels": hand_not_robot,
        "finger_not_hand_pixels": finger_not_hand,
        "palm_base_pixels": palm_base_pixels,
    }
    return result


def dynamic_roi_centers(
    object_mask: np.ndarray,
    *,
    crop_width: int,
    crop_height: int,
    smooth_window: int = 9,
) -> np.ndarray:
    """Return interpolated/smoothed object centres for a shared video crop."""
    mask = np.asanyarray(object_mask)
    if mask.ndim != 3:
        raise ValueError("object mask must have shape (T,H,W)")
    frame_count, height, width = mask.shape
    if not (0 < crop_width <= width and 0 < crop_height <= height):
        raise ValueError("dynamic crop size is outside the source frame")
    if smooth_window <= 0 or smooth_window % 2 == 0:
        raise ValueError("smooth window must be a positive odd number")
    centres = np.full((frame_count, 2), np.nan, dtype=np.float64)
    for frame_index in range(frame_count):
        rows, columns = np.nonzero(np.asarray(mask[frame_index], dtype=bool))
        if len(rows):
            centres[frame_index] = (
                0.5 * (float(columns.min()) + float(columns.max())),
                0.5 * (float(rows.min()) + float(rows.max())),
            )
    indices = np.arange(frame_count)
    for axis, fallback in ((0, 0.5 * (width - 1)), (1, 0.5 * (height - 1))):
        valid = np.isfinite(centres[:, axis])
        if np.any(valid):
            centres[:, axis] = np.interp(
                indices,
                indices[valid],
                centres[valid, axis],
            )
        else:
            centres[:, axis] = fallback
    if smooth_window > 1 and frame_count > 1:
        window = min(smooth_window, frame_count if frame_count % 2 else frame_count - 1)
        if window > 1:
            kernel = np.ones(window, dtype=np.float64) / float(window)
            radius = window // 2
            for axis in range(2):
                padded = np.pad(centres[:, axis], (radius, radius), mode="edge")
                centres[:, axis] = np.convolve(padded, kernel, mode="valid")
    half_width = crop_width / 2.0
    half_height = crop_height / 2.0
    centres[:, 0] = np.clip(centres[:, 0], half_width, width - half_width)
    centres[:, 1] = np.clip(centres[:, 1], half_height, height - half_height)
    return centres


def _header(frame: np.ndarray, label: str, header_height: int = 40) -> np.ndarray:
    height, width = frame.shape[:2]
    output = np.zeros((height + header_height, width, 3), dtype=np.uint8)
    output[header_height:] = frame
    scale = min(0.65, max(0.42, width / 1000.0))
    cv2.putText(
        output,
        label,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def render_dynamic_roi_grid(
    videos: list[NamedVideo],
    output_path: Path,
    *,
    centres: np.ndarray,
    crop_width: int,
    crop_height: int,
    fps: float,
) -> VideoMetadata:
    if len(videos) != 4:
        raise ValueError("dynamic ROI grid requires four videos")
    captures = [cv2.VideoCapture(str(video.path)) for video in videos]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise FileNotFoundError("could not open one or more ROI source videos")
    frame_count = len(centres)
    header_height = 40
    output_size = (crop_width * 2, (crop_height + header_height) * 2)
    temporary = output_path.with_name(f".{output_path.stem}.mpeg4.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        for capture in captures:
            capture.release()
        raise RuntimeError(f"could not open ROI writer: {temporary}")
    try:
        for frame_index, (center_x, center_y) in enumerate(centres):
            x0 = int(round(center_x - crop_width / 2.0))
            y0 = int(round(center_y - crop_height / 2.0))
            panels = []
            for capture, video in zip(captures, videos):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"ROI video {video.path} ended at frame {frame_index}"
                    )
                crop = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]
                if crop.shape[:2] != (crop_height, crop_width):
                    raise RuntimeError("dynamic ROI crop escaped the source frame")
                panels.append(_header(crop, video.label, header_height))
            writer.write(
                np.concatenate(
                    (
                        np.concatenate((panels[0], panels[1]), axis=1),
                        np.concatenate((panels[2], panels[3]), axis=1),
                    ),
                    axis=0,
                )
            )
            if (frame_index + 1) % 100 == 0:
                print(f"[roi-grid] {frame_index + 1}/{frame_count}", flush=True)
    finally:
        for capture in captures:
            capture.release()
        writer.release()

    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ),
        check=True,
    )
    temporary.unlink(missing_ok=True)
    metadata = probe_video(output_path)
    if (
        metadata.width != output_size[0]
        or metadata.height != output_size[1]
        or metadata.frame_count != frame_count
        or abs(metadata.fps - fps) > 1.0e-6
    ):
        raise RuntimeError("dynamic ROI comparison metadata mismatch")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finger_best_dir", type=Path, required=True)
    parser.add_argument("--whole_hand_zero_dir", type=Path, required=True)
    parser.add_argument("--whole_hand_shell_dir", type=Path, required=True)
    parser.add_argument("--whole_hand_shell_temporal_dir", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--reference_overlay_dir", type=Path, required=True)
    parser.add_argument("--barrier_overlay_dir", type=Path, required=True)
    parser.add_argument("--overlay_input", type=Path, required=True)
    parser.add_argument("--joint_names", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    directories = {
        "finger_best": args.finger_best_dir,
        "whole_hand_zero": args.whole_hand_zero_dir,
        "whole_hand_shell": args.whole_hand_shell_dir,
        "whole_hand_shell_temporal": args.whole_hand_shell_temporal_dir,
    }
    sources = {
        mode: _load_source(directory, mode)
        for mode, directory in directories.items()
    }
    reference_metadata = sources["finger_best"]["metadata"]
    assert isinstance(reference_metadata, VideoMetadata)
    reference_mask = sources["finger_best"]["mask"]
    assert isinstance(reference_mask, np.ndarray)
    for mode, source in sources.items():
        metadata = source["metadata"]
        mask = source["mask"]
        assert isinstance(metadata, VideoMetadata)
        assert isinstance(mask, np.ndarray)
        if _metadata_signature(metadata) != _metadata_signature(reference_metadata):
            raise ValueError(f"{mode} video metadata differs")
        if mask.shape != reference_mask.shape:
            raise ValueError(f"{mode} mask shape differs")
    _validate_barrier_controls(sources)
    validate_lattice(
        {
            mode: source["mask"]
            for mode, source in sources.items()
            if isinstance(source["mask"], np.ndarray)
        }
    )
    geometry_verification = verify_overlay_geometry(
        args.reference_overlay_dir,
        args.barrier_overlay_dir,
    )

    object_mask_path = args.object_mask.expanduser().resolve()
    object_mask = np.load(object_mask_path, mmap_mode="r", allow_pickle=False)
    if object_mask.shape != reference_mask.shape:
        raise ValueError("dynamic ROI object mask shape differs")
    videos = [
        NamedVideo(label=label, path=Path(sources[mode]["video"]))
        for mode, label in MODE_SPECS
    ]
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".xhand_barrier_compare.", dir=output_dir.parent)
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    full_metadata = render_comparison_grid_layout(
        videos,
        staging / FULL_VIDEO_NAME,
        layout=GRID,
        overwrite=False,
        crf=18,
        preset="medium",
    )
    centres = dynamic_roi_centers(
        object_mask,
        crop_width=640,
        crop_height=360,
        smooth_window=9,
    )
    roi_metadata = render_dynamic_roi_grid(
        videos,
        staging / ROI_VIDEO_NAME,
        centres=centres,
        crop_width=640,
        crop_height=360,
        fps=float(reference_metadata.fps),
    )
    statistics = _statistics(sources)
    overlay_input = args.overlay_input.expanduser().resolve()
    joint_names = args.joint_names.expanduser().resolve()
    for path in (overlay_input, joint_names):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = {
        "schema_version": 1,
        "comparison": "whole-XHand visual camera-Z object barrier",
        "physical_collision_solver": False,
        "pose_state_modified": False,
        "metric_collision_guarantee": False,
        "layout": [
            [MODE_SPECS[0][0], MODE_SPECS[1][0]],
            [MODE_SPECS[2][0], MODE_SPECS[3][0]],
        ],
        "definitions": {
            "finger_best": "finger-only surface-force plus temporal <=2f",
            "whole_hand_zero": "finger baseline OR whole-XHand Z > visible object Z",
            "whole_hand_shell": "zero barrier plus half-thickness finger and 15mm palm shell",
            "whole_hand_shell_temporal": "thickness barrier plus offline temporal <=2f",
        },
        "videos": {
            "full_frame": FULL_VIDEO_NAME,
            "dynamic_roi": ROI_VIDEO_NAME,
        },
        "metadata": {
            "full_frame": {
                "width": full_metadata.width,
                "height": full_metadata.height,
                "frames": full_metadata.frame_count,
                "fps": float(full_metadata.fps),
            },
            "dynamic_roi": {
                "width": roi_metadata.width,
                "height": roi_metadata.height,
                "frames": roi_metadata.frame_count,
                "fps": float(roi_metadata.fps),
                "source_crop": [640, 360],
                "center_policy": "object bbox interpolation plus 9-frame moving average",
            },
        },
        "sources": {
            mode: str(source["root"])
            for mode, source in sources.items()
        },
        "geometry_verification": geometry_verification,
        "trajectory_provenance": {
            "overlay_input": str(overlay_input),
            "overlay_input_sha256": _sha256(overlay_input),
            "joint_names": str(joint_names),
            "joint_names_sha256": _sha256(joint_names),
            "unchanged_for_all_panels": True,
        },
        "invariants": {
            "finger_best_subset_whole_hand_zero": True,
            "whole_hand_zero_subset_shell": True,
            "whole_hand_shell_subset_temporal": True,
            "renderer_geometry_arrays_bit_identical": True,
            "whole_hand_mask_excludes_rb5": True,
            "all_barrier_residual_violations_zero": True,
        },
        **statistics,
        "provenance_warning": (
            "Visible HaWoR-Z anchored camera-depth surface only; not a "
            "watertight mesh, object SDF, or physical pose-level collision solve."
        ),
    }
    (staging / "comparison_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] XHand barrier comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
