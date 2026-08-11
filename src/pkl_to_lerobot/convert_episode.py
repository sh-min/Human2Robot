"""Convert a single retargeted episode (pkl + RGB) to LeRobot dataset files.

Emits LeRobot v3.0 layout (one episode per file, file_index = episode_index).
Use convert_batch.py to assemble metadata across episodes.

Input directory layout (produced by run_pipeline.sh):
    <episode>/
        rgb/frame_*.jpg
        rgb_hawor/
            final_pose.pkl           (preferred: has wrist pose)
            qpos_xhand_contact_right.pkl  (fallback: finger-only)
            qpos_xhand_contact_left.pkl

Output (v3 layout, one episode):
    <out>/data/chunk-000/file-NNN.parquet
    <out>/videos/observation.images.head_cam/chunk-000/file-NNN.mp4

Usage:
    python -m pkl_to_lerobot.convert_episode \\
        --episode_dir /path/to/episode \\
        --out_dir /path/to/dataset \\
        --episode_index 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import subprocess

HEAD_CAM_KEY = "observation.images.head_cam"
TARGET_IMG_SIZE = 224  # Video frames are downscaled to this square size.
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    BIMANUAL_DIM,
    HANDS,
    final_pose_to_state_action,
    pkl_to_state_action,
    states_to_actions,
)

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _sorted_frames(rgb_dir: str, img_glob: str = "frame_*.jpg") -> list[Path]:
    """Return sorted RGB frame paths, ordered by numeric index in filename."""
    pattern = os.path.join(rgb_dir, img_glob)
    paths = sorted(glob.glob(pattern))
    if not paths:
        for ext in _IMG_EXTENSIONS:
            alt = os.path.join(rgb_dir, f"*{ext}")
            paths = sorted(glob.glob(alt))
            if paths:
                break
    paths = [Path(p) for p in paths]

    def _num(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    return sorted(paths, key=_num)


def _load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_pkl(episode_dir: str, hand: str) -> str | None:
    """Search for the retarget pkl for ``hand`` under episode_dir/rgb_hawor/."""
    hawor = os.path.join(episode_dir, "rgb_hawor")
    candidates = [
        os.path.join(hawor, f"qpos_xhand_contact_{hand}_smooth.pkl"),
        os.path.join(hawor, f"qpos_xhand_contact_{hand}.pkl"),
        os.path.join(hawor, f"qpos_xhand_{hand}_smooth.pkl"),
        os.path.join(hawor, f"qpos_xhand_{hand}.pkl"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def frames_to_mp4(
    frame_paths: list[Path],
    out_path: Path,
    fps: float = 30.0,
) -> None:
    """Encode a sequence of image files into an MP4 using ffmpeg.

    Creates a temporary file list to handle arbitrary filenames without
    requiring a sequential naming pattern.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.with_suffix(".ffconcat")
    try:
        with open(list_file, "w") as f:
            for p in frame_paths:
                f.write(f"file '{p.resolve()}'\n")
                f.write(f"duration {1.0/fps:.6f}\n")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-vf", (
                f"scale={TARGET_IMG_SIZE}:{TARGET_IMG_SIZE}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={TARGET_IMG_SIZE}:{TARGET_IMG_SIZE}:(ow-iw)/2:(oh-ih)/2"
            ),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(int(fps)),
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        if list_file.exists():
            list_file.unlink()


def _robot_video_path(episode_dir: str) -> Path | None:
    episode = Path(episode_dir)
    base = episode / "inpainting_processed" / episode.name / "0"
    for name in (
        "video_overlay_rby1_xhand.mp4",
        "video_overlay_rby1_xhand.mkv",
    ):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,nb_frames",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if value in (None, "N/A"):
        raise ValueError(f"Could not determine frame count for {path}")
    return int(value)


def video_to_mp4(
    source: Path,
    frame_indices: np.ndarray,
    out_path: Path,
    fps: float = 30.0,
) -> None:
    """Select aligned frames from a source video and encode LeRobot video."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError(f"No video frames selected from {source}")
    contiguous = np.array_equal(indices, np.arange(len(indices)))
    filters = []
    if not contiguous:
        select = "+".join(f"eq(n\\,{int(index)})" for index in indices)
        filters.extend([f"select={select}", f"setpts=N/({fps}*TB)"])
    filters.extend([
        (
            f"scale={TARGET_IMG_SIZE}:{TARGET_IMG_SIZE}:"
            "force_original_aspect_ratio=decrease"
        ),
        f"pad={TARGET_IMG_SIZE}:{TARGET_IMG_SIZE}:(ow-iw)/2:(oh-ih)/2",
    ])
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(source),
            "-an",
            "-vf", ",".join(filters),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(int(fps)),
            str(out_path),
        ],
        check=True,
    )


def _resolve_visual_source(
    episode_dir: str,
    visual_source: str,
) -> tuple[str, Path | None]:
    if visual_source == "rgb":
        return "rgb", None
    if visual_source in ("auto", "robot"):
        robot = _robot_video_path(episode_dir)
        if robot is not None:
            return "robot", robot
        if visual_source == "robot":
            raise FileNotFoundError(
                f"No video_overlay_rby1_xhand mp4/mkv in {episode_dir}"
            )
        return "rgb", None
    source = Path(visual_source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Visual source does not exist: {source}")
    return "video", source


def convert_episode(
    episode_dir: str,
    out_dir: str,
    episode_index: int,
    *,
    fps: float = 30.0,
    action_mode: str = "absolute",
    img_glob: str = "frame_*.jpg",
    task_description: str = "manipulate cube",
    chunk: int = 0,
    visual_source: str = "auto",
    allow_legacy_actions: bool = False,
) -> dict:
    """Convert one episode and write parquet + mp4.

    Returns a metadata dict for the episode (for episodes.jsonl).
    """
    episode_dir = str(Path(episode_dir).resolve())
    out_dir = str(Path(out_dir).resolve())
    rgb_dir = os.path.join(episode_dir, "rgb")

    # --- Load action data ---
    # Prefer final_pose.pkl (contains wrist pose + finger qpos for both hands).
    # Fall back to per-hand qpos pkl files (finger-only, no wrist).
    hawor_dir = os.path.join(episode_dir, "rgb_hawor")
    final_pose_path = os.path.join(hawor_dir, "final_pose.pkl")

    if os.path.isfile(final_pose_path):
        final_pose = _load_pkl(final_pose_path)
        print(f"  loaded final_pose.pkl (T={final_pose['T']})")
        if (
            final_pose.get("coordinate_frame") != "rby1_base"
            and not allow_legacy_actions
        ):
            raise ValueError(
                f"{final_pose_path} is not calibrated to rby1_base. "
                "Run scripts/export_policy_trajectories.sh first."
            )
        states, actions, valid = final_pose_to_state_action(
            final_pose, action_mode=action_mode
        )
    else:
        if not allow_legacy_actions:
            raise FileNotFoundError(
                f"Calibrated final_pose.pkl missing: {final_pose_path}. "
                "Run scripts/export_policy_trajectories.sh first."
            )
        pkls: dict[str, dict] = {}
        for hand in HANDS:
            pkl_path = _find_pkl(episode_dir, hand)
            if pkl_path is not None:
                pkls[hand] = _load_pkl(pkl_path)
                print(f"  [{hand}] loaded {pkl_path}")
        if not pkls:
            raise FileNotFoundError(
                f"No final_pose.pkl or retarget pkl found in {hawor_dir}/"
            )
        states, actions, valid = pkl_to_state_action(pkls, action_mode=action_mode)

    T = states.shape[0]

    # --- Load and align the visual observation ---
    visual_kind, visual_video = _resolve_visual_source(
        episode_dir, visual_source
    )
    frames: list[Path] = []
    if visual_video is not None:
        source_frames = _video_frame_count(visual_video)
        print(f"  visual -> {visual_video} ({source_frames} frames)")
    else:
        frames = _sorted_frames(rgb_dir, img_glob)
        if len(frames) == 0:
            raise FileNotFoundError(
                f"No RGB frames in {rgb_dir} (glob={img_glob})"
            )
        source_frames = len(frames)
        print(f"  visual -> {rgb_dir} ({source_frames} frames)")

    # pkl T may differ from frame count if HaWoR trimmed the sequence.
    # Align to the shorter length.
    n_frames = min(T, source_frames)
    states = states[:n_frames]
    valid = valid[:n_frames]
    if frames:
        frames = frames[:n_frames]

    # Filter to valid frames only.
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise ValueError(f"No valid frames in {episode_dir}")

    states = states[valid_idx]
    # Rebuild next-step actions after dropping invalid frames.  Otherwise an
    # action immediately before an invalid gap can point at a removed state.
    actions = states_to_actions(states, action_mode)
    if frames:
        frames = [frames[i] for i in valid_idx]
    T_valid = len(valid_idx)

    # --- Write MP4 (v3 layout: videos/<key>/chunk-XXX/file-NNN.mp4) ---
    chunk_str = f"chunk-{chunk:03d}"
    file_str = f"file-{episode_index:03d}"
    video_dir = Path(out_dir) / "videos" / HEAD_CAM_KEY / chunk_str
    mp4_path = video_dir / f"{file_str}.mp4"
    if visual_video is not None:
        video_to_mp4(visual_video, valid_idx, mp4_path, fps=fps)
    else:
        frames_to_mp4(frames, mp4_path, fps=fps)
    print(f"  video -> {mp4_path}  ({T_valid} frames, {TARGET_IMG_SIZE}x{TARGET_IMG_SIZE})")

    # --- Build Parquet table ---
    timestamps = (np.arange(T_valid, dtype=np.float64) / fps).tolist()
    global_start = 0  # batch converter rewrites the real value.

    rows = {
        "observation.state": [states[t].tolist() for t in range(T_valid)],
        "action": [actions[t].tolist() for t in range(T_valid)],
        "timestamp": timestamps,
        "frame_index": list(range(T_valid)),
        "episode_index": [episode_index] * T_valid,
        "index": list(range(global_start, global_start + T_valid)),
        "task_index": [0] * T_valid,
    }

    table = pa.table(rows)
    parquet_dir = Path(out_dir) / "data" / chunk_str
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{file_str}.parquet"
    pq.write_table(table, str(parquet_path))
    print(f"  parquet -> {parquet_path}  ({T_valid} rows)")

    return {
        "episode_index": episode_index,
        "tasks": [task_description],
        "length": T_valid,
        "source_episode_id": Path(episode_dir).name,
        "visual_source": (
            str(visual_video) if visual_video is not None else str(rgb_dir)
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Convert one retargeted episode to LeRobot format.")
    ap.add_argument("--episode_dir", required=True, help="Path to episode directory")
    ap.add_argument("--out_dir", required=True, help="Output dataset root")
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--action_mode", default="absolute", choices=["absolute", "delta"])
    ap.add_argument("--img_glob", default="frame_*.jpg")
    ap.add_argument("--task", default="manipulate object")
    ap.add_argument(
        "--visual_source",
        default="auto",
        help="'auto' (prefer robot composite), 'robot', 'rgb', or a video path.",
    )
    ap.add_argument(
        "--allow_legacy_actions",
        action="store_true",
        help="Allow camera-frame/finger-only legacy actions (not recommended).",
    )
    args = ap.parse_args()

    meta = convert_episode(
        args.episode_dir,
        args.out_dir,
        args.episode_index,
        fps=args.fps,
        action_mode=args.action_mode,
        img_glob=args.img_glob,
        task_description=args.task,
        visual_source=args.visual_source,
        allow_legacy_actions=args.allow_legacy_actions,
    )
    print(f"\nEpisode metadata: {meta}")


if __name__ == "__main__":
    main()
