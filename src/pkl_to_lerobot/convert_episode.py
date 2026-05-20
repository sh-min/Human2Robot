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
            "-vf", f"scale={TARGET_IMG_SIZE}:{TARGET_IMG_SIZE}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(int(fps)),
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        if list_file.exists():
            list_file.unlink()


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
        states, actions, valid = final_pose_to_state_action(
            final_pose, action_mode=action_mode
        )
    else:
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

    # --- Load and align frames ---
    frames = _sorted_frames(rgb_dir, img_glob)
    if len(frames) == 0:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir} (glob={img_glob})")

    # pkl T may differ from frame count if HaWoR trimmed the sequence.
    # Align to the shorter length.
    n_frames = min(T, len(frames))
    states = states[:n_frames]
    actions = actions[:n_frames]
    valid = valid[:n_frames]
    frames = frames[:n_frames]

    # Filter to valid frames only.
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise ValueError(f"No valid frames in {episode_dir}")

    states = states[valid_idx]
    actions = actions[valid_idx]
    frames = [frames[i] for i in valid_idx]
    T_valid = len(valid_idx)

    # --- Write MP4 (v3 layout: videos/<key>/chunk-XXX/file-NNN.mp4) ---
    chunk_str = f"chunk-{chunk:03d}"
    file_str = f"file-{episode_index:03d}"
    video_dir = Path(out_dir) / "videos" / HEAD_CAM_KEY / chunk_str
    mp4_path = video_dir / f"{file_str}.mp4"
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
    }


def main():
    ap = argparse.ArgumentParser(description="Convert one retargeted episode to LeRobot format.")
    ap.add_argument("--episode_dir", required=True, help="Path to episode directory")
    ap.add_argument("--out_dir", required=True, help="Output dataset root")
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--action_mode", default="absolute", choices=["absolute", "delta"])
    ap.add_argument("--img_glob", default="frame_*.jpg")
    ap.add_argument("--task", default="manipulate cube")
    args = ap.parse_args()

    meta = convert_episode(
        args.episode_dir,
        args.out_dir,
        args.episode_index,
        fps=args.fps,
        action_mode=args.action_mode,
        img_glob=args.img_glob,
        task_description=args.task,
    )
    print(f"\nEpisode metadata: {meta}")


if __name__ == "__main__":
    main()
