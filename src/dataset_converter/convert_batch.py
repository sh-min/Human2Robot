"""Batch-convert multiple retargeted episodes into a complete LeRobot dataset.

Scans a root directory for episode folders (each containing ``rgb/`` and
``rgb_hawor/``), converts them via ``convert_episode``, and assembles all
required metadata files (info.json, episodes.jsonl, tasks.jsonl, stats.json,
modality.json).

Expected input layout:
    <data_root>/
        episode_0/
            rgb/ ...
            rgb_hawor/ ...
        episode_1/ ...
        ...

Output (LeRobot v2, GR00T-compatible):
    <out_dir>/
        data/chunk-000/episode_000000.parquet ...
        videos/chunk-000/observation.images.head_cam/episode_000000.mp4 ...
        meta/
            info.json
            episodes.jsonl
            tasks.jsonl
            stats.json
            modality.json

Usage:
    PYTHONPATH=$PWD/src python -m dataset_converter.convert_batch \\
        --data_root /path/to/episodes \\
        --out_dir /path/to/lerobot_dataset \\
        --task "manipulate rubik's cube"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dataset_converter.convert_episode import convert_episode
from dataset_converter.schema import BIMANUAL_DIM, write_modality_json


def _discover_episodes(data_root: str) -> list[str]:
    """Find sub-directories that look like retargeted episodes.

    An episode directory must contain ``rgb/`` and ``rgb_hawor/``.
    """
    episodes = []
    for name in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, name)
        if not os.path.isdir(d):
            continue
        has_rgb = os.path.isdir(os.path.join(d, "rgb"))
        has_hawor = os.path.isdir(os.path.join(d, "rgb_hawor"))
        if has_rgb and has_hawor:
            episodes.append(d)
    return episodes


def _compute_stats(out_dir: str) -> dict:
    """Compute per-column min/max/mean/std over all parquet files."""
    data_dir = Path(out_dir) / "data"
    all_states = []
    all_actions = []
    for pf in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(str(pf))
        states = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.array(table.column("action").to_pylist(), dtype=np.float32)
        all_states.append(states)
        all_actions.append(actions)

    if not all_states:
        return {}

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    def _col_stats(arr: np.ndarray, name: str) -> dict:
        return {
            name: {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
            }
        }

    stats = {}
    stats.update(_col_stats(states, "observation.state"))
    stats.update(_col_stats(actions, "action"))
    return stats


def _reindex_parquets(out_dir: str) -> int:
    """Re-assign global ``index`` across all episode parquets. Returns total frame count."""
    data_dir = Path(out_dir) / "data"
    parquets = sorted(data_dir.rglob("*.parquet"))
    global_idx = 0
    for pf in parquets:
        table = pq.read_table(str(pf))
        n = table.num_rows
        import pyarrow as pa

        new_index = pa.array(list(range(global_idx, global_idx + n)), type=pa.int64())
        col_idx = table.schema.get_field_index("index")
        table = table.set_column(col_idx, "index", new_index)
        pq.write_table(table, str(pf))
        global_idx += n
    return global_idx


def convert_batch(
    data_root: str,
    out_dir: str,
    *,
    fps: float = 30.0,
    action_mode: str = "absolute",
    img_glob: str = "frame_*.jpg",
    task_description: str = "manipulate cube",
) -> None:
    """Convert all episodes under ``data_root`` into a LeRobot v2 dataset."""
    episodes = _discover_episodes(data_root)
    if not episodes:
        raise FileNotFoundError(
            f"No episode directories found in {data_root}. "
            "Each episode needs rgb/ and rgb_hawor/ subdirectories."
        )

    print(f"Found {len(episodes)} episode(s) in {data_root}\n")

    episode_metas = []
    for idx, ep_dir in enumerate(episodes):
        print(f"--- Episode {idx}: {os.path.basename(ep_dir)} ---")
        try:
            meta = convert_episode(
                ep_dir,
                out_dir,
                idx,
                fps=fps,
                action_mode=action_mode,
                img_glob=img_glob,
                task_description=task_description,
            )
            episode_metas.append(meta)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIPPED: {e}")
        print()

    if not episode_metas:
        raise RuntimeError("All episodes failed conversion.")

    # Re-assign global indices across all episodes.
    total_frames = _reindex_parquets(out_dir)

    # --- Write metadata ---
    meta_dir = Path(out_dir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for m in episode_metas:
            f.write(json.dumps(m) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    # info.json
    info = {
        "codebase_version": "v2.1",
        "fps": fps,
        "video": True,
        "encoding": {"vcodec": "libx264", "pix_fmt": "yuv420p"},
        "total_episodes": len(episode_metas),
        "total_frames": total_frames,
        "data_path": "data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "shapes": {
            "observation.state": [BIMANUAL_DIM],
            "action": [BIMANUAL_DIM],
            "observation.images.head_cam": {"width": None, "height": None, "channels": 3},
        },
        "names": {
            "observation.state": _state_names(),
            "action": _state_names(),
        },
        "action_mode": action_mode,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    # stats.json
    stats = _compute_stats(out_dir)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    # modality.json (GR00T)
    write_modality_json(out_dir)

    print(f"Dataset complete: {out_dir}")
    print(f"  episodes: {len(episode_metas)}")
    print(f"  total frames: {total_frames}")
    print(f"  action_mode: {action_mode}")


def _state_names() -> list[str]:
    """Human-readable names for each dimension of the 38-D state/action vector."""
    from dataset_converter.schema import STATE_FIELDS

    names = [""] * BIMANUAL_DIM
    for field in STATE_FIELDS:
        dim = field.end - field.start
        for i in range(dim):
            names[field.start + i] = f"{field.name}_{i}"
    return names


def main():
    ap = argparse.ArgumentParser(description="Batch-convert episodes to LeRobot dataset.")
    ap.add_argument("--data_root", required=True, help="Root dir containing episode sub-dirs")
    ap.add_argument("--out_dir", required=True, help="Output dataset root")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--action_mode", default="absolute", choices=["absolute", "delta"])
    ap.add_argument("--img_glob", default="frame_*.jpg")
    ap.add_argument("--task", default="manipulate cube", help="Task description string")
    args = ap.parse_args()

    convert_batch(
        args.data_root,
        args.out_dir,
        fps=args.fps,
        action_mode=args.action_mode,
        img_glob=args.img_glob,
        task_description=args.task,
    )


if __name__ == "__main__":
    main()
