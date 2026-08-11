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
    PYTHONPATH=$PWD/src python -m pkl_to_lerobot.convert_batch \\
        --data_root /path/to/episodes \\
        --out_dir /path/to/lerobot_dataset \\
        --task_spec configs/tasks/kitchen.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from object_config import load_object_spec, public_object_spec
from task_config import load_task_spec, public_task_spec

from .convert_episode import HEAD_CAM_KEY, TARGET_IMG_SIZE, convert_episode
from .schema import BIMANUAL_DIM, write_modality_json


def _discover_episodes(
    data_root: str,
    episode_glob: str = "*",
) -> list[str]:
    """Find sub-directories that look like retargeted episodes.

    An episode directory must contain ``rgb/`` and ``rgb_hawor/``.
    """
    episodes = []
    for path in sorted(Path(data_root).glob(episode_glob)):
        if not path.is_dir():
            continue
        has_rgb = (path / "rgb").is_dir()
        has_hawor = (path / "rgb_hawor").is_dir()
        if has_rgb and has_hawor:
            episodes.append(str(path))
    return episodes


def _object_pose_metadata(episode_dir: str) -> dict | None:
    """Load optional episode-level object pose metadata."""
    path = Path(episode_dir) / "object_pose.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _episode_stats(df: pd.DataFrame) -> dict[str, dict[str, list]]:
    """Per-episode min/max/mean/std/count for vector and scalar columns."""
    stats: dict[str, dict[str, list]] = {}
    for col in ("observation.state", "action"):
        arr = np.array(df[col].to_list(), dtype=np.float32)
        stats[col] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    for col in ("timestamp", "frame_index", "index", "episode_index", "task_index"):
        arr = np.asarray(df[col].to_numpy(), dtype=np.float64).reshape(-1)
        stats[col] = {
            "min": [float(arr.min())],
            "max": [float(arr.max())],
            "mean": [float(arr.mean())],
            "std": [float(arr.std())],
            "count": [int(arr.shape[0])],
        }
    return stats


def _aggregate_stats(per_ep: list[dict]) -> dict:
    """Frame-weighted aggregation across episodes for observation.state and action."""
    agg = {}
    for col in ("observation.state", "action"):
        counts = np.array([ep[col]["count"][0] for ep in per_ep], dtype=np.float64)
        means = np.array([ep[col]["mean"] for ep in per_ep], dtype=np.float64)
        mins = np.array([ep[col]["min"] for ep in per_ep], dtype=np.float64)
        maxs = np.array([ep[col]["max"] for ep in per_ep], dtype=np.float64)
        # frame-weighted mean
        w = counts / counts.sum()
        mean = (means * w[:, None]).sum(axis=0)
        # combined variance across episodes:
        # var = sum_e w_e * (var_e + (mean_e - mean)^2)
        vars_ = np.array([np.square(ep[col]["std"]) for ep in per_ep], dtype=np.float64)
        var = (w[:, None] * (vars_ + np.square(means - mean))).sum(axis=0)
        std = np.sqrt(np.clip(var, 0.0, None))
        agg[col] = {
            "min": mins.min(axis=0).tolist(),
            "max": maxs.max(axis=0).tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return agg


def _head_cam_placeholder_stats() -> dict:
    """Channel-wise placeholder stats for the head_cam video feature.

    Real values are overridden by IMAGENET_STATS in lerobot when
    ``dataset.use_imagenet_stats=True`` (the default).
    """
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.5]], [[0.5]], [[0.5]]],
        "count": [1],
    }


def _flatten_episode_stats(stats: dict) -> dict:
    """{'feature': {'min': [...]}} -> {'stats/feature/min': [...]}."""
    return {
        f"stats/{feature}/{stat_name}": val
        for feature, s in stats.items()
        for stat_name, val in s.items()
    }


def _build_features(fps: int) -> dict[str, dict]:
    """LeRobot v3 features dict."""
    names = _state_names()
    return {
        HEAD_CAM_KEY: {
            "dtype": "video",
            "shape": [3, TARGET_IMG_SIZE, TARGET_IMG_SIZE],
            "names": ["channels", "height", "width"],
            "info": {
                "video.fps": float(fps),
                "video.height": TARGET_IMG_SIZE,
                "video.width": TARGET_IMG_SIZE,
                "video.channels": 3,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [BIMANUAL_DIM],
            "names": names,
            "fps": fps,
        },
        "action": {
            "dtype": "float32",
            "shape": [BIMANUAL_DIM],
            "names": names,
            "fps": fps,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
    }


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
    task_description: str | None = None,
    visual_source: str = "auto",
    allow_legacy_actions: bool = False,
    skip_failed: bool = False,
    object_spec_path: str | None = None,
    task_spec_path: str | None = None,
    episode_glob: str | None = None,
) -> None:
    """Convert all episodes under ``data_root`` into a LeRobot v3 dataset."""
    object_spec = (
        load_object_spec(object_spec_path, check_assets=False)
        if object_spec_path is not None
        else None
    )
    task_spec = (
        load_task_spec(task_spec_path, check_objects=False)
        if task_spec_path is not None
        else None
    )
    if task_description is None:
        if task_spec is not None:
            task_description = task_spec["instruction"]
        elif object_spec is not None:
            task_description = object_spec["task"]["instruction"]
        else:
            task_description = "manipulate object"
    if episode_glob is None:
        if task_spec is not None:
            episode_glob = task_spec["dataset"]["episode_glob"]
        elif object_spec is not None:
            episode_glob = object_spec["dataset"]["episode_glob"]
        else:
            episode_glob = "*"
    episodes = _discover_episodes(data_root, episode_glob)
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
                visual_source=visual_source,
                allow_legacy_actions=allow_legacy_actions,
            )
            episode_metas.append(meta)
        except (FileNotFoundError, ValueError) as e:
            if not skip_failed:
                raise
            print(f"  SKIPPED: {e}")
        print()

    if not episode_metas:
        raise RuntimeError("All episodes failed conversion.")

    # Re-assign global indices across all episodes.
    total_frames = _reindex_parquets(out_dir)

    # --- Build per-episode stats + metadata rows (v3) ---
    meta_dir = Path(out_dir) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(out_dir) / "data" / "chunk-000"

    per_episode_stats: list[dict] = []
    episode_rows: list[dict] = []
    cumulative = 0
    for ep_idx, ep_meta in enumerate(episode_metas):
        pf = data_dir / f"file-{ep_idx:03d}.parquet"
        df = pq.read_table(str(pf)).to_pandas()
        n = len(df)
        ep_stats = _episode_stats(df)
        per_episode_stats.append(ep_stats)
        row = {
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": ep_idx,
            "dataset_from_index": cumulative,
            "dataset_to_index": cumulative + n,
            "length": n,
            "tasks": ep_meta["tasks"],
            "source_episode_id": ep_meta["source_episode_id"],
            "visual_source": ep_meta["visual_source"],
            "object_pose_json": json.dumps(
                _object_pose_metadata(
                    str(Path(data_root) / ep_meta["source_episode_id"])
                ),
                separators=(",", ":"),
            ),
            f"videos/{HEAD_CAM_KEY}/chunk_index": 0,
            f"videos/{HEAD_CAM_KEY}/file_index": ep_idx,
            f"videos/{HEAD_CAM_KEY}/from_timestamp": 0.0,
            f"videos/{HEAD_CAM_KEY}/to_timestamp": n / float(fps),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        row.update(_flatten_episode_stats(ep_stats))
        episode_rows.append(row)
        cumulative += n

    # --- meta/episodes/chunk-000/file-000.parquet ---
    ep_parquet = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    ep_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(ep_parquet, index=False)

    # --- meta/tasks.parquet ---
    tasks_df = pd.DataFrame({"task_index": [0]},
                            index=pd.Index([task_description], name="task"))
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # --- meta/stats.json (aggregated + head_cam placeholder) ---
    agg_stats = _aggregate_stats(per_episode_stats)
    agg_stats[HEAD_CAM_KEY] = _head_cam_placeholder_stats()
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(agg_stats, f, indent=2)
        f.write("\n")

    # --- meta/info.json (v3.0) ---
    info = {
        "codebase_version": "v3.0",
        "robot_type": "rby1_xhand",
        "object_id": (
            object_spec["object_id"] if object_spec is not None else None
        ),
        "task_id": task_spec["task_id"] if task_spec is not None else None,
        "object_ids": (
            task_spec["object_ids"]
            if task_spec is not None
            else ([object_spec["object_id"]] if object_spec is not None else [])
        ),
        "total_episodes": len(episode_metas),
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": int(fps),
        "splits": {"train": f"0:{len(episode_metas)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": _build_features(int(fps)),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    # modality.json (GR00T) — still useful for GR00T finetune side.
    write_modality_json(out_dir)
    if object_spec is not None:
        with open(meta_dir / "object_spec.json", "w") as f:
            json.dump(public_object_spec(object_spec), f, indent=2)
            f.write("\n")
    if task_spec is not None:
        with open(meta_dir / "task_spec.json", "w") as f:
            json.dump(
                public_task_spec(task_spec),
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

    print(f"Dataset complete: {out_dir}")
    print(f"  episodes: {len(episode_metas)}")
    print(f"  total frames: {total_frames}")
    print(f"  action_mode: {action_mode}")
    if object_spec is not None:
        print(f"  object_id: {object_spec['object_id']}")
    if task_spec is not None:
        print(f"  task_id: {task_spec['task_id']}")
        print(f"  object_ids: {', '.join(task_spec['object_ids'])}")


def _state_names() -> list[str]:
    """Human-readable names for each dimension of the 38-D state/action vector."""
    from .schema import STATE_FIELDS

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
    ap.add_argument(
        "--task",
        default=None,
        help="Task description. Defaults to object spec instruction.",
    )
    ap.add_argument(
        "--object_spec",
        default=None,
        help="Optional single-object YAML copied into dataset metadata.",
    )
    ap.add_argument(
        "--task_spec",
        default=None,
        help="Optional multi-object task YAML copied into dataset metadata.",
    )
    ap.add_argument(
        "--episode_glob",
        default=None,
        help="Episode directory glob; defaults to object spec or '*'.",
    )
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
    ap.add_argument(
        "--skip_failed",
        action="store_true",
        help="Keep converting after a broken episode (default: fail fast).",
    )
    args = ap.parse_args()

    convert_batch(
        args.data_root,
        args.out_dir,
        fps=args.fps,
        action_mode=args.action_mode,
        img_glob=args.img_glob,
        task_description=args.task,
        visual_source=args.visual_source,
        allow_legacy_actions=args.allow_legacy_actions,
        skip_failed=args.skip_failed,
        object_spec_path=args.object_spec,
        task_spec_path=args.task_spec,
        episode_glob=args.episode_glob,
    )


if __name__ == "__main__":
    main()
