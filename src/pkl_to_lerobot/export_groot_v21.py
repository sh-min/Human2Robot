"""Export the local LeRobot v3 dataset as GR00T-flavoured LeRobot v2.1.

Official Isaac-GR00T N1.7 consumes LeRobot v2.1 plus ``meta/modality.json``.
The converter uses hard links for large parquet/video payloads when possible,
so keeping both layouts does not normally duplicate the media on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from object_config import load_object_spec, public_object_spec


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        )
    )


def _v21_features(features: dict) -> dict:
    converted = json.loads(json.dumps(features))
    for feature in converted.values():
        feature.pop("fps", None)
        if feature.get("dtype") == "video":
            info = feature["info"]
            feature["shape"] = [
                int(info["video.height"]),
                int(info["video.width"]),
                int(info["video.channels"]),
            ]
            feature["names"] = ["height", "width", "channels"]
    return converted


def export_groot_v21(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
    object_spec: str | Path | None = None,
) -> Path:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    source_info_path = source / "meta" / "info.json"
    if not source_info_path.is_file():
        raise FileNotFoundError(source_info_path)
    if (
        destination == source
        or destination in source.parents
        or source in destination.parents
    ):
        raise ValueError(
            "GR00T output and source cannot contain one another"
        )
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{destination} is not empty; pass --overwrite to rebuild "
                "this generated dataset"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    source_info = json.loads(source_info_path.read_text())
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot v3.0 source, got "
            f"{source_info.get('codebase_version')!r}"
        )
    episode_count = int(source_info["total_episodes"])
    data_files = sorted((source / "data").rglob("*.parquet"))
    video_files = sorted(
        (source / "videos" / "observation.images.head_cam").rglob("*.mp4")
    )
    if len(data_files) != episode_count or len(video_files) != episode_count:
        raise ValueError(
            f"Expected {episode_count} parquet/video files, got "
            f"{len(data_files)}/{len(video_files)}"
        )

    episode_meta_path = (
        source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    if not episode_meta_path.is_file():
        raise FileNotFoundError(episode_meta_path)
    episode_df = pq.read_table(episode_meta_path).to_pandas()
    if len(episode_df) != episode_count:
        raise ValueError(
            f"Episode metadata has {len(episode_df)} rows, "
            f"expected {episode_count}"
        )

    episode_rows: list[dict] = []
    task_names: list[str] = []
    for episode_index in range(episode_count):
        chunk = episode_index // 1000
        _link_or_copy(
            data_files[episode_index],
            destination
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet",
        )
        _link_or_copy(
            video_files[episode_index],
            destination
            / "videos"
            / f"chunk-{chunk:03d}"
            / "observation.images.head_cam"
            / f"episode_{episode_index:06d}.mp4",
        )
        row = episode_df.iloc[episode_index]
        tasks = list(row["tasks"])
        for task in tasks:
            if task not in task_names:
                task_names.append(task)
        episode_row = {
            "episode_index": episode_index,
            "tasks": tasks,
            "length": int(row["length"]),
            "source_episode_id": str(row["source_episode_id"]),
        }
        pose_json = row.get("object_pose_json")
        if isinstance(pose_json, str) and pose_json != "null":
            episode_row["object_pose"] = json.loads(pose_json)
        episode_rows.append(episode_row)

    meta = destination / "meta"
    _jsonl(meta / "episodes.jsonl", episode_rows)
    _jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": index, "task": task}
            for index, task in enumerate(task_names)
        ],
    )

    normalized_object_spec = (
        load_object_spec(object_spec, check_assets=True)
        if object_spec is not None
        else None
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": source_info.get("robot_type", "rby1_xhand"),
        "object_id": (
            normalized_object_spec["object_id"]
            if normalized_object_spec
            else source_info.get("object_id")
        ),
        "task_id": source_info.get("task_id"),
        "object_ids": source_info.get("object_ids", []),
        "total_episodes": episode_count,
        "total_frames": int(source_info["total_frames"]),
        "total_tasks": len(task_names),
        "chunks_size": 1000,
        "fps": int(source_info["fps"]),
        "splits": {"train": f"0:{episode_count}"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": _v21_features(source_info["features"]),
        "total_chunks": (episode_count - 1) // 1000,
        "total_videos": episode_count,
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    modality_path = source / "meta" / "modality.json"
    if not modality_path.is_file():
        raise FileNotFoundError(modality_path)
    modality = json.loads(modality_path.read_text())
    modality["annotation"] = {
        "human.task_description": {
            "original_key": "task_index",
        }
    }
    (meta / "modality.json").write_text(
        json.dumps(modality, indent=2) + "\n"
    )

    for filename in ("stats.json", "object_spec.json", "task_spec.json"):
        source_path = source / "meta" / filename
        if source_path.is_file():
            shutil.copy2(source_path, meta / filename)
    if normalized_object_spec is not None:
        (meta / "object_spec.json").write_text(
            json.dumps(
                public_object_spec(normalized_object_spec),
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"GR00T LeRobot v2.1 dataset: {destination}")
    print(
        f"  episodes={episode_count} frames={info['total_frames']} "
        f"tasks={len(task_names)}"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--object_spec")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_groot_v21(
        args.source,
        args.out,
        overwrite=args.overwrite,
        object_spec=args.object_spec,
    )


if __name__ == "__main__":
    main()
