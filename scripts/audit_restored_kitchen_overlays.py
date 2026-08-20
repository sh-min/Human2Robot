#!/usr/bin/env python3
"""Audit restored kitchen overlays and publish policy/V-JEPA staging roots."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
    }


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def validate_annotation(path: Path, episode: str, frame_count: int) -> tuple[dict, Counter]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("episode") != episode:
        raise ValueError(f"{path}: episode={value.get('episode')!r}, expected {episode!r}")
    if int(value.get("num_frames", -1)) != frame_count:
        raise ValueError(f"{path}: num_frames does not match {frame_count}")
    expected_start = 0
    labels: Counter = Counter()
    for index, segment in enumerate(value.get("segments", [])):
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        label = str(segment["label"])
        if start != expected_start or end < start or end >= frame_count:
            raise ValueError(f"{path}: invalid/non-contiguous segment {index}: {segment}")
        expected_start = end + 1
        labels[label] += end - start + 1
    if expected_start != frame_count:
        raise ValueError(f"{path}: annotations end at {expected_start - 1}, expected {frame_count - 1}")
    return value, labels


def validate_hand_pkl(path: Path, frame_count: int) -> dict:
    value = load_pickle(path)
    data = np.asarray(value["data"])
    wrist_pos = np.asarray(value["wrist_pos"])
    wrist_quat = np.asarray(value["wrist_quat"])
    valid = np.asarray(value["valid"], dtype=bool)
    expected = {
        "data": (frame_count, 12),
        "wrist_pos": (frame_count, 3),
        "wrist_quat": (frame_count, 4),
        "valid": (frame_count,),
    }
    actual = {
        "data": data.shape,
        "wrist_pos": wrist_pos.shape,
        "wrist_quat": wrist_quat.shape,
        "valid": valid.shape,
    }
    if actual != expected:
        raise ValueError(f"{path}: shapes={actual}, expected={expected}")
    if valid.any() and not (
        np.isfinite(data[valid]).all()
        and np.isfinite(wrist_pos[valid]).all()
        and np.isfinite(wrist_quat[valid]).all()
    ):
        raise ValueError(f"{path}: non-finite values in valid frames")
    return {"valid_frames": int(valid.sum()), "total_frames": frame_count}


def ensure_policy_stage(stage: Path, episodes: list[dict]) -> None:
    expected = {item["flat_id"]: Path(item["work_episode"]) for item in episodes}
    if stage.exists():
        existing = {item.name for item in stage.iterdir() if item.is_dir()}
        if existing != set(expected):
            raise ValueError(f"{stage}: existing episode set does not match audited data")
        return
    stage.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{stage.name}.", dir=stage.parent))
    try:
        for name, target in expected.items():
            (temporary / name).symlink_to(os.path.relpath(target, temporary))
        os.replace(temporary, stage)
    except Exception:
        temporary.rmdir()
        raise


def ensure_vjepa_stage(stage: Path, episodes: list[dict]) -> None:
    expected = {item["flat_id"] for item in episodes}
    if stage.exists():
        existing = {item.name for item in stage.iterdir() if item.is_dir()}
        if existing != expected:
            raise ValueError(f"{stage}: existing episode set does not match audited data")
        return
    stage.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{stage.name}.", dir=stage.parent))
    try:
        for item in episodes:
            source = Path(item["work_episode"])
            episode = temporary / item["flat_id"]
            episode.mkdir()
            for name, target in (
                ("rgb", source / "robot_overlay.mp4"),
                ("rgb_hawor", source / "rgb_hawor"),
                ("gt_labels.json", source / "gt_labels.json"),
            ):
                (episode / name).symlink_to(os.path.relpath(target, episode))
        os.replace(temporary, stage)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="TAG=DATASET_ROOT")
    parser.add_argument("--policy_stage", type=Path, required=True)
    parser.add_argument("--vjepa_stage", type=Path, required=True)
    parser.add_argument("--expected_total", type=int, default=45)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    source_specs = []
    for raw in args.source:
        tag, separator, value = raw.partition("=")
        if not separator or not tag:
            raise ValueError(f"invalid --source {raw!r}; expected TAG=PATH")
        source_specs.append((tag, Path(value).resolve()))

    episodes: list[dict] = []
    missing: list[str] = []
    label_frames: Counter = Counter()
    for tag, source_root in source_specs:
        for source_episode in sorted(path for path in source_root.iterdir() if path.is_dir()):
            episode_id = source_episode.name
            work_episode = work_root / tag / episode_id
            overlay = work_episode / "robot_overlay.mp4"
            if not overlay.is_file():
                missing.append(f"{tag}/{episode_id}")
                continue
            annotation_path = source_episode / "gt_labels.json"
            annotation_raw = json.loads(annotation_path.read_text(encoding="utf-8"))
            expected_frames = int(annotation_raw["num_frames"])
            annotation, labels = validate_annotation(
                annotation_path, episode_id, expected_frames
            )
            video = probe_video(overlay)
            if video["frames"] != expected_frames:
                raise ValueError(f"{overlay}: frames={video['frames']}, expected={expected_frames}")
            if abs(video["fps"] - float(annotation["fps"])) > 1e-3:
                raise ValueError(f"{overlay}: fps={video['fps']}, expected={annotation['fps']}")

            hawor = work_episode / "rgb_hawor"
            npz_path = hawor / "retarget_input.npz"
            with np.load(npz_path) as npz:
                if np.asarray(npz["valid"]).shape != (2, expected_frames):
                    raise ValueError(f"{npz_path}: invalid valid-mask shape")
                npz_valid = np.asarray(npz["valid"], dtype=bool).sum(axis=1)
            hands = {
                side: validate_hand_pkl(
                    hawor / f"qpos_xhand_{side}_smooth.pkl", expected_frames
                )
                for side in ("right", "left")
            }
            if not any(value["valid_frames"] for value in hands.values()):
                raise ValueError(f"{work_episode}: neither hand has valid frames")
            label_frames.update(labels)
            episodes.append(
                {
                    "flat_id": f"{tag}__{episode_id}",
                    "source_tag": tag,
                    "episode": episode_id,
                    "frames": expected_frames,
                    "fps": float(annotation["fps"]),
                    "resolution_wh": [video["width"], video["height"]],
                    "labels": dict(labels),
                    "hands": hands,
                    "hawor_valid_left_right": [int(npz_valid[0]), int(npz_valid[1])],
                    "work_episode": str(work_episode),
                    "robot_overlay": str(overlay.resolve()),
                    "gt_labels": str(annotation_path.resolve()),
                }
            )

    source_total = sum(1 for _, root in source_specs for path in root.iterdir() if path.is_dir())
    if source_total != args.expected_total:
        raise ValueError(f"source episode count={source_total}, expected={args.expected_total}")
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing {len(missing)} completed overlays: {missing}")
    if len(episodes) != args.expected_total and not args.allow_incomplete:
        raise RuntimeError(f"audited={len(episodes)}, expected={args.expected_total}")
    flat_ids = [item["flat_id"] for item in episodes]
    if len(flat_ids) != len(set(flat_ids)):
        raise ValueError("flattened episode IDs are not unique")

    policy_stage = args.policy_stage.resolve()
    vjepa_stage = args.vjepa_stage.resolve()
    ensure_policy_stage(policy_stage, episodes)
    ensure_vjepa_stage(vjepa_stage, episodes)
    manifest = {
        "schema_version": 1,
        "kind": "restored_kitchen_robot_overlay_policy_manifest",
        "expected_episodes": args.expected_total,
        "audited_episodes": len(episodes),
        "missing": missing,
        "label_frames": dict(sorted(label_frames.items())),
        "policy_stage": str(policy_stage),
        "vjepa_stage": str(vjepa_stage),
        "episodes": episodes,
    }
    manifest_path = work_root / "policy_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"audited={len(episodes)}/{args.expected_total} missing={len(missing)}")
    print(f"labels={dict(sorted(label_frames.items()))}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
