#!/usr/bin/env python3
"""Prepare one or more numbered Choco video archives for annotation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add only new episodes to an existing prepared dataset.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.append:
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if args.append and not (output / "dataset_manifest.json").is_file():
        raise FileNotFoundError(
            f"append requires an existing prepared manifest: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    episodes = []
    try:
        for archive in args.archive:
            archive = archive.resolve()
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    member_path = Path(member.filename)
                    if member.is_dir() or member_path.suffix.lower() not in VIDEO_SUFFIXES:
                        continue
                    group = member_path.parts[0].replace(".", "")
                    episode = f"{group}__{member_path.stem}"
                    episode_dir = staging / episode
                    episode_dir.mkdir()
                    source_path = episode_dir / f"source{member_path.suffix.upper()}"
                    with bundle.open(member) as source, source_path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    (episode_dir / "rgb").symlink_to(source_path.name)
                    episodes.append({
                        "episode": episode,
                        "group": member_path.parts[0],
                        "archive": str(archive),
                        "archive_member": member.filename,
                        "video": str(source_path),
                    })
        if not episodes:
            raise ValueError("archives contain no supported videos")
        names = [item["episode"] for item in episodes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate episode names across archives")
        if args.append:
            collisions = sorted(name for name in names if (output / name).exists())
            if collisions:
                raise FileExistsError(
                    f"refusing to replace existing episodes: {collisions}"
                )
            previous = json.loads(
                (output / "dataset_manifest.json").read_text()
            )
            existing_names = {
                item["episode"] for item in previous["episodes"]
            }
            if existing_names & set(names):
                raise ValueError("manifest contains colliding episode names")
            for episode_dir in sorted(staging.iterdir()):
                if episode_dir.is_dir():
                    os.replace(episode_dir, output / episode_dir.name)
            episodes = list(previous["episodes"]) + episodes
        manifest = {
            "schema_version": 1,
            "purpose": "2026-08-13 Choco annotation",
            "label_profile": "kitchen_choco",
            "labels": [
                "HangCup",
                "StackContainers",
                "PlaceLightGreenSnackBoxInTrashBin",
                "PlaceRedSnackBoxInTrashBin",
                "WipeFloorWithSponge",
                "Transition",
            ],
            "episodes": sorted(
                [
                    {
                        **item,
                        "video": str(
                            output
                            / item["episode"]
                            / Path(item["video"]).name
                        ),
                    }
                    for item in episodes
                ],
                key=lambda item: item["episode"],
            ),
            "total_episodes": len(episodes),
        }
        if args.append:
            temporary_manifest = output / ".dataset_manifest.json.tmp"
            temporary_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
            os.replace(temporary_manifest, output / "dataset_manifest.json")
            staging.rmdir()
        else:
            (staging / "dataset_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
            os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"episodes": len(episodes), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
