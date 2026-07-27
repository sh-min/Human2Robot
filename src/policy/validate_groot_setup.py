"""Validate a GR00T N1.7 dataset and custom modality config without weights."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _import_config(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "skill2policy_groot_modality", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def validate(dataset: Path, modality_config: Path) -> None:
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import (
        LeRobotEpisodeLoader,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag

    _import_config(modality_config)
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    if tag not in MODALITY_CONFIGS:
        raise KeyError(f"{tag!r} was not registered by {modality_config}")
    configs = MODALITY_CONFIGS[tag]
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset),
        modality_configs=configs,
    )
    if len(loader) < 1:
        raise ValueError(f"No episodes in {dataset}")
    trajectory = loader[0]
    if len(trajectory) < 1:
        raise ValueError(f"Episode 0 has no frames in {dataset}")
    print(
        f"GR00T dataset OK: episodes={len(loader)} "
        f"episode0_frames={len(trajectory)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--modality_config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.dataset.resolve(), args.modality_config.resolve())


if __name__ == "__main__":
    main()
