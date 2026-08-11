"""Validated task configuration for multi-object kitchen datasets."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from object_config import load_object_spec


SCHEMA_VERSION = 1
_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _resolve_path(value: str, spec_path: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    return str(path.resolve())


def load_task_spec(
    path: str | Path,
    *,
    check_objects: bool = False,
) -> dict:
    """Load and validate a multi-object task YAML specification."""
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    raw = yaml.safe_load(spec_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{spec_path} must contain a YAML mapping")
    spec = deepcopy(raw)

    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{spec_path}: schema_version must be {SCHEMA_VERSION}"
        )
    task_id = spec.get("task_id")
    if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id):
        raise ValueError(
            "task_id must contain lowercase letters, digits, '_' or '-'"
        )
    instruction = spec.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    spec["instruction"] = instruction.strip()

    labels = spec.get("action_labels")
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label.strip() for label in labels)
    ):
        raise ValueError("action_labels must be a non-empty string list")
    labels = [label.strip() for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("action_labels must be unique")
    spec["action_labels"] = labels

    object_paths = spec.get("object_specs")
    if (
        not isinstance(object_paths, list)
        or not object_paths
        or not all(isinstance(item, str) and item.strip() for item in object_paths)
    ):
        raise ValueError("object_specs must be a non-empty path list")
    resolved_objects = [
        _resolve_path(item, spec_path) for item in object_paths
    ]
    if len(set(resolved_objects)) != len(resolved_objects):
        raise ValueError("object_specs must not contain duplicates")
    object_ids: list[str] = []
    for object_path in resolved_objects:
        object_spec = load_object_spec(
            object_path,
            check_assets=check_objects,
        )
        if object_spec["object_id"] in object_ids:
            raise ValueError(
                f"duplicate object_id {object_spec['object_id']!r}"
            )
        object_ids.append(object_spec["object_id"])
    spec["object_specs"] = resolved_objects
    spec["object_ids"] = object_ids

    dataset = spec.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping")
    defaults = {
        "recordings_root": f"../../data/{task_id}_dataset/recordings",
        "lerobot_v3_root": f"../../data/lerobot_{task_id}",
        "groot_v21_root": f"../../data/groot_{task_id}",
    }
    for key, default in defaults.items():
        dataset[key] = _resolve_path(str(dataset.get(key, default)), spec_path)
    episode_glob = dataset.get("episode_glob", "*")
    if (
        not isinstance(episode_glob, str)
        or not episode_glob.strip()
        or "/" in episode_glob
        or "\\" in episode_glob
    ):
        raise ValueError("dataset.episode_glob must match directory names")
    dataset["episode_glob"] = episode_glob

    spec["_spec_path"] = str(spec_path)
    return spec


def public_task_spec(spec: dict) -> dict:
    """Drop loader-only fields before storing a task spec in metadata."""
    return {
        key: value
        for key, value in spec.items()
        if not key.startswith("_")
    }


def get_value(spec: dict, dotted_key: str) -> Any:
    value: Any = spec
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("spec")
    validate.add_argument("--check-objects", action="store_true")

    get = sub.add_parser("get")
    get.add_argument("spec")
    get.add_argument("key")

    show = sub.add_parser("show")
    show.add_argument("spec")

    args = parser.parse_args()
    spec = load_task_spec(
        args.spec,
        check_objects=getattr(args, "check_objects", False),
    )
    if args.command == "validate":
        print(
            f"OK task_id={spec['task_id']} "
            f"objects={len(spec['object_ids'])} "
            f"labels={len(spec['action_labels'])}"
        )
    elif args.command == "get":
        value = get_value(spec, args.key)
        if isinstance(value, (dict, list)):
            print(json.dumps(value, ensure_ascii=False))
        else:
            print(value)
    else:
        print(
            yaml.safe_dump(
                public_task_spec(spec),
                sort_keys=False,
                allow_unicode=True,
            ),
            end="",
        )


if __name__ == "__main__":
    main()
