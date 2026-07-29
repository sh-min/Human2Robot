"""Validated object/task configuration shared by data, sim, and policy code.

An object spec is intentionally dataset-local in meaning but repository-local
in structure: paths inside YAML are resolved relative to the YAML file, while
dataset/output paths may be overridden by the shell entry points.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
_OBJECT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_PRIMITIVES = {"box", "sphere", "cylinder", "capsule"}
_SUCCESS_TYPES = {"none", "lift"}


def _require(
    mapping: dict,
    key: str,
    expected_type: type | tuple[type, ...],
    where: str,
):
    if key not in mapping:
        raise ValueError(f"{where}.{key} is required")
    value = mapping[key]
    if not isinstance(value, expected_type):
        expected_name = (
            "/".join(item.__name__ for item in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise ValueError(
            f"{where}.{key} must be {expected_name}, "
            f"got {type(value).__name__}"
        )
    return value


def _vector(value: Any, length: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must be a list of length {length}")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must contain numbers") from exc
    return result


def _range(value: Any, where: str) -> list[float]:
    result = _vector(value, 2, where)
    if result[0] > result[1]:
        raise ValueError(f"{where} lower bound exceeds upper bound")
    return result


def _resolve_path(value: str, spec_path: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    return str(path.resolve())


def load_object_spec(
    path: str | Path,
    *,
    check_assets: bool = False,
) -> dict:
    """Load, validate, and normalize an object YAML specification."""
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
    object_id = _require(spec, "object_id", str, "root")
    if not _OBJECT_ID.fullmatch(object_id):
        raise ValueError(
            "root.object_id must contain lowercase letters, digits, '_' or '-'"
        )

    task = _require(spec, "task", dict, "root")
    instruction = _require(task, "instruction", str, "task").strip()
    if not instruction:
        raise ValueError("task.instruction cannot be empty")

    geometry = _require(spec, "geometry", dict, "root")
    primitive = geometry.get("primitive")
    visual_mesh = geometry.get("visual_mesh")
    collision_mesh = geometry.get("collision_mesh")
    mjcf = geometry.get("mjcf")
    if primitive is None and visual_mesh is None and mjcf is None:
        raise ValueError(
            "geometry requires primitive, visual_mesh, or mjcf"
        )
    if mjcf is not None and any(
        value is not None
        for value in (primitive, visual_mesh, collision_mesh)
    ):
        raise ValueError(
            "geometry.mjcf cannot be combined with primitive or mesh fields"
        )
    if primitive is not None:
        if primitive not in _PRIMITIVES:
            raise ValueError(
                f"geometry.primitive must be one of {sorted(_PRIMITIVES)}"
            )
        dimensions = geometry.get("dimensions_m")
        expected = {
            "box": 3,
            "sphere": 1,
            "cylinder": 2,
            "capsule": 2,
        }[primitive]
        dimensions = _vector(
            dimensions, expected, "geometry.dimensions_m"
        )
        if min(dimensions) <= 0:
            raise ValueError("geometry.dimensions_m values must be positive")
        geometry["dimensions_m"] = dimensions

    for key in ("visual_mesh", "collision_mesh", "mjcf"):
        if geometry.get(key):
            geometry[key] = _resolve_path(str(geometry[key]), spec_path)
            if check_assets and not Path(geometry[key]).is_file():
                raise FileNotFoundError(geometry[key])
    scale = geometry.get("scale", [1.0, 1.0, 1.0])
    geometry["scale"] = _vector(scale, 3, "geometry.scale")
    if min(geometry["scale"]) <= 0:
        raise ValueError("geometry.scale values must be positive")
    geometry["rgba"] = _vector(
        geometry.get("rgba", [0.7, 0.7, 0.7, 1.0]),
        4,
        "geometry.rgba",
    )

    physics = _require(spec, "physics", dict, "root")
    physics["mass_kg"] = float(
        _require(physics, "mass_kg", (int, float), "physics")
    )
    if physics["mass_kg"] <= 0:
        raise ValueError("physics.mass_kg must be positive")
    physics["friction"] = _vector(
        physics.get("friction", [0.8, 0.005, 0.0001]),
        3,
        "physics.friction",
    )

    spawn = _require(spec, "spawn", dict, "root")
    spawn["position"] = _vector(
        _require(spawn, "position", list, "spawn"),
        3,
        "spawn.position",
    )
    spawn["quaternion_xyzw"] = _vector(
        spawn.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        4,
        "spawn.quaternion_xyzw",
    )
    if sum(v * v for v in spawn["quaternion_xyzw"]) < 1e-12:
        raise ValueError("spawn.quaternion_xyzw cannot be zero")
    randomization = spawn.setdefault("randomization", {})
    if not isinstance(randomization, dict):
        raise ValueError("spawn.randomization must be a mapping")
    for axis in ("x_range", "y_range", "z_range", "yaw_range_deg"):
        if axis in randomization:
            randomization[axis] = _range(
                randomization[axis], f"spawn.randomization.{axis}"
            )

    success = spec.setdefault("success", {"type": "none"})
    if not isinstance(success, dict):
        raise ValueError("success must be a mapping")
    success_type = success.get("type", "none")
    if success_type not in _SUCCESS_TYPES:
        raise ValueError(
            f"success.type must be one of {sorted(_SUCCESS_TYPES)}"
        )
    success["type"] = success_type
    if success_type == "lift":
        height = float(success.get("height_delta_m", 0.1))
        if height <= 0:
            raise ValueError("success.height_delta_m must be positive")
        success["height_delta_m"] = height
    success["terminate_on_success"] = bool(
        success.get("terminate_on_success", True)
    )

    dataset = spec.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping")
    defaults = {
        "recordings_root": f"data/object_datasets/{object_id}/recordings",
        "lerobot_v3_root": f"data/lerobot_{object_id}",
        "groot_v21_root": f"data/groot_{object_id}",
    }
    for key, default in defaults.items():
        dataset[key] = _resolve_path(
            str(dataset.get(key, default)), spec_path
        )
    episode_glob = dataset.get("episode_glob", "*")
    if not isinstance(episode_glob, str) or not episode_glob.strip():
        raise ValueError("dataset.episode_glob must be a non-empty string")
    if "/" in episode_glob or "\\" in episode_glob:
        raise ValueError(
            "dataset.episode_glob must match directory names, not paths"
        )
    dataset["episode_glob"] = episode_glob

    control = spec.setdefault("control", {})
    if not isinstance(control, dict):
        raise ValueError("control must be a mapping")
    active_hands = control.get("active_hands", ["left"])
    if (
        not isinstance(active_hands, list)
        or not active_hands
        or set(active_hands) - {"left", "right"}
    ):
        raise ValueError(
            "control.active_hands must be a non-empty list of left/right"
        )
    control["active_hands"] = active_hands

    spec["_spec_path"] = str(spec_path)
    return spec


def public_object_spec(spec: dict) -> dict:
    """Drop loader-only keys before copying a spec into dataset metadata."""
    return {
        key: value
        for key, value in spec.items()
        if not key.startswith("_")
    }


def get_value(spec: dict, dotted_key: str):
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
    validate.add_argument("--check-assets", action="store_true")

    get = sub.add_parser("get")
    get.add_argument("spec")
    get.add_argument("key")

    show = sub.add_parser("show")
    show.add_argument("spec")

    args = parser.parse_args()
    spec = load_object_spec(
        args.spec,
        check_assets=getattr(args, "check_assets", False),
    )
    if args.command == "validate":
        print(
            f"OK object_id={spec['object_id']} "
            f"instruction={spec['task']['instruction']!r}"
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
                public_object_spec(spec),
                sort_keys=False,
                allow_unicode=True,
            ),
            end="",
        )


if __name__ == "__main__":
    main()
