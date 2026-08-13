"""Load and validate semantic metadata for kitchen skill labels.

The semantic file augments stable class IDs; it never rewrites annotations.
This separation prevents old checkpoints and evaluation files from silently
changing meaning while allowing Grounding DINO/SAM2 prompts to evolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROLE_KEYS = ("source_objects", "target_objects", "tool_objects")


def load_action_semantics(path: str | Path) -> dict[str, Any]:
    """Return a validated action/object semantic configuration."""

    semantic_path = Path(path)
    with semantic_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("semantic config must be a mapping")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("semantic config must use schema_version 1")

    labels = config.get("action_labels")
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        raise ValueError("action_labels must be a non-empty unique list")
    objects = config.get("objects")
    actions = config.get("actions")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("objects must be a non-empty mapping")
    if not isinstance(actions, dict) or set(actions) != set(labels):
        raise ValueError("actions must exactly match action_labels")

    for object_name, value in objects.items():
        if not isinstance(object_name, str) or not object_name:
            raise ValueError("object names must be non-empty strings")
        prompts = value.get("prompts") if isinstance(value, dict) else None
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"{object_name}: prompts must be a non-empty list")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError(f"{object_name}: every prompt must be non-empty text")
        grounding_queries = value.get("grounding_queries", prompts[:1])
        if not isinstance(grounding_queries, list) or not grounding_queries:
            raise ValueError(
                f"{object_name}: grounding_queries must be a non-empty list"
            )
        if any(
            not isinstance(prompt, str) or not prompt.strip()
            for prompt in grounding_queries
        ):
            raise ValueError(
                f"{object_name}: every grounding query must be non-empty text"
            )
        max_instances = value.get("max_instances", 1)
        if not isinstance(max_instances, int) or max_instances <= 0:
            raise ValueError(f"{object_name}: max_instances must be positive")

    known_objects = set(objects)
    for label in labels:
        action = actions[label]
        if not isinstance(action, dict):
            raise ValueError(f"{label}: action metadata must be a mapping")
        if not str(action.get("ko", "")).strip() or not str(action.get("en", "")).strip():
            raise ValueError(f"{label}: both ko and en descriptions are required")
        referenced = set()
        for role in ROLE_KEYS:
            names = action.get(role)
            if not isinstance(names, list):
                raise ValueError(f"{label}: {role} must be a list")
            referenced.update(names)
        unknown = referenced - known_objects
        if unknown:
            raise ValueError(f"{label}: unknown objects {sorted(unknown)}")

    contract = config.get("conditioning_contract", {})
    if contract.get("forbid_ground_truth_selected_prompt") is not True:
        raise ValueError("semantic config must forbid ground-truth-selected prompts")
    if contract.get("use_shared_object_bank_for_every_clip") is not True:
        raise ValueError("every clip must use the shared object bank")
    return config


def object_prompt_bank(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten canonical objects into a stable open-vocabulary prompt bank."""

    return [
        {
            "name": name,
            "ko": value["ko"],
            "prompts": list(value["prompts"]),
            "grounding_queries": list(
                value.get("grounding_queries", value["prompts"][:1])
            ),
            "max_instances": int(value.get("max_instances", 1)),
        }
        for name, value in config["objects"].items()
        if value.get("grounding_enabled", True)
    ]


def display_label(config: dict[str, Any], label: str, language: str = "ko") -> str:
    """Format a stable ID and its human-readable action description."""

    if label not in config["actions"]:
        raise KeyError(label)
    if language not in ("ko", "en"):
        raise ValueError("language must be 'ko' or 'en'")
    return f"{label} — {config['actions'][label][language]}"
