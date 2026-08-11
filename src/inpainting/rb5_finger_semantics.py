"""XHand finger-link contract shared by the Isaac renderer and tests.

The imported XHand USD merges each fixed fingertip into its preceding moving
body.  The twelve body prims below therefore cover every visible phalanx while
excluding the palm, flange, and RB5 arm.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

import numpy as np

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_LABEL_IDS = {
    finger: index + 1 for index, finger in enumerate(FINGER_NAMES)
}
SEMANTIC_TYPE = "rb5_finger"
FINGER_SEMANTIC_COLORS_RGBA = {
    "thumb": (255, 32, 32, 255),
    "index": (32, 255, 32, 255),
    "middle": (32, 32, 255, 255),
    "ring": (255, 224, 32, 255),
    "pinky": (255, 32, 224, 255),
}


def xhand_finger_link_names(side: str) -> dict[str, tuple[str, ...]]:
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported XHand side: {side!r}")
    prefix = f"{side}_hand_"
    return {
        "thumb": (
            prefix + "thumb_bend_link",
            prefix + "thumb_rota_link1",
            prefix + "thumb_rota_link2",
        ),
        "index": (
            prefix + "index_bend_link",
            prefix + "index_rota_link1",
            prefix + "index_rota_link2",
        ),
        "middle": (
            prefix + "mid_link1",
            prefix + "mid_link2",
        ),
        "ring": (
            prefix + "ring_link1",
            prefix + "ring_link2",
        ),
        "pinky": (
            prefix + "pinky_link1",
            prefix + "pinky_link2",
        ),
    }


def expected_finger_link_names(side: str) -> frozenset[str]:
    groups = xhand_finger_link_names(side)
    return frozenset(name for names in groups.values() for name in names)


def validate_finger_link_names(
    side: str,
    discovered_names: Iterable[str],
) -> frozenset[str]:
    expected = expected_finger_link_names(side)
    discovered = frozenset(discovered_names)
    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)
    if missing or unexpected:
        raise RuntimeError(
            "XHand finger-link contract mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(discovered) != 12:
        raise RuntimeError(
            f"XHand must expose exactly 12 moving finger links, got {len(discovered)}"
        )
    return discovered


def _semantic_labels(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        if ":" in value:
            semantic_type, label = value.split(":", 1)
            return {semantic_type: label}
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _rgba_key(value: object) -> tuple[int, int, int, int] | None:
    if isinstance(value, (tuple, list)):
        parts = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not (
            (stripped.startswith("(") and stripped.endswith(")"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ):
            return None
        parts = [item.strip() for item in stripped[1:-1].split(",")]
    else:
        return None
    if len(parts) != 4:
        return None
    try:
        rgba = tuple(int(item) for item in parts)
    except (TypeError, ValueError):
        return None
    if any(value < 0 or value > 255 for value in rgba):
        return None
    return rgba


def finger_labels_from_semantics(
    semantic_ids: np.ndarray,
    robot_mask: np.ndarray,
    semantic_info: Mapping[str, object],
) -> np.ndarray:
    """Resolve exact per-finger labels from Isaac's ``idToLabels`` metadata."""
    ids = np.asarray(semantic_ids)
    mask = np.asarray(robot_mask, dtype=bool)
    colorized = ids.shape == mask.shape + (4,)
    if ids.shape == mask.shape + (1,):
        ids = ids[..., 0]
    elif not colorized and ids.shape != mask.shape:
        raise ValueError(
            f"semantic/robot shape mismatch: {ids.shape} vs {mask.shape}"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"semantic IDs must be integer, got {ids.dtype}")
    id_to_labels = semantic_info.get("idToLabels")
    if not isinstance(id_to_labels, Mapping):
        raise RuntimeError(
            "semantic metadata is missing the idToLabels mapping"
        )

    finger_keys: list[tuple[object, int]] = []
    for raw_id, raw_labels in id_to_labels.items():
        labels = _semantic_labels(raw_labels)
        if SEMANTIC_TYPE not in labels:
            continue
        label = str(labels[SEMANTIC_TYPE])
        if label.upper() in {"BACKGROUND", "UNLABELLED"}:
            continue
        if label not in FINGER_LABEL_IDS:
            raise RuntimeError(
                f"unexpected {SEMANTIC_TYPE} label {label!r}"
            )
        if colorized:
            semantic_key = _rgba_key(raw_id)
            if semantic_key is None:
                raise RuntimeError(
                    f"invalid semantic RGBA key {raw_id!r}"
                )
        else:
            try:
                semantic_key = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid semantic ID {raw_id!r}"
                ) from exc
        finger_keys.append((semantic_key, FINGER_LABEL_IDS[label]))

    output = np.zeros(mask.shape, dtype=np.uint8)
    for semantic_key, finger_label in finger_keys:
        if colorized:
            matched = np.all(
                ids == np.asarray(semantic_key, dtype=ids.dtype),
                axis=-1,
            )
        else:
            matched = ids == semantic_key
        output[matched & mask] = finger_label
    return output


def isolate_finger_semantics(
    semantic_ids: np.ndarray,
    robot_mask: np.ndarray,
    semantic_info: Mapping[str, object],
) -> np.ndarray:
    """Return only pixels mapped explicitly to an XHand finger label."""
    return (
        finger_labels_from_semantics(
            semantic_ids,
            robot_mask,
            semantic_info,
        )
        > 0
    )
