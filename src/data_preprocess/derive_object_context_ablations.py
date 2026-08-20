"""Derive deterministic controls from Grounding-DINO + SAM2 sidecars.

The script never edits the source sidecar.  It writes alternate sidecars next
to it so every classifier experiment can reuse exactly the same V-JEPA tokens,
recording split, detections and confidence values.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch


VARIANT_KEYS = {
    "bbox": "vlm_bbox_object_context",
    "zero": "vlm_zero_object_context",
    "channel_shuffle": "vlm_channel_shuffle_object_context",
    "temporal_shuffle": "vlm_temporal_shuffle_object_context",
}


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def masks_to_patch_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Replace each SAM occupancy mask by its enclosing patch-grid box."""

    if masks.ndim != 3:
        raise ValueError("masks must have shape [T,K,S]")
    side = int(round(math.sqrt(masks.shape[-1])))
    if side * side != masks.shape[-1]:
        raise ValueError("bbox control requires a square spatial-token grid")
    grid = masks.float().reshape(*masks.shape[:2], side, side)
    boxed = torch.zeros_like(grid)
    for token_idx, object_idx in torch.nonzero(
        grid.amax(dim=(-2, -1)) > 0, as_tuple=False
    ).tolist():
        occupied = torch.nonzero(
            grid[token_idx, object_idx] > 0, as_tuple=False
        )
        row_min, col_min = occupied.amin(dim=0).tolist()
        row_max, col_max = occupied.amax(dim=0).tolist()
        boxed[token_idx, object_idx, row_min : row_max + 1, col_min : col_max + 1] = 1
    return boxed.reshape_as(masks).to(masks.dtype)


def derive_sidecar(source: dict, variant: str, recording_id: str) -> dict:
    masks = torch.as_tensor(source["masks"]).clone()
    confidence = torch.as_tensor(source["confidence"]).clone()
    if variant == "bbox":
        masks = masks_to_patch_boxes(masks)
    elif variant == "zero":
        masks.zero_()
        confidence.zero_()
    elif variant == "channel_shuffle":
        for token_idx in range(len(masks)):
            generator = torch.Generator().manual_seed(
                stable_seed(f"{recording_id}:channel:{token_idx}")
            )
            permutation = torch.randperm(masks.shape[1], generator=generator)
            masks[token_idx] = masks[token_idx, permutation].clone()
            confidence[token_idx] = confidence[token_idx, permutation].clone()
    elif variant == "temporal_shuffle":
        generator = torch.Generator().manual_seed(
            stable_seed(f"{recording_id}:temporal")
        )
        permutation = torch.randperm(len(masks), generator=generator)
        masks = masks[permutation]
        confidence = confidence[permutation]
    else:
        raise ValueError(f"unknown ablation variant: {variant}")

    derived = dict(source)
    derived["kind"] = f"object_context_ablation_{variant}"
    derived["masks"] = masks
    derived["confidence"] = confidence
    derived["derived_from"] = "vlm_sam_object_context.pt"
    derived["ablation"] = {
        "variant": variant,
        "deterministic": True,
        "recording_id": recording_id,
    }
    return derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--source-key", default="vlm_sam_object_context",
        help="Source sidecar stem.",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANT_KEYS),
        default=sorted(VARIANT_KEYS),
    )
    args = parser.parse_args()

    source_paths = sorted(args.data_root.glob(f"*/{args.source_key}.pt"))
    if not source_paths:
        raise FileNotFoundError(
            f"no {args.source_key}.pt sidecars below {args.data_root}"
        )
    written = 0
    for source_path in source_paths:
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if int(source.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported sidecar schema: {source_path}")
        for variant in args.variants:
            output_path = source_path.with_name(f"{VARIANT_KEYS[variant]}.pt")
            torch.save(
                derive_sidecar(source, variant, source_path.parent.name),
                output_path,
            )
            written += 1
    print(f"Derived {written} sidecars from {len(source_paths)} recordings")


if __name__ == "__main__":
    main()
