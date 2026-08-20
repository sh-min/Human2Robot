#!/usr/bin/env python3
"""Bind the tracked SH object masks into every VGGT-Omega input manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1"
SPECS = {
    "cup": (44, 58),
    "snack": (120, 144),
    "lock": (267, 289),
    "sweep": (341, 462),
}


def record(path: Path, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    payload.update(extra)
    return payload


def main() -> int:
    for key, (start, reference) in SPECS.items():
        inputs = PILOT / key / "inputs"
        manifest_path = inputs / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sh_frame = int(manifest["selection"]["sh_frame_index"])
        expected = reference + 5
        if sh_frame != expected:
            raise ValueError(f"{key}: expected SH {expected}, got {sh_frame}")
        masks = np.load(
            PILOT / key / "object_pose_tracking/sh_sam2/object_mask_sam2.npy",
            mmap_mode="r",
        )
        mask = np.asarray(masks[reference - start], dtype=bool)
        output = inputs / f"sh_mask_modal_sam2_frame{sh_frame:06d}.png"
        if not cv2.imwrite(str(output), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write {output}")
        manifest["outputs"]["sh_modal_mask"] = record(
            output,
            view="SH",
            pipeline_camera="camera_1",
            frame_index=sh_frame,
            representation="binary PNG with values 0 and 255",
            provenance_class="SAM2_Hiera_Large_model_inferred_not_ground_truth",
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{key}: {output} pixels={int(mask.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
