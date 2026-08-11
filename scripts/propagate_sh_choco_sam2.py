#!/usr/bin/env python3
"""Propagate the frame-192 SH Choco mask through the matched 52-frame clip.

The output is explicitly model-inferred evidence.  It is useful for checking
the MH pose track, but is not treated as a manual annotation or metric stereo
measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/kitchen_dataset/26.08.05_stereo_calibrated/1"
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1/choco"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_stats(mask: np.ndarray) -> dict[str, object]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"area_px": 0, "centroid_xy": None, "bbox_xyxy": None}
    return {
        "area_px": int(len(xs)),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
    }


def make_overlay(image: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    tint = np.zeros_like(out)
    tint[:, :, 1] = 255
    out[mask] = cv2.addWeighted(out, 0.45, tint, 0.55, 0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=192)
    parser.add_argument("--end", type=int, default=243)
    parser.add_argument("--rgb-dir", type=Path, default=DATASET / "camera_1/rgb")
    parser.add_argument("--seed-mask", type=Path, default=PILOT / "inputs/sh_mask_modal_sam2_frame000192.png")
    parser.add_argument("--sam2-root", type=Path, default=ROOT / "third_party/sam2")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "third_party/sam2/checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--config", default="sam2_hiera_l.yaml")
    parser.add_argument("--output", type=Path, default=PILOT / "object_pose_tracking/sh_sam2")
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("end must be >= start")
    frames = list(range(args.start, args.end + 1))
    paths = [args.rgb_dir / f"rgb_frame{i:06d}.jpg" for i in frames]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing SH frames: {missing[:3]}")
    seed = cv2.imread(str(args.seed_mask), cv2.IMREAD_GRAYSCALE)
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if seed is None or first is None:
        raise ValueError("failed to read seed mask or first frame")
    seed = seed > 127
    if seed.shape != first.shape[:2] or int(seed.sum()) < 100:
        raise ValueError(f"invalid seed mask shape/area: {seed.shape}, {int(seed.sum())}")

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sh_choco_sam2_") as tmp:
        tmp_path = Path(tmp)
        for local_index, source in enumerate(paths):
            os.symlink(source.resolve(), tmp_path / f"{local_index:06d}.jpg")

        sys.path.insert(0, str(args.sam2_root.resolve()))
        import torch  # noqa: PLC0415
        from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415

        predictor = build_sam2_video_predictor(
            args.config, str(args.checkpoint.resolve()), device="cuda"
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(
                video_path=str(tmp_path), offload_video_to_cpu=True, offload_state_to_cpu=False
            )
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
            masks = np.zeros((len(frames), *seed.shape), dtype=bool)
            for local_index, object_ids, logits in predictor.propagate_in_video(state):
                ids = [int(v) for v in object_ids]
                object_index = ids.index(1)
                masks[int(local_index)] = logits[object_index, 0].detach().cpu().numpy() > 0

    if np.any(masks.sum(axis=(1, 2)) == 0):
        empty = np.flatnonzero(masks.sum(axis=(1, 2)) == 0).tolist()
        raise RuntimeError(f"SAM2 returned empty masks at local frames {empty}")
    # Preserve the exact user-reviewed seed on the conditioning frame.
    masks[0] = seed

    np.save(args.output / "frame_indices.npy", np.asarray(frames, dtype=np.int32))
    np.save(args.output / "sh_choco_mask_sam2.npy", masks)
    stats = [mask_stats(mask) for mask in masks]
    areas = np.asarray([item["area_px"] for item in stats], dtype=np.float64)
    ratios = np.maximum(areas[1:], 1) / np.maximum(areas[:-1], 1)

    video_path = args.output / "sh_choco_sam2_overlay.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {video_path}")
    for frame, image_path, mask in zip(frames, paths, masks, strict=True):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        writer.write(make_overlay(image, mask, f"SH frame {frame} | SAM2 inferred Choco mask"))
    writer.release()

    report = {
        "schema_version": 1,
        "kind": "sh_choco_sam2_mask_track",
        "status": "model_inferred_not_ground_truth",
        "frame_indices": frames,
        "mh_frame_mapping": {"rule": "mh = sh - 5", "mh_start": args.start - 5, "mh_end": args.end - 5},
        "seed": {"frame": args.start, "path": str(args.seed_mask.resolve()), "sha256": sha256(args.seed_mask)},
        "model": {"name": "SAM2 Hiera Large", "config": args.config, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint)},
        "statistics": stats,
        "quality": {
            "empty_frames": np.flatnonzero(areas == 0).tolist(),
            "area_min_px": int(areas.min()),
            "area_max_px": int(areas.max()),
            "largest_adjacent_area_ratio": float(max(ratios.max(initial=1.0), (1.0 / ratios).max(initial=1.0))),
        },
        "outputs": {"masks": str((args.output / "sh_choco_mask_sam2.npy").resolve()), "overlay_video": str(video_path.resolve())},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["quality"], indent=2))
    print(video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
