#!/usr/bin/env python3
"""Create and propagate SH masks for every non-transition object segment."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/cube_dataset/26.08.05_stereo_calibrated/1"
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1"
SAM_ROOT = ROOT / "third_party/sam2"
CHECKPOINT = SAM_ROOT / "checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"

# MH interval/reference and an explicit SH prompt box on reference+5.
SPECS = {
    "cup": {"label": "Cup", "start": 44, "end": 92, "reference": 58, "box": [735, 225, 890, 360]},
    "snack": {"label": "Snack", "start": 120, "end": 159, "reference": 144, "box": [500, 145, 710, 235]},
    "lock": {"label": "Lock", "start": 267, "end": 307, "reference": 289, "box": [775, 315, 950, 440]},
    "sweep": {"label": "Sweep", "start": 341, "end": 518, "reference": 462, "box": [775, 285, 910, 365]},
}


def overlay(image, mask, text):
    out = image.copy(); tint = np.zeros_like(out); tint[:, :, 1] = 255
    out[mask] = cv2.addWeighted(out, .45, tint, .55, 0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    cv2.rectangle(out, (0, 0), (1280, 44), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    sys.path.insert(0, str(SAM_ROOT))
    import torch
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    seeds = {}
    image_model = build_sam2(CONFIG, str(CHECKPOINT), device="cuda")
    predictor = SAM2ImagePredictor(image_model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for key, spec in SPECS.items():
            sh_frame = spec["reference"] + 5
            image_bgr = cv2.imread(str(DATASET / f"camera_1/rgb/rgb_frame{sh_frame:06d}.jpg"))
            predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(box=np.asarray(spec["box"], np.float32), multimask_output=True)
            x0, y0, x1, y1 = spec["box"]
            box_area = max((x1 - x0) * (y1 - y0), 1)
            adjusted = []
            for candidate, predicted_iou in zip(masks, scores, strict=True):
                area = max(int(candidate.sum()), 1)
                inside = int(candidate[y0:y1, x0:x1].sum())
                outside_fraction = 1.0 - inside / area
                size_penalty = abs(float(np.log(area / box_area)))
                adjusted.append(float(predicted_iou) - 2.0 * outside_fraction - 0.05 * size_penalty)
            best = int(np.argmax(adjusted))
            seeds[key] = masks[best].astype(bool)
            if int(seeds[key].sum()) < 200:
                raise RuntimeError(f"{key} seed mask is implausibly small")
            print(f"{key} seed SH={sh_frame} score={float(scores[best]):.3f} area={int(seeds[key].sum())}")
    del predictor, image_model
    torch.cuda.empty_cache()

    video_predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device="cuda")
    reports = {}
    for key, spec in SPECS.items():
        start_sh, end_sh, seed_sh = spec["start"] + 5, spec["end"] + 5, spec["reference"] + 5
        sh_frames = np.arange(start_sh, end_sh + 1, dtype=np.int32)
        seed_local = int(seed_sh - start_sh)
        output = PILOT / key / "object_pose_tracking/sh_sam2"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"sam2_{key}_") as tmp:
            directory = Path(tmp)
            for local, frame in enumerate(sh_frames):
                os.symlink((DATASET / f"camera_1/rgb/rgb_frame{int(frame):06d}.jpg").resolve(), directory / f"{local:06d}.jpg")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = video_predictor.init_state(str(directory), offload_video_to_cpu=True)
                video_predictor.add_new_mask(state, seed_local, 1, seeds[key])
                masks = np.zeros((len(sh_frames), 720, 1280), dtype=bool)
                for reverse in (False, True):
                    for local, ids, logits in video_predictor.propagate_in_video(state, start_frame_idx=seed_local, reverse=reverse):
                        object_index = [int(v) for v in ids].index(1)
                        masks[int(local)] = logits[object_index, 0].detach().cpu().numpy() > 0
        masks[seed_local] = seeds[key]
        areas = masks.sum(axis=(1, 2))
        if np.any(areas == 0):
            raise RuntimeError(f"{key} has empty propagated masks")
        np.save(output / "frame_indices_sh.npy", sh_frames)
        np.save(output / "object_mask_sam2.npy", masks)
        writer = cv2.VideoWriter(str(output / "sh_object_mask_sam2.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
        for frame, mask in zip(sh_frames, masks, strict=True):
            image = cv2.imread(str(DATASET / f"camera_1/rgb/rgb_frame{int(frame):06d}.jpg"))
            writer.write(overlay(image, mask, f"{spec['label']} | SH {frame} | SAM2 inferred"))
        writer.release()
        report = {"schema_version": 1, "kind": "multi_object_sh_sam2_track", "label": spec["label"], "mh_interval": [spec["start"], spec["end"]], "sh_interval": [start_sh, end_sh], "seed_sh_frame": seed_sh, "prompt_box_xyxy": spec["box"], "area_min_px": int(areas.min()), "area_max_px": int(areas.max()), "model_inferred_not_ground_truth": True}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports[key] = report
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
