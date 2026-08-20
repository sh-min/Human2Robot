"""Replace heuristic HaCo object boxes with Grounding-DINO detections.

HaCo remains responsible for finding contact intervals and localising the
grasp.  Grounding-DINO runs the same open-vocabulary prompt bank on every seed
frame, and the detection nearest the HaCo grasp is selected as SAM2's box
prompt.  No action label, recording ID, or per-video coordinate is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skill_classifier.action_semantics import (  # noqa: E402
    load_action_semantics,
    object_prompt_bank,
)


DEFAULT_SEMANTICS = (
    REPO_ROOT / "src/skill_classifier/config/kitchen_action_semantics.yaml"
)
DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"


def _read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not decode frame {index} from {video}")
    return frame


def _distance_to_box(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to an xyxy box (zero inside)."""
    dx = np.maximum(np.maximum(box[0] - points[:, 0], 0), points[:, 0] - box[2])
    dy = np.maximum(np.maximum(box[1] - points[:, 1], 0), points[:, 1] - box[3])
    return np.hypot(dx, dy)


def _intersection_fraction(box: np.ndarray, region: np.ndarray) -> float:
    corner0 = np.maximum(box[:2], region[:2])
    corner1 = np.minimum(box[2:], region[2:])
    size = np.maximum(corner1 - corner0, 0)
    intersection = float(size.prod())
    area = float(np.maximum(box[2:] - box[:2], 0).prod())
    return intersection / area if area else 0.0


def contact_grounding_score(detection: dict, segment: dict) -> float:
    """Rank semantic detections by confidence and HaCo grasp proximity."""
    box = np.asarray(detection["box"], dtype=np.float32)
    region = np.asarray(segment["box"], dtype=np.float32)
    points = np.asarray(segment.get("positive_points", []), dtype=np.float32).reshape(-1, 2)
    if not len(points):
        points = ((region[:2] + region[2:]) / 2)[None]
    distance = float(_distance_to_box(points, box).min())
    region_diag = max(1.0, float(np.linalg.norm(region[2:] - region[:2])))
    proximity = float(np.exp(-4.0 * distance / region_diag))
    coverage = _intersection_fraction(box, region)
    # DINO confidence distinguishes equally local objects; contact geometry
    # dominates so a confident rack/bin behind a held item cannot win merely
    # because it is larger.
    return 2.0 * proximity + coverage + float(detection["score"])


def _detect(processor, model, rgb: np.ndarray, bank: list[dict], args) -> list[dict]:
    image = Image.fromarray(rgb)
    queries = [
        (item["name"], query.strip(" .") + " .")
        for item in bank
        for query in item["grounding_queries"]
    ]
    detections = []
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if args.device.startswith("cuda") else nullcontext())
    for start in range(0, len(queries), args.batch_size):
        batch = queries[start:start + args.batch_size]
        inputs = processor(
            images=[image] * len(batch),
            text=[caption for _, caption in batch],
            padding=True,
            return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode(), autocast:
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[rgb.shape[:2]] * len(batch),
        )
        for (name, caption), result in zip(batch, results, strict=True):
            for box, score, phrase in zip(
                    result["boxes"].detach().cpu().numpy(),
                    result["scores"].detach().cpu().numpy(),
                    result["text_labels"], strict=True):
                detections.append({
                    "name": name,
                    "prompt": caption,
                    "phrase": str(phrase),
                    "box": np.asarray(box, dtype=np.float32),
                    "score": float(score),
                })
    return detections


def _positive_points(box: np.ndarray, old_points: list) -> list[list[int]]:
    points = np.asarray(old_points, dtype=np.float32).reshape(-1, 2)
    inset = np.array([max(3.0, 0.08 * (box[2] - box[0])),
                      max(3.0, 0.08 * (box[3] - box[1]))])
    low, high = box[:2] + inset, box[2:] - inset
    inside = points[(points >= low).all(axis=1) & (points <= high).all(axis=1)]
    centre = ((box[:2] + box[2:]) / 2)[None]
    selected = np.concatenate([inside[:1], centre], axis=0)[:2]
    return np.rint(selected).astype(int).tolist()


def _draw_seed(frame: np.ndarray, segment: dict, detections: list[dict], chosen: dict | None
               ) -> np.ndarray:
    canvas = frame.copy()
    old = np.asarray(segment["box"], dtype=int)
    cv2.rectangle(canvas, tuple(old[:2]), tuple(old[2:]), (0, 180, 255), 3)
    for detection in detections:
        box = np.rint(detection["box"]).astype(int)
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (160, 160, 160), 1)
    label = "DINO: no local detection (HaCo fallback)"
    if chosen is not None:
        box = np.rint(chosen["box"]).astype(int)
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (20, 240, 20), 4)
        label = (f"DINO: {chosen['name']} {chosen['score']:.2f} "
                 f"contact-score {chosen['contact_score']:.2f}")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(canvas, f"{segment['name']} seed={segment['seed_frame']} | {label}",
                (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--grounding_model", default=DEFAULT_MODEL)
    parser.add_argument("--box_threshold", type=float, default=0.20)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--min_contact_score", type=float, default=2.15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    payload = json.loads(args.segments_json.read_text(encoding="utf-8"))
    segments = payload["segments"] if isinstance(payload, dict) else payload
    semantics = load_action_semantics(args.semantics)
    bank = object_prompt_bank(semantics)
    print("[info] fixed prompt bank:", ", ".join(item["name"] for item in bank))
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model).to(args.device).eval()

    preview_frames = []
    grounded = 0
    for segment in segments:
        frame = _read_frame(args.video, int(segment["seed_frame"]))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = _detect(processor, model, rgb, bank, args)
        for detection in detections:
            detection["contact_score"] = contact_grounding_score(detection, segment)
        local = [d for d in detections if d["contact_score"] >= args.min_contact_score]
        chosen = max(local, key=lambda d: d["contact_score"], default=None)
        segment["haco_box_before_dino"] = list(segment["box"])
        segment["grounding_dino"] = {
            "selected": chosen is not None,
            "fixed_prompt_bank": [item["name"] for item in bank],
            "candidate_count": len(detections),
        }
        if chosen is not None:
            grounded += 1
            box = np.rint(chosen["box"]).astype(int)
            height, width = frame.shape[:2]
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            segment["box"] = box.tolist()
            segment["positive_points"] = _positive_points(
                chosen["box"], segment.get("positive_points", []))
            segment["grounding_dino"].update({
                "object_name": chosen["name"],
                "decoded_phrase": chosen["phrase"],
                "prompt": chosen["prompt"],
                "confidence": chosen["score"],
                "contact_score": chosen["contact_score"],
            })
        preview_frames.append(_draw_seed(frame, segment, detections, chosen))
        status = (f"{chosen['name']} conf={chosen['score']:.3f} "
                  f"contact={chosen['contact_score']:.3f}"
                  if chosen is not None else "HaCo fallback")
        print(f"[seed] {segment['name']}: {status}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "source": "Grounding-DINO + HaCo generalized-v1",
        "notes": ("Same prompt bank and thresholds for every recording; HaCo contact "
                  "geometry selects among semantic detections; no action label or "
                  "per-video coordinates."),
        "grounding_model": args.grounding_model,
        "thresholds": {
            "box": args.box_threshold,
            "text": args.text_threshold,
            "minimum_contact_score": args.min_contact_score,
        },
        "segments": segments,
    }
    args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    thumb_width = 640
    thumbs = [cv2.resize(frame, (thumb_width, int(frame.shape[0] * thumb_width / frame.shape[1])))
              for frame in preview_frames]
    while len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[0]))
    montage = np.concatenate([
        np.concatenate(thumbs[i:i + 2], axis=1)
        for i in range(0, len(thumbs), 2)
    ], axis=0)
    cv2.imwrite(str(args.preview), montage)
    print(f"[ok] grounded {grounded}/{len(segments)} segments -> {args.output}")
    print(f"[ok] wrote {args.preview}")


if __name__ == "__main__":
    main()
