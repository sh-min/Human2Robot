"""Extract label-leak-free Grounding-DINO + SAM2 object context.

Every video is grounded with the same canonical object prompt bank.  Detected
boxes prompt SAM2 image segmentation, and the masks are aligned to the 24x24
V-JEPA 2.1 patch grid.  The result is stored beside (not inside) features.pt so
the frozen V-JEPA baseline remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skill_classifier.action_semantics import (  # noqa: E402
    load_action_semantics,
    object_prompt_bank,
)


DEFAULT_SEMANTICS = (
    PROJECT_ROOT / "src/skill_classifier/config/kitchen_action_semantics.yaml"
)
DEFAULT_SAM2_ROOT = PROJECT_ROOT / "third_party/sam2"
DEFAULT_SAM2_CHECKPOINT = (
    DEFAULT_SAM2_ROOT / "checkpoints/sam2_hiera_large.pt"
)
DEFAULT_SAM2_CONFIG = "sam2_hiera_l.yaml"
DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_CONTEXT_KEY = "vlm_sam_object_context"
PATCH_SIZE = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def match_object_name(
    detected_phrase: str,
    prompt_bank: list[dict[str, Any]],
) -> str | None:
    """Map Grounding-DINO's decoded phrase to one canonical object name."""

    detected = _normalize_phrase(detected_phrase)
    if not detected:
        return None
    matches: list[tuple[int, str]] = []
    for item in prompt_bank:
        for prompt in item["prompts"]:
            normalized = _normalize_phrase(prompt)
            if detected == normalized:
                matches.append((1000 + len(normalized), item["name"]))
            elif normalized in detected or detected in normalized:
                matches.append((len(normalized), item["name"]))
    if not matches:
        return None
    return max(matches)[1]


def grounding_caption(prompt_bank: list[dict[str, Any]]) -> str:
    """Build the fixed caption used for every frame and every split."""

    primary_prompts = [item["prompts"][0].strip(" .") for item in prompt_bank]
    return " . ".join(primary_prompts) + " ."


def _ground_each_object(
    processor,
    grounder,
    rgb: np.ndarray,
    prompt_bank: list[dict[str, Any]],
    *,
    device: str,
    box_threshold: float,
    text_threshold: float,
    batch_size: int,
    autocast,
) -> list[dict[str, Any]]:
    """Ground canonical objects independently to avoid cross-phrase merging."""

    if batch_size <= 0:
        raise ValueError("grounding prompt batch size must be positive")
    image = Image.fromarray(rgb)
    queries = [
        (item, prompt.strip(" .") + " .")
        for item in prompt_bank
        for prompt in item["grounding_queries"]
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for start in range(0, len(queries), batch_size):
        rows = queries[start : start + batch_size]
        captions = [caption for _, caption in rows]
        inputs = processor(
            images=[image] * len(rows),
            text=captions,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode(), autocast:
            outputs = grounder(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[rgb.shape[:2]] * len(rows),
        )
        for (item, caption), result in zip(rows, results, strict=True):
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            phrases = result["text_labels"]
            grouped[item["name"]].extend(
                {
                    "name": item["name"],
                    "phrase": str(phrase),
                    "prompt": caption,
                    "box": box,
                    "score": float(score),
                }
                for box, score, phrase in zip(
                    boxes, scores, phrases, strict=True
                )
            )
    selected: list[dict[str, Any]] = []
    for item in prompt_bank:
        kept: list[dict[str, Any]] = []
        candidates = sorted(
            grouped[item["name"]], key=lambda value: value["score"], reverse=True
        )
        for candidate in candidates:
            box = np.asarray(candidate["box"], dtype=np.float32)
            overlaps = []
            for existing in kept:
                other = np.asarray(existing["box"], dtype=np.float32)
                x0, y0 = np.maximum(box[:2], other[:2])
                x1, y1 = np.minimum(box[2:], other[2:])
                intersection = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
                area = max(0.0, float(box[2] - box[0])) * max(
                    0.0, float(box[3] - box[1])
                )
                other_area = max(0.0, float(other[2] - other[0])) * max(
                    0.0, float(other[3] - other[1])
                )
                union = area + other_area - intersection
                overlaps.append(intersection / union if union > 0 else 0.0)
            if not overlaps or max(overlaps) < 0.75:
                kept.append(candidate)
            if len(kept) >= int(item["max_instances"]):
                break
        selected.extend(kept)
    return selected


def mask_to_patch_occupancy(
    mask: np.ndarray,
    *,
    crop_size: int,
    spatial_profile: str,
) -> np.ndarray:
    """Apply V-JEPA geometry and average mask occupancy per image patch."""

    value = np.asarray(mask, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"mask must be HxW, got {value.shape}")
    if crop_size <= 0 or crop_size % PATCH_SIZE:
        raise ValueError("crop_size must be a positive multiple of patch size")
    height, width = value.shape
    if spatial_profile == "legacy_stretch":
        cropped = cv2.resize(
            value, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR
        )
    elif spatial_profile == "vjepa2_eval_center_crop":
        short_side = int(256.0 / 224.0 * crop_size)
        scale = short_side / min(height, width)
        resized_width = max(crop_size, int(round(width * scale)))
        resized_height = max(crop_size, int(round(height * scale)))
        resized = cv2.resize(
            value,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = (resized_width - crop_size) // 2
        top = (resized_height - crop_size) // 2
        cropped = resized[top : top + crop_size, left : left + crop_size]
    else:
        raise ValueError(f"unknown spatial profile: {spatial_profile}")
    if cropped.shape != (crop_size, crop_size):
        raise RuntimeError(f"mask crop failed: {cropped.shape}")
    grid = crop_size // PATCH_SIZE
    occupancy = cropped.reshape(
        grid, PATCH_SIZE, grid, PATCH_SIZE
    ).mean(axis=(1, 3))
    return np.clip(occupancy, 0.0, 1.0).reshape(-1).astype(np.float32)


def _decode_frames(source: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(frame_indices)
    frames: dict[int, np.ndarray] = {}
    if source.is_dir():
        paths = sorted(
            path
            for path in source.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for frame_index in wanted:
            if frame_index < 0 or frame_index >= len(paths):
                raise ValueError(f"frame {frame_index} outside {source}")
            bgr = cv2.imread(str(paths[frame_index]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot decode {paths[frame_index]}")
            frames[frame_index] = bgr
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {source}")
        frame_index = 0
        while wanted:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                frames[frame_index] = bgr
                wanted.remove(frame_index)
            frame_index += 1
        capture.release()
    missing = sorted(set(frame_indices) - set(frames))
    if missing:
        raise RuntimeError(f"failed to decode source frames {missing[:8]}")
    return frames


def _select_detections(
    result: dict[str, Any],
    prompt_bank: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    boxes = result["boxes"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    phrases = result["text_labels"]
    for box, score, phrase in zip(boxes, scores, phrases, strict=True):
        name = match_object_name(str(phrase), prompt_bank)
        if name is None:
            continue
        grouped[name].append(
            {"name": name, "phrase": str(phrase), "box": box, "score": float(score)}
        )
    selected = []
    limits = {item["name"]: int(item["max_instances"]) for item in prompt_bank}
    for item in prompt_bank:
        candidates = sorted(
            grouped[item["name"]], key=lambda value: value["score"], reverse=True
        )
        selected.extend(candidates[: limits[item["name"]]])
    return selected


def suppress_cross_object_duplicate_boxes(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Keep one semantic name when two queries return the same object box."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must lie in [0,1]")
    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        detections, key=lambda value: value["score"], reverse=True
    ):
        box = np.asarray(candidate["box"], dtype=np.float32)
        duplicate = False
        for existing in kept:
            if existing["name"] == candidate["name"]:
                continue
            other = np.asarray(existing["box"], dtype=np.float32)
            top_left = np.maximum(box[:2], other[:2])
            bottom_right = np.minimum(box[2:], other[2:])
            intersection_size = np.maximum(bottom_right - top_left, 0.0)
            intersection = float(intersection_size.prod())
            area = float(np.maximum(box[2:] - box[:2], 0.0).prod())
            other_area = float(np.maximum(other[2:] - other[:2], 0.0).prod())
            union = area + other_area - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _segment_detections(
    predictor,
    rgb: np.ndarray,
    detections: list[dict[str, Any]],
    prompt_bank: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    height, width = rgb.shape[:2]
    masks = {
        item["name"]: np.zeros((height, width), dtype=bool)
        for item in prompt_bank
    }
    confidence = {item["name"]: 0.0 for item in prompt_bank}
    if not detections:
        return masks, confidence
    boxes = np.stack([item["box"] for item in detections]).astype(np.float32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    boxes = boxes[valid]
    detections = [item for item, keep in zip(detections, valid, strict=True) if keep]
    if not detections:
        return masks, confidence
    predictor.set_image(rgb)
    predicted_masks, sam_scores, _ = predictor.predict(
        box=boxes, multimask_output=False
    )
    predicted_masks = np.asarray(predicted_masks)
    sam_scores = np.asarray(sam_scores).reshape(-1)
    if predicted_masks.ndim == 4:
        predicted_masks = predicted_masks[:, 0]
    if predicted_masks.ndim == 2:
        predicted_masks = predicted_masks[None]
    if len(predicted_masks) != len(detections):
        raise RuntimeError("SAM2 result count does not match grounded boxes")
    for item, mask, sam_score in zip(
        detections, predicted_masks, sam_scores, strict=True
    ):
        name = item["name"]
        masks[name] |= np.asarray(mask) > 0
        confidence[name] = max(
            confidence[name],
            float(item["score"]) * max(0.0, float(sam_score)),
        )
    return masks, confidence


def _preview_frame(
    bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    confidence: dict[str, float],
    prompt_bank: list[dict[str, Any]],
    frame_index: int,
) -> np.ndarray:
    colors = [
        (52, 152, 219),
        (46, 204, 113),
        (155, 89, 182),
        (241, 196, 15),
        (230, 126, 34),
        (231, 76, 60),
        (26, 188, 156),
        (149, 165, 166),
    ]
    panel_width = max(320, bgr.shape[1] // 3)
    panel_height = max(180, bgr.shape[0] // 3)
    base = cv2.resize(
        bgr, (panel_width, panel_height), interpolation=cv2.INTER_AREA
    )
    panels = [base.copy()]
    cv2.rectangle(panels[0], (0, 0), (panel_width, 35), (10, 10, 10), -1)
    cv2.putText(
        panels[0],
        f"RAW | source frame {frame_index}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for object_index, item in enumerate(prompt_bank):
        name = item["name"]
        mask = masks[name]
        color = colors[object_index % len(colors)]
        panel = (base.astype(np.float32) * 0.32).astype(np.uint8)
        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (panel_width, panel_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        if resized_mask.any():
            colored = np.empty_like(panel)
            colored[:] = color
            panel[resized_mask] = cv2.addWeighted(
                base[resized_mask], 0.45, colored[resized_mask], 0.55, 0
            )
            contours, _ = cv2.findContours(
                resized_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(panel, contours, -1, color, 2)
        cv2.rectangle(panel, (0, 0), (panel_width, 35), (10, 10, 10), -1)
        cv2.putText(
            panel,
            f"{name} | confidence {confidence[name]:.2f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    while len(panels) < 9:
        empty = np.zeros_like(base)
        cv2.putText(
            empty,
            "unused prompt slot",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        panels.append(empty)
    return np.concatenate(
        [np.concatenate(panels[row * 3 : (row + 1) * 3], axis=1) for row in range(3)],
        axis=0,
    )


def _load_models(args):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounder = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model
    ).to(args.device)
    grounder.eval()

    sys.path.insert(0, str(args.sam2_root.resolve()))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_model = build_sam2(
        args.sam2_config, str(args.sam2_checkpoint), device=args.device
    )
    sam_predictor = SAM2ImagePredictor(sam_model)
    return processor, grounder, sam_predictor


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def _feature_groups(root: Path, recording_glob: str) -> list[list[Path]]:
    patterns = [value.strip() for value in recording_glob.split(",") if value.strip()]
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(f"{pattern}/features.pt")
        }
    )
    groups: dict[tuple[Any, ...], list[Path]] = defaultdict(list)
    for path in paths:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        source = Path(bundle["input_provenance"]["rgb"]["path"]).resolve()
        centers = tuple(map(int, bundle["token_center_frame_indices"].tolist()))
        parameters = bundle["input_provenance"]["parameters"]
        key = (
            str(source),
            centers,
            int(parameters["crop_size"]),
            str(bundle["spatial_profile"]),
            int(bundle["vjepa_orig_dense"].shape[1]),
        )
        groups[key].append(path)
    return list(groups.values())


def extract_group(
    feature_paths: list[Path],
    *,
    args,
    semantics: dict[str, Any],
    prompt_bank: list[dict[str, Any]],
    processor,
    grounder,
    sam_predictor,
) -> None:
    reference = torch.load(
        feature_paths[0], map_location="cpu", weights_only=False
    )
    parameters = reference["input_provenance"]["parameters"]
    source = Path(reference["input_provenance"]["rgb"]["path"]).resolve()
    centers = list(map(int, reference["token_center_frame_indices"].tolist()))
    crop_size = int(parameters["crop_size"])
    spatial_profile = str(reference["spatial_profile"])
    spatial_tokens = int(reference["vjepa_orig_dense"].shape[1])
    object_names = [item["name"] for item in prompt_bank]
    caption = grounding_caption(prompt_bank)
    frames = _decode_frames(source, centers)

    all_masks = np.zeros(
        (len(centers), len(prompt_bank), spatial_tokens), dtype=np.float16
    )
    all_confidence = np.zeros((len(centers), len(prompt_bank)), dtype=np.float32)
    token_detections: list[dict[str, Any]] = []
    preview_path = feature_paths[0].with_name(
        f"{args.context_key}_preview.mp4"
    )
    writer = None
    if args.preview:
        first = frames[centers[0]]
        preview_width = 3 * max(320, first.shape[1] // 3)
        preview_height = 3 * max(180, first.shape[0] // 3)
        writer = cv2.VideoWriter(
            str(preview_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(reference.get("token_rate_hz", 2.0)),
            (preview_width, preview_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open preview writer: {preview_path}")

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda")
        else nullcontext()
    )
    for token_index, frame_index in enumerate(centers):
        bgr = frames[frame_index]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = _ground_each_object(
            processor,
            grounder,
            rgb,
            prompt_bank,
            device=args.device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            batch_size=args.grounding_prompt_batch_size,
            autocast=autocast,
        )
        detections = suppress_cross_object_duplicate_boxes(detections)
        token_detections.append(
            {
                "token_index": token_index,
                "source_frame": frame_index,
                "detections": [
                    {
                        "object_name": item["name"],
                        "decoded_phrase": item["phrase"],
                        "object_prompt": item["prompt"],
                        "grounding_score": float(item["score"]),
                        "box_xyxy": [float(value) for value in item["box"]],
                    }
                    for item in detections
                ],
            }
        )
        with torch.inference_mode(), autocast:
            masks, confidence = _segment_detections(
                sam_predictor, rgb, detections, prompt_bank
            )
        for object_index, object_name in enumerate(object_names):
            occupancy = mask_to_patch_occupancy(
                masks[object_name],
                crop_size=crop_size,
                spatial_profile=spatial_profile,
            )
            if len(occupancy) != spatial_tokens:
                raise RuntimeError(
                    f"mask/V-JEPA token mismatch: {len(occupancy)} vs {spatial_tokens}"
                )
            all_masks[token_index, object_index] = occupancy.astype(np.float16)
            all_confidence[token_index, object_index] = confidence[object_name]
        if writer is not None:
            writer.write(
                _preview_frame(bgr, masks, confidence, prompt_bank, frame_index)
            )
        print(
            f"  token {token_index + 1:03d}/{len(centers):03d} frame={frame_index} "
            + " ".join(
                f"{name}={confidence[name]:.2f}" for name in object_names
            ),
            flush=True,
        )
    if writer is not None:
        writer.release()

    semantics_path = Path(args.semantics).resolve()
    payload = {
        "schema_version": 1,
        "kind": "grounding_dino_sam2_vjepa_patch_context",
        "object_names": object_names,
        "masks": torch.from_numpy(all_masks),
        "confidence": torch.from_numpy(all_confidence),
        "num_tokens": len(centers),
        "spatial_tokens": spatial_tokens,
        "token_center_frame_indices": torch.tensor(centers, dtype=torch.int64),
        "spatial_profile": spatial_profile,
        "crop_size": crop_size,
        "patch_size": PATCH_SIZE,
        "grounding_caption": caption,
        "grounding_queries": {
            item["name"]: list(item["grounding_queries"])
            for item in prompt_bank
        },
        "prompt_bank": prompt_bank,
        "grounding_model": args.grounding_model,
        "sam2": {
            "config": args.sam2_config,
            "checkpoint": _file_signature(args.sam2_checkpoint),
        },
        "thresholds": {
            "box": float(args.box_threshold),
            "text": float(args.text_threshold),
            "cross_object_box_iou_suppression": 0.85,
        },
        "token_detections": token_detections,
        "semantics": {
            "path": str(semantics_path),
            "sha256": _sha256(semantics_path),
            "action_labels": list(semantics["action_labels"]),
        },
        "conditioning_contract": {
            "same_prompt_bank_for_every_clip": True,
            "objects_grounded_as_independent_queries": True,
            "ground_truth_label_used_for_prompt_selection": False,
            "annotation_files_rewritten": False,
        },
        "source_rgb": _file_signature(source),
    }
    for feature_path in feature_paths:
        bundle = torch.load(feature_path, map_location="cpu", weights_only=False)
        if tuple(bundle["token_center_frame_indices"].tolist()) != tuple(centers):
            raise RuntimeError("grouped feature alignment changed during extraction")
        destination = feature_path.with_name(f"{args.context_key}.pt")
        _atomic_torch_save(payload, destination)
        print(f"[saved] {destination}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recording-glob", default="*")
    parser.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--context-key", default=DEFAULT_CONTEXT_KEY)
    parser.add_argument("--grounding-model", default=DEFAULT_GROUNDING_MODEL)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-prompt-batch-size", type=int, default=4)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.box_threshold <= 1 or not 0 <= args.text_threshold <= 1:
        raise ValueError("grounding thresholds must lie in [0,1]")
    if not args.sam2_root.is_dir() or not args.sam2_checkpoint.is_file():
        raise FileNotFoundError("local SAM2 implementation/checkpoint is missing")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    semantics = load_action_semantics(args.semantics)
    prompt_bank = object_prompt_bank(semantics)
    groups = _feature_groups(args.data_root, args.recording_glob)
    if not groups:
        raise ValueError("no feature bundles matched")
    pending = []
    for group in groups:
        destinations = [path.with_name(f"{args.context_key}.pt") for path in group]
        if not args.overwrite and all(path.is_file() for path in destinations):
            print(f"[skip] context exists for {len(group)} aligned bundle(s)")
        else:
            pending.append(group)
    if not pending:
        return 0
    print(
        f"Grounding caption (identical for every frame): {grounding_caption(prompt_bank)}"
    )
    print(f"Unique source/alignment groups: {len(pending)}", flush=True)
    processor, grounder, sam_predictor = _load_models(args)
    for index, group in enumerate(pending, 1):
        print(
            f"[{index}/{len(pending)}] {group[0].parent.name} "
            f"({len(group)} feature variants)",
            flush=True,
        )
        extract_group(
            group,
            args=args,
            semantics=semantics,
            prompt_bank=prompt_bank,
            processor=processor,
            grounder=grounder,
            sam_predictor=sam_predictor,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
