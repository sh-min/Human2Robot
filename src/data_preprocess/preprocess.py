"""Build aligned V-JEPA/MANO/label feature bundles per recording.

The default ``legacy_dense`` profile preserves the historical dense-frame
behavior.  ``vjepa2_4fps`` matches the published V-JEPA 2 ViT-L temporal and
spatial evaluation contract:

* sample the source at 4 FPS;
* pair two sampled frames per tubelet/token;
* preserve aspect ratio, resize the short side to 292 for a 256 crop, then
  center crop;
* align MANO and frame labels through the exact original-frame indices.

Outputs are written atomically to ``features.pt``.  Existing bundles are
reused only when their schema, profile, shapes, and input provenance match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocess.feature_extractor import (  # noqa: E402
    SUPPORTED_BACKBONES,
    VJEPA21_BACKBONE,
    VJEPA21FeatureExtractor,
    VJEPAFeatureExtractor,
    load_pretrained_encoder,
    load_pretrained_vjepa21_encoder,
)
from utils.labels import ACTION_LABELS  # noqa: E402
from utils.utils import rotmat_to_axis_angle  # noqa: E402


TUBELET = 2
FEATURE_SCHEMA_VERSION = 2
LEGACY_DENSE_PROFILE = "legacy_dense"
VJEPA2_4FPS_PROFILE = "vjepa2_4fps"
SAMPLING_PROFILES = (LEGACY_DENSE_PROFILE, VJEPA2_4FPS_PROFILE)
LEGACY_STRETCH = "legacy_stretch"
VJEPA2_EVAL_CROP = "vjepa2_eval_center_crop"
SPATIAL_PROFILES = (LEGACY_STRETCH, VJEPA2_EVAL_CROP)

VJEPA_MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255.0
VJEPA_STD = torch.tensor([0.229, 0.224, 0.225]) * 255.0
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class TokenAlignment:
    """Exact mapping from source frames to sampled V-JEPA tubelets."""

    source_num_frames: int
    source_fps: float
    sample_fps: float
    sampling_profile: str
    sampled_frame_indices: np.ndarray
    model_frame_indices: np.ndarray
    token_frame_indices: np.ndarray
    token_center_frame_indices: np.ndarray
    frame_to_token: np.ndarray

    @property
    def num_tokens(self) -> int:
        return int(len(self.token_frame_indices))


def build_token_alignment(
    num_frames: int,
    source_fps: float,
    *,
    sampling_profile: str = LEGACY_DENSE_PROFILE,
    sample_fps: float = 4.0,
    tubelet_size: int = TUBELET,
) -> TokenAlignment:
    """Create a deterministic source-frame/token mapping.

    ``vjepa2_4fps`` repeats a final unpaired sampled frame so the tail remains
    represented.  ``legacy_dense`` retains the old floor behavior while still
    exposing explicit alignment metadata.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    if tubelet_size != 2:
        raise ValueError("this feature bundle currently requires tubelet_size=2")
    if sampling_profile not in SAMPLING_PROFILES:
        raise ValueError(f"unknown sampling profile: {sampling_profile}")

    if sampling_profile == LEGACY_DENSE_PROFILE:
        effective_sample_fps = float(source_fps)
        sampled = np.arange(num_frames, dtype=np.int64)
        token_count = len(sampled) // tubelet_size
        if token_count == 0:
            raise ValueError("legacy_dense needs at least two source frames")
        token_frames = sampled[: token_count * tubelet_size].reshape(
            token_count, tubelet_size
        )
        # Historical inference passed every frame through the final padded
        # clip even when the last odd frame did not produce a saved token.
        model_frames = sampled.copy()
    else:
        if not np.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        if sample_fps > source_fps + 1.0e-9:
            raise ValueError("sample_fps cannot exceed source_fps")
        effective_sample_fps = float(sample_fps)
        sample_count = (
            int(np.floor((num_frames - 1) * sample_fps / source_fps + 1.0e-9))
            + 1
        )
        sampled = np.rint(
            np.arange(sample_count, dtype=np.float64)
            * source_fps
            / sample_fps
        ).astype(np.int64)
        sampled = np.unique(np.clip(sampled, 0, num_frames - 1))
        if not len(sampled):
            raise RuntimeError("temporal sampling produced no frames")
        if len(sampled) % tubelet_size:
            model_frames = np.concatenate([sampled, sampled[-1:]])
        else:
            model_frames = sampled.copy()
        token_frames = model_frames.reshape(-1, tubelet_size)

    centers = np.floor(
        token_frames.astype(np.float64).mean(axis=1) + 0.5
    ).astype(np.int64)
    source_indices = np.arange(num_frames, dtype=np.int64)
    frame_to_token = np.abs(
        source_indices[:, None] - centers[None, :]
    ).argmin(axis=1).astype(np.int64)
    return TokenAlignment(
        source_num_frames=int(num_frames),
        source_fps=float(source_fps),
        sample_fps=effective_sample_fps,
        sampling_profile=sampling_profile,
        sampled_frame_indices=sampled,
        model_frame_indices=model_frames,
        token_frame_indices=token_frames,
        token_center_frame_indices=centers,
        frame_to_token=frame_to_token,
    )


def _discover_images(directory: Path) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no image frames in {directory}")
    return paths


def source_frame_count(source: Path) -> int:
    if source.is_dir():
        return len(_discover_images(source))
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    capture.release()
    if count <= 0:
        raise ValueError(f"empty video: {source}")
    return count


def aligned_source_frame_count(rec_dir: Path, source: Path) -> int:
    """Use an annotated video prefix while rejecting a truncated source."""
    available = source_frame_count(source)
    gt_path = rec_dir / "gt_labels.json"
    if not source.is_file() or not gt_path.is_file():
        return available
    annotated = int(json.loads(gt_path.read_text())["num_frames"])
    if available < annotated:
        raise ValueError(
            f"{source}: video has {available} frames but GT needs {annotated}"
        )
    return annotated


def preprocess_rgb_frame(
    rgb: np.ndarray,
    crop_size: int,
    spatial_profile: str,
) -> np.ndarray:
    """Apply either the historical warp or Meta's evaluation center crop."""

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape (H,W,3), got {image.shape}")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if spatial_profile == LEGACY_STRETCH:
        return cv2.resize(
            image,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
    if spatial_profile != VJEPA2_EVAL_CROP:
        raise ValueError(f"unknown spatial profile: {spatial_profile}")

    height, width = image.shape[:2]
    short_side = int(256.0 / 224.0 * crop_size)
    scale = short_side / min(height, width)
    resized_width = max(crop_size, int(round(width * scale)))
    resized_height = max(crop_size, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(image).resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
    )
    left = (resized_width - crop_size) // 2
    top = (resized_height - crop_size) // 2
    cropped = resized[top : top + crop_size, left : left + crop_size]
    if cropped.shape != (crop_size, crop_size, 3):
        raise RuntimeError(f"center crop failed: {cropped.shape}")
    return cropped


def sample_color_jitter(
    recording: str,
    seed: int,
    *,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> dict[str, float] | None:
    """Sample one reproducible jitter transform for an entire recording."""

    ranges = {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "hue": float(hue),
    }
    if any(value < 0 for value in ranges.values()):
        raise ValueError("color-jitter ranges must be non-negative")
    if brightness >= 1 or contrast >= 1 or saturation >= 1:
        raise ValueError("brightness/contrast/saturation jitter must be below 1")
    if hue > 0.5:
        raise ValueError("hue jitter must not exceed 0.5")
    if not any(ranges.values()):
        return None
    digest = hashlib.sha256(f"{seed}:{recording}".encode()).digest()
    stable_seed = int.from_bytes(digest[:8], "little", signed=False)
    generator = np.random.default_rng(stable_seed)
    return {
        "brightness": float(generator.uniform(1.0 - brightness, 1.0 + brightness)),
        "contrast": float(generator.uniform(1.0 - contrast, 1.0 + contrast)),
        "saturation": float(generator.uniform(1.0 - saturation, 1.0 + saturation)),
        "hue": float(generator.uniform(-hue, hue)),
        "seed": int(seed),
    }


def apply_color_jitter(
    rgb: np.ndarray,
    parameters: dict[str, float] | None,
) -> np.ndarray:
    """Apply temporally consistent brightness/contrast/saturation/hue jitter."""

    image = np.asarray(rgb, dtype=np.uint8)
    if parameters is None:
        return image
    value = image.astype(np.float32) / 255.0
    value *= float(parameters["brightness"])
    grayscale = (
        value[..., 0] * 0.2989
        + value[..., 1] * 0.5870
        + value[..., 2] * 0.1140
    )
    contrast_center = float(grayscale.mean())
    value = (
        value - contrast_center
    ) * float(parameters["contrast"]) + contrast_center
    grayscale = (
        value[..., 0] * 0.2989
        + value[..., 1] * 0.5870
        + value[..., 2] * 0.1140
    )[..., None]
    value = grayscale + float(parameters["saturation"]) * (value - grayscale)
    value = np.clip(value, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(value, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = np.mod(hsv[..., 0] + float(parameters["hue"]) * 360.0, 360.0)
    value = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)


def _load_model_frames(
    source: Path,
    alignment: TokenAlignment,
    crop_size: int,
    spatial_profile: str,
    color_jitter: dict[str, float] | None = None,
) -> np.ndarray:
    actual_count = source_frame_count(source)
    if actual_count < alignment.source_num_frames:
        raise ValueError(
            f"aligned source count mismatch for {source}: "
            f"{actual_count} < {alignment.source_num_frames}"
        )
    if source.is_dir() and actual_count != alignment.source_num_frames:
        raise ValueError(
            f"aligned image count mismatch for {source}: "
            f"{actual_count} != {alignment.source_num_frames}"
        )
    needed = set(map(int, alignment.model_frame_indices.tolist()))
    decoded: dict[int, np.ndarray] = {}
    if source.is_dir():
        paths = _discover_images(source)
        for frame_index in sorted(needed):
            with Image.open(paths[frame_index]) as image:
                rgb = np.asarray(image.convert("RGB"))
            decoded[frame_index] = apply_color_jitter(
                preprocess_rgb_frame(rgb, crop_size, spatial_profile),
                color_jitter,
            )
    else:
        capture = cv2.VideoCapture(str(source))
        frame_index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in needed:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded[frame_index] = apply_color_jitter(
                    preprocess_rgb_frame(rgb, crop_size, spatial_profile),
                    color_jitter,
                )
            frame_index += 1
        capture.release()
    missing = sorted(needed - set(decoded))
    if missing:
        raise RuntimeError(f"failed to decode aligned frames: {missing[:8]}")
    return np.stack(
        [decoded[int(index)] for index in alignment.model_frame_indices]
    )


def extract_vjepa(
    source,
    feat_extractor,
    device,
    crop_size,
    num_frames,
    batch_size,
    *,
    alignment: TokenAlignment | None = None,
    spatial_profile: str = LEGACY_STRETCH,
    spatial_pool: str = "mean",
    color_jitter: dict[str, float] | None = None,
):
    """Run V-JEPA and return ``(features, source_frame_count)``.

    Omitting ``alignment`` preserves the old dense tuple API.
    """

    source = Path(source)
    if alignment is None:
        count = source_frame_count(source)
        alignment = build_token_alignment(
            count,
            source_fps=1.0,
            sampling_profile=LEGACY_DENSE_PROFILE,
        )
    frames_np = _load_model_frames(
        source,
        alignment,
        crop_size,
        spatial_profile,
        color_jitter,
    )
    model_frame_count = len(frames_np)
    frames = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2)
    frames = (
        frames - VJEPA_MEAN[None, :, None, None]
    ) / VJEPA_STD[None, :, None, None]

    pad_to = ((model_frame_count + num_frames - 1) // num_frames) * num_frames
    if pad_to > model_frame_count:
        pad = frames[-1:].expand(pad_to - model_frame_count, -1, -1, -1)
        frames = torch.cat([frames, pad], dim=0)

    num_clips = pad_to // num_frames
    clips = frames.view(
        num_clips, num_frames, 3, crop_size, crop_size
    ).permute(0, 2, 1, 3, 4)

    all_tokens = []
    device = torch.device(device)
    for index in range(0, num_clips, batch_size):
        batch = clips[index : index + batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = feat_extractor(batch)
        all_tokens.append(output.cpu().float())
    all_tokens = torch.cat(all_tokens, dim=0)

    tokens_per_clip = num_frames // TUBELET
    num_spatial = (crop_size // 16) ** 2
    embed_dim = all_tokens.shape[-1]
    all_tokens = all_tokens.view(
        num_clips, tokens_per_clip, num_spatial, embed_dim
    )
    if spatial_pool == "mean":
        all_tokens = all_tokens.mean(dim=2).reshape(-1, embed_dim)
    elif spatial_pool == "none":
        all_tokens = all_tokens.reshape(-1, num_spatial, embed_dim)
    else:
        raise ValueError("spatial_pool must be mean or none")
    all_tokens = all_tokens[: alignment.num_tokens]
    if len(all_tokens) != alignment.num_tokens:
        raise RuntimeError(
            f"V-JEPA token count {len(all_tokens)} != {alignment.num_tokens}"
        )
    return all_tokens, alignment.source_num_frames


def extract_mano_frames(
    rec_dir: Path,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return full-rate MANO axis-angle ``(F,2,48)`` and validity ``(F,2)``."""

    json_path = rec_dir / "result.json"
    npz_path = rec_dir / "rgb_hawor" / "retarget_input.npz"
    features = np.zeros((num_frames, 2, 48), dtype=np.float32)
    valid = np.zeros((num_frames, 2), dtype=bool)
    if not json_path.exists():
        if not npz_path.exists():
            return None
        with np.load(npz_path) as data:
            n = min(num_frames, data["mano_global_orient"].shape[1])
            valid_raw = np.asarray(data["valid"], dtype=bool)
            if valid_raw.shape == (2, n):
                valid[:n] = valid_raw.T
            elif valid_raw.shape[0] == 2 and valid_raw.shape[1] >= n:
                valid[:n] = valid_raw[:, :n].T
            elif valid_raw.shape[1] == 2 and valid_raw.shape[0] >= n:
                valid[:n] = valid_raw[:n]
            else:
                raise ValueError(f"invalid MANO validity shape: {valid_raw.shape}")
            for side in range(2):
                features[:n, side, :3] = data["mano_global_orient"][side, :n]
                features[:n, side, 3:] = data["mano_hand_pose"][
                    side, :n
                ].reshape(n, 45)
                features[:n, side][~valid[:n, side]] = 0
        return features, valid

    rgb_dir = rec_dir / "rgb"
    frame_names = [path.stem for path in _discover_images(rgb_dir)]
    payload = json.loads(json_path.read_text())
    for frame_index, frame_name in enumerate(frame_names[:num_frames]):
        for hand in payload.get(frame_name, []):
            side = int(hand["is_right"])
            mano = hand.get("mano_params", {})
            if "hand_pose" not in mano:
                continue
            global_orient = rotmat_to_axis_angle(mano["global_orient"][0])
            hand_pose = np.concatenate(
                [
                    rotmat_to_axis_angle(mano["hand_pose"][joint])
                    for joint in range(15)
                ]
            )
            features[frame_index, side] = np.concatenate(
                [global_orient, hand_pose]
            )
            valid[frame_index, side] = True
    return features, valid


def align_mano_to_tokens(
    mano_frames: np.ndarray,
    mano_valid: np.ndarray,
    alignment: TokenAlignment,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align full-rate MANO without averaging invalid and valid poses."""

    features = np.asarray(mano_frames, dtype=np.float32)
    valid = np.asarray(mano_valid, dtype=bool)
    expected_features = (alignment.source_num_frames, 2, 48)
    expected_valid = (alignment.source_num_frames, 2)
    if features.shape != expected_features or valid.shape != expected_valid:
        raise ValueError(
            f"MANO shapes {features.shape}/{valid.shape} != "
            f"{expected_features}/{expected_valid}"
        )
    if alignment.sampling_profile == VJEPA2_4FPS_PROFILE:
        indices = alignment.token_center_frame_indices
        token_features = features[indices].copy()
        token_valid = valid[indices].copy()
        token_features[~token_valid] = 0
    else:
        pairs = alignment.token_frame_indices
        token_features = (
            features[pairs[:, 0]] + features[pairs[:, 1]]
        ) / 2.0
        token_valid = valid[pairs[:, 0]] & valid[pairs[:, 1]]
    return (
        torch.from_numpy(token_features.reshape(alignment.num_tokens, -1)),
        torch.from_numpy(token_valid),
    )


def normalize_action_labels(labels: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate and freeze the dataset-specific class-index contract."""

    normalized = tuple(str(label).strip() for label in labels)
    if not normalized or any(not label for label in normalized):
        raise ValueError("action labels must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"action labels must be unique: {normalized}")
    return normalized


def _frame_labels(
    rec_dir: Path,
    num_frames: int,
    action_labels: list[str] | tuple[str, ...] = ACTION_LABELS,
) -> np.ndarray:
    action_labels = normalize_action_labels(action_labels)
    label_to_index = {label: index for index, label in enumerate(action_labels)}
    labels = np.full(num_frames, -1, dtype=np.int32)
    gt_path = rec_dir / "gt_labels.json"
    if not gt_path.exists():
        return labels
    payload = json.loads(gt_path.read_text())
    for segment in payload.get("segments", []):
        if segment["label"] not in label_to_index:
            continue
        start = max(0, int(segment["start_frame"]))
        end = min(num_frames - 1, int(segment["end_frame"]))
        labels[start : end + 1] = label_to_index[segment["label"]]
    return labels


def labels_for_alignment(
    rec_dir: Path,
    alignment: TokenAlignment,
    *,
    boundary_policy: str = "ignore",
    action_labels: list[str] | tuple[str, ...] = ACTION_LABELS,
) -> torch.Tensor:
    """Align frame labels; mixed 4-FPS tubelets default to ignored (-1)."""

    if boundary_policy not in {"ignore", "center"}:
        raise ValueError("boundary_policy must be ignore or center")
    action_labels = normalize_action_labels(action_labels)
    label_to_index = {label: index for index, label in enumerate(action_labels)}
    frame_labels = _frame_labels(
        rec_dir,
        alignment.source_num_frames,
        action_labels,
    )
    if alignment.sampling_profile == LEGACY_DENSE_PROFILE:
        # Preserve the historical inclusive segment-to-token overwrite rule.
        result = np.full(alignment.num_tokens, -1, dtype=np.int32)
        gt_path = rec_dir / "gt_labels.json"
        if gt_path.exists():
            payload = json.loads(gt_path.read_text())
            for segment in payload.get("segments", []):
                if segment["label"] not in label_to_index:
                    continue
                label = label_to_index[segment["label"]]
                start = max(0, int(segment["start_frame"]) // TUBELET)
                end = min(
                    alignment.num_tokens - 1,
                    int(segment["end_frame"]) // TUBELET,
                )
                result[start : end + 1] = label
        return torch.from_numpy(result)

    pairs = alignment.token_frame_indices
    centers = alignment.token_center_frame_indices
    result = frame_labels[centers].copy()
    if boundary_policy == "ignore":
        mixed = frame_labels[pairs[:, 0]] != frame_labels[pairs[:, 1]]
        result[mixed] = -1
    return torch.from_numpy(result)


def _path_signature(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    if path.is_file():
        return {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    images = _discover_images(path)
    image_stats = [image.stat() for image in images]
    return {
        "path": str(path.resolve()),
        "count": len(images),
        "first": images[0].name,
        "last": images[-1].name,
        "total_size": int(sum(item.st_size for item in image_stats)),
        "max_mtime_ns": int(max(item.st_mtime_ns for item in image_stats)),
        "first_size": int(image_stats[0].st_size),
        "last_size": int(image_stats[-1].st_size),
        "directory_mtime_ns": int(stat.st_mtime_ns),
    }


def _robot_video(rec_dir: Path) -> Path | None:
    candidates = (
        rec_dir
        / "camera_2"
        / "visibility"
        / "processed"
        / "view"
        / "0"
        / "stereo_occlusion"
        / "video_overlay_visibility_haco.mp4",
        rec_dir
        / "inpainting_processed"
        / rec_dir.name
        / "0"
        / "video_overlay_rby1_xhand.mp4",
        rec_dir
        / "inpainting_processed"
        / rec_dir.name
        / "0"
        / "video_overlay_xhand.mp4",
    )
    return next((path for path in candidates if path.is_file()), None)


def _build_provenance(
    rec_dir: Path,
    checkpoint: Path,
    *,
    sampling_profile: str,
    spatial_profile: str,
    source_fps: float,
    sample_fps: float,
    crop_size: int,
    clip_frames: int,
    boundary_policy: str,
    action_labels: list[str] | tuple[str, ...],
    allow_missing_mano: bool = False,
    backbone: str = "vjepa2_vitl256",
    color_jitter: dict[str, float] | None = None,
) -> dict:
    provenance = {
        "checkpoint": _path_signature(checkpoint),
        "rgb": _path_signature(rec_dir / "rgb"),
        "masked_rgb": _path_signature(rec_dir / "hand_cube_mask_overlayed"),
        "robot_video": _path_signature(_robot_video(rec_dir)),
        "hawor": _path_signature(rec_dir / "rgb_hawor" / "retarget_input.npz"),
        "result_json": _path_signature(rec_dir / "result.json"),
        "gt_labels": _path_signature(rec_dir / "gt_labels.json"),
        "parameters": {
            "sampling_profile": sampling_profile,
            "spatial_profile": spatial_profile,
            "source_fps": float(source_fps),
            "sample_fps": float(sample_fps),
            "crop_size": int(crop_size),
            "clip_frames": int(clip_frames),
            "tubelet_size": TUBELET,
            "label_boundary_policy": boundary_policy,
            "action_labels": list(normalize_action_labels(action_labels)),
            "allow_missing_mano": bool(allow_missing_mano),
            "backbone": backbone,
        },
    }
    if color_jitter is not None:
        provenance["parameters"]["color_jitter"] = {
            key: float(value) if key != "seed" else int(value)
            for key, value in color_jitter.items()
        }
    return provenance


def feature_sampling_signature(bundle: dict) -> tuple:
    """Return the training/inference compatibility signature."""

    return (
        bundle.get("sampling_profile", LEGACY_DENSE_PROFILE),
        float(bundle.get("sample_fps", bundle.get("source_fps", 0.0))),
        float(bundle.get("token_rate_hz", 0.0)),
        int(bundle.get("clip_frames", 16)),
        int(bundle.get("tubelet_size", TUBELET)),
        bundle.get("spatial_profile", LEGACY_STRETCH),
    )


def validate_feature_bundle(
    bundle: dict,
    *,
    expected_provenance: dict | None = None,
) -> None:
    required = {
        "vjepa_orig",
        "mano",
        "mano_valid_per_token",
        "labels_per_token",
        "num_frames",
        "num_tokens",
        "recording",
        "sampled_frame_indices",
        "token_frame_indices",
        "token_center_frame_indices",
        "frame_to_token",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"feature bundle missing keys: {missing}")
    token_count = int(bundle["num_tokens"])
    frame_count = int(bundle["num_frames"])
    if token_count <= 0 or frame_count <= 0:
        raise ValueError("feature bundle frame/token counts must be positive")
    shapes = {
        "vjepa_orig": (token_count, 1024),
        "mano": (token_count, 96),
        "mano_valid_per_token": (token_count, 2),
        "labels_per_token": (token_count,),
        "token_frame_indices": (token_count, TUBELET),
        "token_center_frame_indices": (token_count,),
        "frame_to_token": (frame_count,),
    }
    for key, shape in shapes.items():
        if tuple(bundle[key].shape) != shape:
            raise ValueError(f"{key} shape {tuple(bundle[key].shape)} != {shape}")
    action_labels = normalize_action_labels(
        bundle.get("action_labels", ACTION_LABELS)
    )
    labels = torch.as_tensor(bundle["labels_per_token"])
    if labels.numel() and (
        int(labels.min()) < -1 or int(labels.max()) >= len(action_labels)
    ):
        raise ValueError(
            "labels_per_token contains an index outside action_labels"
        )
    sampled = torch.as_tensor(bundle["sampled_frame_indices"])
    if sampled.ndim != 1 or len(sampled) <= 0:
        raise ValueError("sampled_frame_indices must be a non-empty vector")
    if int(sampled.min()) < 0 or int(sampled.max()) >= frame_count:
        raise ValueError("sampled_frame_indices contains an out-of-range frame")
    token_frames = torch.as_tensor(bundle["token_frame_indices"])
    centers = torch.as_tensor(bundle["token_center_frame_indices"])
    if int(token_frames.min()) < 0 or int(token_frames.max()) >= frame_count:
        raise ValueError("token_frame_indices contains an out-of-range frame")
    if int(centers.min()) < 0 or int(centers.max()) >= frame_count:
        raise ValueError("token_center_frame_indices contains an out-of-range frame")
    for key in ("vjepa_orig_masked", "vjepa_robot"):
        if key in bundle and tuple(bundle[key].shape) != (token_count, 1024):
            raise ValueError(f"{key} is not token-aligned")
    if "vjepa_orig_dense" in bundle:
        dense = torch.as_tensor(bundle["vjepa_orig_dense"])
        if dense.ndim != 3 or dense.shape[0] != token_count or dense.shape[2] != 1024:
            raise ValueError(
                "vjepa_orig_dense must have shape "
                f"({token_count}, spatial_tokens, 1024), got {tuple(dense.shape)}"
            )
        if dense.shape[1] <= 0:
            raise ValueError("vjepa_orig_dense must contain spatial tokens")
    mapping = torch.as_tensor(bundle["frame_to_token"])
    if int(mapping.min()) < 0 or int(mapping.max()) >= token_count:
        raise ValueError("frame_to_token contains an out-of-range token")
    if int(bundle.get("feature_schema_version", 0)) != FEATURE_SCHEMA_VERSION:
        raise ValueError("feature bundle is not schema version 2")
    if expected_provenance is not None and bundle.get("input_provenance") != expected_provenance:
        raise ValueError("feature bundle input provenance is stale")


def _resolve_source_fps(
    rec_dir: Path,
    requested: float | None,
    sampling_profile: str,
) -> float:
    gt_path = rec_dir / "gt_labels.json"
    gt_fps = None
    if gt_path.exists():
        gt_fps = float(json.loads(gt_path.read_text())["fps"])
    if requested is not None:
        if gt_fps is not None and not np.isclose(requested, gt_fps, atol=1.0e-6):
            raise ValueError(
                f"{rec_dir}: --source_fps {requested} != GT fps {gt_fps}"
            )
        return float(requested)
    if gt_fps is not None:
        return gt_fps
    if sampling_profile == VJEPA2_4FPS_PROFILE:
        raise ValueError(
            f"{rec_dir}: vjepa2_4fps needs --source_fps or gt_labels.json fps"
        )
    return 30.0


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--recording_glob", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONES,
        default="vjepa2_vitl256",
        help="Frozen video encoder used to build vjepa_orig features.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--store_dense_tokens",
        action="store_true",
        help=(
            "Also store FP16 per-patch features as vjepa_orig_dense for "
            "learned spatial pooling."
        ),
    )
    parser.add_argument("--color_jitter_brightness", type=float, default=0.0)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.0)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.0)
    parser.add_argument("--color_jitter_hue", type=float, default=0.0)
    parser.add_argument("--color_jitter_seed", type=int, default=0)
    parser.add_argument(
        "--sampling_profile",
        choices=SAMPLING_PROFILES,
        default=LEGACY_DENSE_PROFILE,
    )
    parser.add_argument("--source_fps", type=float, default=None)
    parser.add_argument("--sample_fps", type=float, default=4.0)
    parser.add_argument(
        "--spatial_profile",
        choices=SPATIAL_PROFILES,
        default=None,
    )
    parser.add_argument(
        "--label_boundary_policy",
        choices=("ignore", "center"),
        default="ignore",
    )
    parser.add_argument(
        "--action_labels",
        default=",".join(ACTION_LABELS),
        help=(
            "Comma-separated dataset label vocabulary in exact class-index "
            "order (default: the legacy Milk vocabulary)."
        ),
    )
    parser.add_argument(
        "--allow_missing_mano",
        action="store_true",
        help=(
            "Build a V-JEPA-only bundle when HaWoR/MANO is absent by storing "
            "zero MANO values with all validity flags false."
        ),
    )
    args = parser.parse_args()

    action_labels = normalize_action_labels(args.action_labels.split(","))

    if args.num_frames <= 0 or args.num_frames % TUBELET:
        raise ValueError("--num_frames must be a positive multiple of 2")
    if args.backbone == VJEPA21_BACKBONE and args.crop_size != 384:
        raise ValueError("V-JEPA 2.1 ViT-L requires --crop_size 384")
    spatial_profile = args.spatial_profile or (
        VJEPA2_EVAL_CROP
        if args.sampling_profile == VJEPA2_4FPS_PROFILE
        else LEGACY_STRETCH
    )
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    data_root = Path(args.data_root)
    patterns = [part.strip() for part in args.recording_glob.split(",") if part.strip()]
    recordings = sorted(
        {
            directory
            for pattern in patterns
            for directory in data_root.glob(pattern)
            if directory.is_dir()
        }
    )
    if not recordings:
        raise ValueError(f"no recordings under {data_root} for {patterns}")

    jobs = []
    for rec_dir in recordings:
        rgb_dir = rec_dir / "rgb"
        if not rgb_dir.exists() or not (rgb_dir.is_dir() or rgb_dir.is_file()):
            raise FileNotFoundError(f"{rec_dir}: missing rgb image directory/video")
        source_fps = _resolve_source_fps(
            rec_dir, args.source_fps, args.sampling_profile
        )
        original_count = aligned_source_frame_count(rec_dir, rgb_dir)
        alignment = build_token_alignment(
            original_count,
            source_fps,
            sampling_profile=args.sampling_profile,
            sample_fps=args.sample_fps,
        )
        color_jitter = sample_color_jitter(
            rec_dir.name,
            args.color_jitter_seed,
            brightness=args.color_jitter_brightness,
            contrast=args.color_jitter_contrast,
            saturation=args.color_jitter_saturation,
            hue=args.color_jitter_hue,
        )
        provenance = _build_provenance(
            rec_dir,
            checkpoint,
            sampling_profile=args.sampling_profile,
            spatial_profile=spatial_profile,
            source_fps=source_fps,
            sample_fps=alignment.sample_fps,
            crop_size=args.crop_size,
            clip_frames=args.num_frames,
            boundary_policy=args.label_boundary_policy,
            action_labels=action_labels,
            allow_missing_mano=args.allow_missing_mano,
            backbone=args.backbone,
            color_jitter=color_jitter,
        )
        output = rec_dir / "features.pt"
        if output.exists() and not args.overwrite:
            try:
                cached = torch.load(output, map_location="cpu", weights_only=True)
                validate_feature_bundle(
                    cached, expected_provenance=provenance
                )
                if args.store_dense_tokens and "vjepa_orig_dense" not in cached:
                    raise ValueError("cached bundle has no dense spatial tokens")
            except Exception as error:
                raise RuntimeError(
                    f"{output} is stale/incomplete; rerun with --overwrite"
                ) from error
            print(f"[skip] {rec_dir.name}: validated {output.name}")
            continue
        jobs.append((rec_dir, alignment, provenance, color_jitter))

    if not jobs:
        print("All feature bundles are already valid.")
        return

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print("Loading V-JEPA encoder ...")
    if args.backbone == VJEPA21_BACKBONE:
        encoder = load_pretrained_vjepa21_encoder(
            checkpoint_path=checkpoint,
            device=device,
            crop_size=args.crop_size,
            patch_size=16,
            num_frames=args.num_frames,
            tubelet_size=TUBELET,
        )
        feature_extractor = VJEPA21FeatureExtractor(
            encoder, pool="none"
        ).to(device)
    else:
        encoder = load_pretrained_encoder(
            checkpoint_path=checkpoint,
            device=device,
            model_name="vit_large",
            crop_size=args.crop_size,
            patch_size=16,
            num_frames=args.num_frames,
            tubelet_size=TUBELET,
        )
        feature_extractor = VJEPAFeatureExtractor(
            encoder, pool="none"
        ).to(device)

    for job_index, (rec_dir, alignment, provenance, color_jitter) in enumerate(jobs, 1):
        print(
            f"[{job_index}/{len(jobs)}] {rec_dir.name}: "
            f"{alignment.source_num_frames} source frames -> "
            f"{len(alignment.sampled_frame_indices)} sampled -> "
            f"{alignment.num_tokens} tokens"
        )
        extracted_orig, decoded_count = extract_vjepa(
            rec_dir / "rgb",
            feature_extractor,
            device,
            args.crop_size,
            args.num_frames,
            args.batch_size,
            alignment=alignment,
            spatial_profile=spatial_profile,
            spatial_pool="none" if args.store_dense_tokens else "mean",
            color_jitter=color_jitter,
        )
        if decoded_count != alignment.source_num_frames:
            raise RuntimeError("RGB source count changed during extraction")

        mano_payload = extract_mano_frames(
            rec_dir, alignment.source_num_frames
        )
        if mano_payload is None:
            if not args.allow_missing_mano:
                raise FileNotFoundError(
                    f"{rec_dir}: missing result.json and "
                    "rgb_hawor/retarget_input.npz"
                )
            mano = torch.zeros(alignment.num_tokens, 96, dtype=torch.float32)
            mano_valid = torch.zeros(
                alignment.num_tokens, 2, dtype=torch.bool
            )
            mano_source = "missing_zero_filled"
        else:
            mano, mano_valid = align_mano_to_tokens(
                *mano_payload, alignment
            )
            mano_source = "hawor"
        if args.store_dense_tokens:
            vjepa_orig_dense = extracted_orig.to(torch.float16)
            vjepa_orig = extracted_orig.mean(dim=1)
        else:
            vjepa_orig_dense = None
            vjepa_orig = extracted_orig
        bundle = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "video_backbone": args.backbone,
            "vjepa_orig": vjepa_orig,
            "mano": mano,
            "mano_valid_per_token": mano_valid,
            "mano_source": mano_source,
            "labels_per_token": labels_for_alignment(
                rec_dir,
                alignment,
                boundary_policy=args.label_boundary_policy,
                action_labels=action_labels,
            ),
            "action_labels": list(action_labels),
            "num_frames": alignment.source_num_frames,
            "num_tokens": alignment.num_tokens,
            "recording": rec_dir.name,
            "sampling_profile": alignment.sampling_profile,
            "source_fps": alignment.source_fps,
            "sample_fps": alignment.sample_fps,
            "token_rate_hz": alignment.sample_fps / TUBELET,
            "clip_frames": args.num_frames,
            "tubelet_size": TUBELET,
            "spatial_profile": spatial_profile,
            "label_boundary_policy": args.label_boundary_policy,
            "sampled_frame_indices": torch.from_numpy(
                alignment.sampled_frame_indices.copy()
            ),
            "token_frame_indices": torch.from_numpy(
                alignment.token_frame_indices.copy()
            ),
            "token_center_frame_indices": torch.from_numpy(
                alignment.token_center_frame_indices.copy()
            ),
            "frame_to_token": torch.from_numpy(
                alignment.frame_to_token.copy()
            ),
            "input_provenance": provenance,
        }
        if color_jitter is not None:
            bundle["color_jitter"] = provenance["parameters"]["color_jitter"]
        if vjepa_orig_dense is not None:
            bundle["vjepa_orig_dense"] = vjepa_orig_dense

        masked_dir = rec_dir / "hand_cube_mask_overlayed"
        if masked_dir.is_dir():
            bundle["vjepa_orig_masked"], _ = extract_vjepa(
                masked_dir,
                feature_extractor,
                device,
                args.crop_size,
                args.num_frames,
                args.batch_size,
                alignment=alignment,
                spatial_profile=spatial_profile,
            )
        robot_video = _robot_video(rec_dir)
        if robot_video is not None:
            bundle["vjepa_robot"], _ = extract_vjepa(
                robot_video,
                feature_extractor,
                device,
                args.crop_size,
                args.num_frames,
                args.batch_size,
                alignment=alignment,
                spatial_profile=spatial_profile,
            )

        validate_feature_bundle(bundle, expected_provenance=provenance)
        output = rec_dir / "features.pt"
        _atomic_torch_save(bundle, output)
        reloaded = torch.load(output, map_location="cpu", weights_only=True)
        validate_feature_bundle(reloaded, expected_provenance=provenance)
        feature_keys = [
            key for key in ("vjepa_orig", "vjepa_orig_masked", "vjepa_robot")
            if key in bundle
        ]
        print(f"[saved] {output} T={alignment.num_tokens} {feature_keys}")

    print("Done.")


if __name__ == "__main__":
    main()
