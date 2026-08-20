"""Remove the human arm while completing object pixels hidden by the hand.

The existing 08_04 object track is modal, but contact-boundary pixels can be
owned by both the object track and the human hand.  This stage first removes
HaWoR-hand overlap from the trusted observations, then runs a second,
object-only E2FGVI pass over a conservative inferred support:

    trusted = input_modal & ~dilate(HaWoR_hand_support)
    hidden = (amodal_shape - trusted) & dilate(HaWoR_hand_support)

The inferred shape is a per-component convex completion.  Frames whose modal
track collapses or grows into a receptacle use the closest reliable frame,
warped by a similarity transform estimated from the HaWoR hand masks.  The
completed camera-Z surface is filled only inside ``hidden`` from the nearest
valid modal surface sample.  It remains a visible-view depth proxy, not a
watertight object model.

Outputs are published atomically under ``--out_dir``:

    video_hand_removed_modal_only.mp4
    video_object_completed.mp4
    debug_object_completion.mp4
    object_mask_observed_clean.npy
    object_mask_amodal.npy
    object_surface_depth_completed.npy
    completion_evidence.npz
    report.json
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from atomic_directory_publish import publish_directory


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class Segment:
    label: str
    start: int
    end: int


def probe_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    metadata = VideoMetadata(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
    )
    capture.release()
    if (
        metadata.width <= 0
        or metadata.height <= 0
        or metadata.frame_count <= 0
        or metadata.fps <= 0.0
    ):
        raise ValueError(f"invalid video metadata: {path}")
    return metadata


def load_segments(path: Path, frame_count: int) -> list[Segment]:
    payload = json.loads(path.read_text())
    result: list[Segment] = []
    previous_end = -1
    for item in payload.get("segments", []):
        start = max(0, int(item["start_frame"]))
        end = min(frame_count - 1, int(item["end_frame"]))
        label = str(item["label"])
        if start > end:
            continue
        if start <= previous_end:
            raise ValueError("annotation segments overlap or are non-monotonic")
        previous_end = end
        if label.casefold() != "trans":
            result.append(Segment(label=label, start=start, end=end))
    if not result:
        raise ValueError("annotation contains no manipulated-object segments")
    return result


def _ellipse_kernel(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("morphology radius must be non-negative")
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    if value.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if radius == 0:
        return value.astype(bool)
    return cv2.dilate(value, _ellipse_kernel(radius), iterations=1).astype(bool)


def clean_modal_observations(
    modal: np.ndarray,
    hand_support: np.ndarray,
    *,
    hand_dilate_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an input modal track into trusted RGB and hand-contested pixels."""
    modal_value = np.asanyarray(modal)
    hand_value = np.asanyarray(hand_support)
    if modal_value.ndim != 3 or modal_value.shape != hand_value.shape:
        raise ValueError("modal and hand masks must share (T,H,W)")
    if hand_dilate_px < 0:
        raise ValueError("hand dilation must be non-negative")
    trusted = np.zeros(modal_value.shape, dtype=bool)
    contested = np.zeros_like(trusted)
    for frame_index in range(modal_value.shape[0]):
        visible = np.asarray(modal_value[frame_index], dtype=bool)
        hand_region = dilate_mask(
            np.asarray(hand_value[frame_index], dtype=bool),
            hand_dilate_px,
        )
        contested[frame_index] = visible & hand_region
        trusted[frame_index] = visible & ~hand_region
    return trusted, contested


def clean_modal_observations_with_contact(
    modal: np.ndarray,
    hand_support: np.ndarray,
    contact_support: np.ndarray,
    *,
    hand_dilate_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove only modal pixels jointly supported by hand and HaCo contact.

    HaCo does not identify object RGB ownership.  It is therefore used only as
    a spatial selector on top of the projected HaWoR hand, avoiding the much
    broader removal of every modal pixel beneath the hand silhouette.
    """
    modal_value = np.asanyarray(modal)
    hand_value = np.asanyarray(hand_support)
    contact_value = np.asanyarray(contact_support)
    if modal_value.ndim != 3 or not (
        modal_value.shape == hand_value.shape == contact_value.shape
    ):
        raise ValueError("modal, hand, and contact masks must share (T,H,W)")
    if hand_dilate_px < 0:
        raise ValueError("hand dilation must be non-negative")
    trusted = np.zeros(modal_value.shape, dtype=bool)
    contested = np.zeros_like(trusted)
    for frame_index in range(modal_value.shape[0]):
        visible = np.asarray(modal_value[frame_index], dtype=bool)
        hand_region = dilate_mask(
            np.asarray(hand_value[frame_index], dtype=bool),
            hand_dilate_px,
        )
        contested[frame_index] = (
            visible
            & hand_region
            & np.asarray(contact_value[frame_index], dtype=bool)
        )
        trusted[frame_index] = visible & ~contested[frame_index]
    return trusted, contested


def select_haco_hidden_components(
    raw_hidden: np.ndarray,
    contact_support: np.ndarray,
    segments: Iterable[Segment],
    *,
    temporal_grace_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep hidden connected components anchored by projected HaCo contact.

    A short segment-bounded temporal grace keeps the raw candidate on contact
    dropouts adjacent to a directly selected frame.  HaCo remains a selector:
    this function can only remove pixels from ``raw_hidden`` and never creates
    object support of its own.
    """
    hidden_value = np.asanyarray(raw_hidden)
    contact_value = np.asanyarray(contact_support)
    if hidden_value.ndim != 3 or hidden_value.shape != contact_value.shape:
        raise ValueError("hidden and contact masks must share (T,H,W)")
    if temporal_grace_frames < 0:
        raise ValueError("temporal grace must be non-negative")
    selected = np.zeros(hidden_value.shape, dtype=bool)
    direct_frames = np.zeros(hidden_value.shape[0], dtype=bool)
    for frame_index in range(hidden_value.shape[0]):
        frame_hidden = np.asarray(hidden_value[frame_index], dtype=bool)
        frame_contact = np.asarray(contact_value[frame_index], dtype=bool)
        if not np.any(frame_hidden) or not np.any(frame_contact):
            continue
        component_count, labels = cv2.connectedComponents(
            frame_hidden.astype(np.uint8),
            connectivity=8,
        )
        if component_count <= 1:
            continue
        touched = np.unique(labels[frame_contact])
        touched = touched[touched != 0]
        if not len(touched):
            continue
        selected[frame_index] = np.isin(labels, touched)
        direct_frames[frame_index] = bool(selected[frame_index].any())

    temporal_frames = np.zeros_like(direct_frames)
    if temporal_grace_frames:
        for segment in segments:
            start = max(0, int(segment.start))
            end = min(hidden_value.shape[0] - 1, int(segment.end))
            for frame_index in range(start, end + 1):
                if direct_frames[frame_index] or not np.any(
                    hidden_value[frame_index]
                ):
                    continue
                left = max(start, frame_index - temporal_grace_frames)
                right = min(end + 1, frame_index + temporal_grace_frames + 1)
                if np.any(direct_frames[left:right]):
                    selected[frame_index] = np.asarray(
                        hidden_value[frame_index],
                        dtype=bool,
                    )
                    temporal_frames[frame_index] = True
    if np.any(selected & ~np.asarray(hidden_value, dtype=bool)):
        raise RuntimeError("HaCo selection created pixels outside raw hidden support")
    return selected, direct_frames, temporal_frames


def auxiliary_frame_indices(
    frame_count: int,
    frame_offset: int,
) -> np.ndarray:
    """Map primary-frame indices to an aligned auxiliary timeline.

    The stereo manifest convention is ``aux = primary + frame_offset``.
    ``-1`` marks boundary frames that have no aligned auxiliary observation;
    callers must fail open to primary-view evidence for those frames.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    indices = np.arange(frame_count, dtype=np.int64) + int(frame_offset)
    valid = (indices >= 0) & (indices < frame_count)
    return np.where(valid, indices, -1)


def build_dual_haco_contact_support(
    *,
    contact_dir: Path,
    auxiliary_contact_dir: Path | None,
    hawor_npz: Path,
    modal: np.ndarray,
    metadata: VideoMetadata,
    side: str,
    auxiliary_side: str,
    auxiliary_frame_offset: int,
    contact_score_threshold: float,
    contact_point_threshold: float,
    contact_top_fraction: float,
    min_contact_points: int,
    contact_radius_px: int,
    point_probe_radius_px: int,
    hidden_fraction_on: float,
    hidden_fraction_off: float,
    min_on_frames: int,
    hold_frames: int,
) -> dict[str, object]:
    """Project MH HaCo contacts and use SH only as per-finger confidence.

    This intentionally reuses the established contact-selection and temporal
    activation implementation from the overlay compositor.  No SH pixel or
    geometry is projected into MH coordinates.
    """
    from composite_rb5_contact_occlusion import (
        FINGER_NAMES,
        OcclusionConfig,
        _contact_frame_features,
        _contact_frame_selection,
        contact_activation_tracks,
        disk_support,
        sample_local_fraction,
    )

    config = OcclusionConfig(
        contact_score_threshold=contact_score_threshold,
        contact_point_threshold=contact_point_threshold,
        contact_top_fraction=contact_top_fraction,
        min_contact_points=min_contact_points,
        contact_radius_px=contact_radius_px,
        point_probe_radius_px=point_probe_radius_px,
        hidden_fraction_on=hidden_fraction_on,
        hidden_fraction_off=hidden_fraction_off,
        min_on_frames=min_on_frames,
        hold_frames=hold_frames,
    )
    config.validate()
    modal_value = np.asanyarray(modal)
    expected_shape = (
        metadata.frame_count,
        metadata.height,
        metadata.width,
    )
    if modal_value.shape != expected_shape:
        raise ValueError(f"modal mask must have shape {expected_shape}")
    if side not in {"left", "right"} or auxiliary_side not in {
        "left",
        "right",
    }:
        raise ValueError("HaCo side must be left or right")

    assets = Path(__file__).resolve().parents[1] / "retargeting" / "assets"
    parts = np.load(assets / f"finger_part_{side}.npy").astype(np.int32)
    palmar = np.load(assets / f"palmar_mask_{side}.npy").astype(bool)
    primary_scores = np.zeros(
        (metadata.frame_count, len(FINGER_NAMES)),
        dtype=np.float32,
    )
    hidden_fraction = np.zeros_like(primary_scores)
    all_points_uv: list[dict[str, np.ndarray]] = []
    with np.load(hawor_npz, allow_pickle=False) as retarget:
        vertex_key = f"verts_{side}"
        if vertex_key not in retarget.files:
            raise KeyError(f"HaWoR input lacks {vertex_key}")
        if retarget[vertex_key].shape[:2] != (metadata.frame_count, 778):
            raise ValueError("HaWoR vertex track must have shape (T,778,3)")
        focal_px = float(retarget["img_focal"])
        if not np.isfinite(focal_px) or focal_px <= 0:
            raise ValueError("HaWoR img_focal must be finite and positive")
        for frame_index in range(metadata.frame_count):
            contact_path = contact_dir / f"rgb_frame{frame_index:06d}.npz"
            frame_scores, points_uv, _points_z = _contact_frame_features(
                contact_path=contact_path,
                retarget=retarget,
                frame_index=frame_index,
                side=side,
                parts=parts,
                palmar=palmar,
                config=config,
                focal_output_px=focal_px,
                output_width=metadata.width,
                output_height=metadata.height,
            )
            primary_scores[frame_index] = frame_scores
            all_points_uv.append(points_uv)
            visible = np.asarray(modal_value[frame_index], dtype=bool)
            for finger_index, finger in enumerate(FINGER_NAMES):
                fractions = sample_local_fraction(
                    visible,
                    points_uv[finger],
                    config.point_probe_radius_px,
                )
                hidden_fraction[frame_index, finger_index] = (
                    float((fractions >= 0.35).mean())
                    if len(fractions)
                    else 0.0
                )
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[haco-evidence] {frame_index + 1}/{metadata.frame_count}",
                    flush=True,
                )

    auxiliary_scores: np.ndarray | None = None
    mapped_auxiliary_indices = auxiliary_frame_indices(
        metadata.frame_count,
        auxiliary_frame_offset,
    )
    if auxiliary_contact_dir is not None:
        auxiliary_parts = np.load(
            assets / f"finger_part_{auxiliary_side}.npy"
        ).astype(np.int32)
        auxiliary_palmar = np.load(
            assets / f"palmar_mask_{auxiliary_side}.npy"
        ).astype(bool)
        auxiliary_scores = np.full_like(primary_scores, np.nan)
        for frame_index, auxiliary_frame_index in enumerate(
            mapped_auxiliary_indices
        ):
            if auxiliary_frame_index < 0:
                continue
            auxiliary_path = (
                auxiliary_contact_dir
                / f"rgb_frame{int(auxiliary_frame_index):06d}.npz"
            )
            frame_scores, _selected = _contact_frame_selection(
                contact_path=auxiliary_path,
                frame_index=int(auxiliary_frame_index),
                side=auxiliary_side,
                parts=auxiliary_parts,
                palmar=auxiliary_palmar,
                config=config,
            )
            auxiliary_scores[frame_index] = frame_scores

    _primary_fused, _primary_evidence, primary_active, _primary_gates = (
        contact_activation_tracks(
            primary_scores,
            None,
            hidden_fraction,
            config,
        )
    )
    fused_scores, evidence, active, gates = contact_activation_tracks(
        primary_scores,
        auxiliary_scores,
        hidden_fraction,
        config,
    )
    support = np.zeros(expected_shape, dtype=bool)
    seed_pixels_by_finger = np.zeros_like(primary_scores, dtype=np.int64)
    projected_active = np.zeros_like(active)
    for frame_index in range(metadata.frame_count):
        for finger_index, finger in enumerate(FINGER_NAMES):
            if not active[frame_index, finger_index]:
                continue
            finger_support = disk_support(
                all_points_uv[frame_index][finger],
                (metadata.height, metadata.width),
                config.contact_radius_px,
            )
            if np.any(finger_support):
                projected_active[frame_index, finger_index] = True
                support[frame_index] |= finger_support
                seed_pixels_by_finger[frame_index, finger_index] = int(
                    finger_support.sum()
                )
    return {
        "finger_names": np.asarray(FINGER_NAMES),
        "primary_focal_px": focal_px,
        "primary_scores": primary_scores,
        "auxiliary_scores": auxiliary_scores,
        "auxiliary_frame_indices": mapped_auxiliary_indices,
        "fused_scores": fused_scores,
        "hidden_fraction": hidden_fraction,
        "evidence": evidence,
        "primary_active": primary_active,
        "active": active,
        "projected_active": projected_active,
        "support": support,
        "seed_pixels_by_finger": seed_pixels_by_finger,
        "gates": gates,
        "config": config,
    }


def component_convex_hulls(
    mask: np.ndarray,
    *,
    minimum_contour_area: float = 30.0,
) -> np.ndarray:
    """Fill each component hull independently so separate objects never join."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    output = np.zeros_like(binary)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        if cv2.contourArea(contour) < minimum_contour_area:
            continue
        cv2.fillConvexPoly(output, cv2.convexHull(contour), 1)
    return output.astype(bool)


def _mask_pose(mask: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Return centroid, undirected PCA angle, and RMS radius for a hand mask."""
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if len(rows) < 20:
        return None
    points = np.stack((columns, rows), axis=1).astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / float(len(points))
    values, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, int(np.argmax(values))]
    angle = math.atan2(float(principal[1]), float(principal[0]))
    radius = math.sqrt(max(float(np.trace(covariance)), 1.0))
    return center, angle, radius


def _undirected_angle_delta(target: float, source: float) -> float:
    delta = target - source
    while delta > math.pi / 2.0:
        delta -= math.pi
    while delta < -math.pi / 2.0:
        delta += math.pi
    return delta


def warp_mask_by_hand_pose(
    source_mask: np.ndarray,
    source_hand: np.ndarray,
    target_hand: np.ndarray,
) -> np.ndarray:
    """Warp a held-object prior using the hand's 2-D similarity motion."""
    source = np.asarray(source_mask, dtype=np.uint8)
    if source.shape != np.asarray(source_hand).shape or source.shape != np.asarray(
        target_hand
    ).shape:
        raise ValueError("source/object hand masks must share one shape")
    source_pose = _mask_pose(source_hand)
    target_pose = _mask_pose(target_hand)
    if source_pose is None or target_pose is None:
        return np.zeros_like(source, dtype=bool)
    source_center, source_angle, source_radius = source_pose
    target_center, target_angle, target_radius = target_pose
    scale = float(np.clip(target_radius / source_radius, 0.65, 1.55))
    angle = _undirected_angle_delta(target_angle, source_angle)
    cosine = math.cos(angle) * scale
    sine = math.sin(angle) * scale
    linear = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    translation = target_center - linear @ source_center
    transform = np.concatenate((linear, translation[:, None]), axis=1)
    height, width = source.shape
    return cv2.warpAffine(
        source,
        transform.astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def infer_hidden_support(
    modal: np.ndarray,
    hand_support: np.ndarray,
    *,
    prior: np.ndarray | None = None,
    hand_dilate_px: int = 4,
    max_modal_distance_px: float = 64.0,
    minimum_contour_area: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative amodal and newly inferred hidden-object masks."""
    visible = np.asarray(modal, dtype=bool)
    hand = np.asarray(hand_support, dtype=bool)
    if visible.shape != hand.shape:
        raise ValueError("modal and hand masks must share one shape")
    if prior is None:
        shape_prior = component_convex_hulls(
            visible,
            minimum_contour_area=minimum_contour_area,
        )
        constrain_to_modal = True
    else:
        shape_prior = np.asarray(prior, dtype=bool)
        if shape_prior.shape != visible.shape:
            raise ValueError("temporal prior shape differs from modal mask")
        constrain_to_modal = False
    hand_region = dilate_mask(hand, hand_dilate_px)
    hidden = shape_prior & ~visible & hand_region
    if constrain_to_modal and np.any(visible) and max_modal_distance_px > 0:
        distance = cv2.distanceTransform(
            (~visible).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        hidden &= distance <= float(max_modal_distance_px)
    hidden = cv2.morphologyEx(
        hidden.astype(np.uint8),
        cv2.MORPH_CLOSE,
        _ellipse_kernel(2),
    ).astype(bool)
    hidden &= ~visible
    hidden &= hand_region
    return visible | hidden, hidden


def _reliable_frames(
    modal: np.ndarray,
    segment: Segment,
    *,
    minimum_ratio: float,
    maximum_ratio: float,
) -> tuple[np.ndarray, float]:
    areas = np.asarray(
        [
            int(np.asarray(modal[index], dtype=bool).sum())
            for index in range(segment.start, segment.end + 1)
        ],
        dtype=np.float64,
    )
    # Include collapsed frames in the primary median.  Excluding them can move
    # a bimodal interval (for example a held carton that later merges with a
    # receptacle mask) onto the large leaked component.  Only fall back to the
    # positive subset when at least half of the annotated interval is empty.
    median = float(np.median(areas))
    if median <= 100.0:
        positive = areas[areas > 100]
        median = float(np.median(positive)) if len(positive) else 0.0
    if median <= 0.0:
        return np.empty(0, dtype=np.int64), median
    local = np.nonzero(
        (areas >= minimum_ratio * median) & (areas <= maximum_ratio * median)
    )[0]
    return local.astype(np.int64) + segment.start, median


def select_reference_bank(
    modal: np.ndarray,
    hand_support: np.ndarray,
    reliable_frames: np.ndarray,
    *,
    maximum_references: int = 6,
    minimum_frame_gap: int = 5,
) -> list[int]:
    """Choose widely spaced frames with much visible object and little hand."""
    candidates: list[tuple[float, int]] = []
    for frame_index in reliable_frames:
        index = int(frame_index)
        visible = np.asarray(modal[index], dtype=bool)
        hand = np.asarray(hand_support[index], dtype=bool)
        area = int(visible.sum())
        overlap = int(np.sum(visible & hand))
        candidates.append((float(area) - 1.5 * float(overlap), index))
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _score, frame_index in candidates:
        if all(abs(frame_index - other) >= minimum_frame_gap for other in selected):
            selected.append(frame_index)
        if len(selected) >= maximum_references:
            break
    return selected


def best_warped_reference_prior(
    visible: np.ndarray,
    target_hand: np.ndarray,
    reference_bank: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    target_frame: int,
    minimum_coverage: float = 0.45,
) -> tuple[np.ndarray, int, float] | None:
    """Return the hand-aligned reference silhouette that covers current RGB."""
    current = np.asarray(visible, dtype=bool)
    hand = np.asarray(target_hand, dtype=bool)
    current_area = int(current.sum())
    if current_area < 100:
        return None
    best: tuple[float, np.ndarray, int, float] | None = None
    kernel = _ellipse_kernel(3)
    for reference_frame, reference_hull, reference_hand in reference_bank:
        if reference_frame == target_frame and len(reference_bank) > 1:
            continue
        warped = warp_mask_by_hand_pose(reference_hull, reference_hand, hand)
        warped_area = int(warped.sum())
        if warped_area <= 0:
            continue
        size_ratio = warped_area / float(current_area)
        if not 0.30 <= size_ratio <= 4.0:
            continue
        expanded = cv2.dilate(warped.astype(np.uint8), kernel, iterations=1).astype(
            bool
        )
        coverage = float(np.sum(expanded & current)) / float(current_area)
        score = coverage - 0.0005 * abs(reference_frame - target_frame)
        if best is None or score > best[0]:
            best = (score, warped, reference_frame, coverage)
    if best is None or best[3] < minimum_coverage:
        return None
    return best[1], best[2], best[3]


def build_completion_masks(
    modal: np.ndarray,
    hand_support: np.ndarray,
    arm_mask: np.ndarray,
    segments: Iterable[Segment],
    *,
    hand_dilate_px: int = 4,
    arm_dilate_px: int = 5,
    object_inpaint_dilate_px: int = 8,
    max_modal_distance_px: float = 64.0,
    reliable_minimum_ratio: float = 0.40,
    reliable_maximum_ratio: float = 1.80,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    """Build amodal, hidden, and object-only E2FGVI masks for one clip."""
    modal_value = np.asanyarray(modal)
    hand_value = np.asanyarray(hand_support)
    arm_value = np.asanyarray(arm_mask)
    if modal_value.ndim != 3 or not (
        modal_value.shape == hand_value.shape == arm_value.shape
    ):
        raise ValueError("modal, hand, and arm masks must share (T,H,W)")
    frame_count, height, width = modal_value.shape
    amodal = np.asarray(modal_value, dtype=bool).copy()
    hidden = np.zeros((frame_count, height, width), dtype=bool)
    object_inpaint = np.zeros_like(hidden)
    reports: list[dict[str, object]] = []
    for segment in segments:
        reliable, median_area = _reliable_frames(
            modal_value,
            segment,
            minimum_ratio=reliable_minimum_ratio,
            maximum_ratio=reliable_maximum_ratio,
        )
        reliable_set = set(int(value) for value in reliable)
        reference_frames = select_reference_bank(
            modal_value,
            hand_value,
            reliable,
        )
        reference_bank = [
            (
                frame_index,
                component_convex_hulls(
                    np.asarray(modal_value[frame_index], dtype=bool)
                ),
                np.asarray(hand_value[frame_index], dtype=bool),
            )
            for frame_index in reference_frames
        ]
        temporal_fallbacks = 0
        temporal_reference_frames = 0
        for frame_index in range(segment.start, segment.end + 1):
            visible = np.asarray(modal_value[frame_index], dtype=bool)
            hand = np.asarray(hand_value[frame_index], dtype=bool)
            prior = None
            if frame_index not in reliable_set and len(reliable):
                reference = int(
                    reliable[int(np.argmin(np.abs(reliable - frame_index)))]
                )
                reference_hull = component_convex_hulls(
                    np.asarray(modal_value[reference], dtype=bool)
                )
                prior = warp_mask_by_hand_pose(
                    reference_hull,
                    np.asarray(hand_value[reference], dtype=bool),
                    hand,
                )
                if np.any(prior):
                    temporal_fallbacks += 1
                else:
                    prior = None
            elif reference_bank:
                warped_reference = best_warped_reference_prior(
                    visible,
                    hand,
                    reference_bank,
                    target_frame=frame_index,
                )
                if warped_reference is not None:
                    warped, _reference_frame, _coverage = warped_reference
                    prior = component_convex_hulls(visible) | warped
                    temporal_reference_frames += 1
            amodal_frame, hidden_frame = infer_hidden_support(
                visible,
                hand,
                prior=prior,
                hand_dilate_px=hand_dilate_px,
                max_modal_distance_px=max_modal_distance_px,
            )
            amodal[frame_index] = amodal_frame
            hidden[frame_index] = hidden_frame
            inpaint_region = dilate_mask(
                hidden_frame,
                object_inpaint_dilate_px,
            )
            inpaint_region &= dilate_mask(
                hand,
                hand_dilate_px + object_inpaint_dilate_px,
            )
            inpaint_region &= ~visible
            object_inpaint[frame_index] = inpaint_region
        reports.append(
            {
                "label": segment.label,
                "start": segment.start,
                "end": segment.end,
                "median_modal_area_px": int(round(median_area)),
                "reliable_frames": int(len(reliable)),
                "reference_bank_frames": reference_frames,
                "temporal_fallback_frames": temporal_fallbacks,
                "temporal_reference_frames": temporal_reference_frames,
                "hidden_pixels": int(hidden[segment.start : segment.end + 1].sum()),
            }
        )

    # Remove the union of the SAM arm and HaWoR model hand everywhere except
    # pixels the cleaned modal track owns.  SAM occasionally misses fingers,
    # while its arm component can also include the held object; the two-mask
    # union plus trusted-modal priority resolves both cases.
    hand_removal = np.zeros_like(hidden)
    for frame_index in range(frame_count):
        arm_region = dilate_mask(
            np.asarray(arm_value[frame_index], dtype=bool),
            arm_dilate_px,
        )
        hand_region = dilate_mask(
            np.asarray(hand_value[frame_index], dtype=bool),
            hand_dilate_px,
        )
        hand_removal[frame_index] = (
            arm_region | hand_region
        ) & ~np.asarray(modal_value[frame_index], dtype=bool)
    if np.any(hidden & np.asarray(modal_value, dtype=bool)):
        raise RuntimeError("hidden completion overlaps observed modal object")
    if np.any(object_inpaint & np.asarray(modal_value, dtype=bool)):
        raise RuntimeError("object inpaint mask overlaps observed modal object")
    return amodal, hidden, object_inpaint, hand_removal, reports


def extend_surface_frame(
    surface_depth: np.ndarray,
    modal: np.ndarray,
    hidden: np.ndarray,
) -> np.ndarray:
    """Fill hidden pixels from their nearest valid modal surface sample."""
    surface = np.asarray(surface_depth, dtype=np.float32)
    visible = np.asarray(modal, dtype=bool)
    inferred = np.asarray(hidden, dtype=bool)
    if not (surface.shape == visible.shape == inferred.shape):
        raise ValueError("surface/modal/hidden shapes differ")
    output = surface.copy()
    if not np.any(inferred):
        return output
    valid = visible & np.isfinite(surface) & (surface > 0.02) & (surface < 5.0)
    if not np.any(valid):
        return output
    support = valid | inferred
    rows, columns = np.nonzero(support)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    valid_crop = valid[y0:y1, x0:x1]
    hidden_crop = inferred[y0:y1, x0:x1]
    _distance, indices = distance_transform_edt(
        ~valid_crop,
        return_distances=True,
        return_indices=True,
    )
    source_rows, source_columns = indices
    depth_crop = output[y0:y1, x0:x1]
    nearest = depth_crop[source_rows, source_columns]
    depth_crop[hidden_crop] = nearest[hidden_crop]
    output[y0:y1, x0:x1] = depth_crop
    return output


def temporal_surface_fallbacks(
    surface_depth: np.ndarray,
    modal: np.ndarray,
    segments: Iterable[Segment],
) -> np.ndarray:
    """Interpolate a scalar object camera-Z for frames with no valid surface."""
    surface = np.asanyarray(surface_depth)
    visible = np.asanyarray(modal)
    if surface.ndim != 3 or surface.shape != visible.shape:
        raise ValueError("surface depth and modal mask must share (T,H,W)")
    result = np.full(surface.shape[0], np.nan, dtype=np.float32)
    for segment in segments:
        frame_indices = np.arange(segment.start, segment.end + 1)
        values = np.full(len(frame_indices), np.nan, dtype=np.float64)
        for local_index, frame_index in enumerate(frame_indices):
            depth = np.asarray(surface[frame_index], dtype=np.float32)
            valid = (
                np.asarray(visible[frame_index], dtype=bool)
                & np.isfinite(depth)
                & (depth > 0.02)
                & (depth < 5.0)
            )
            if np.any(valid):
                values[local_index] = float(np.median(depth[valid]))
        valid_indices = np.nonzero(np.isfinite(values))[0]
        if len(valid_indices):
            values = np.interp(
                np.arange(len(values)),
                valid_indices,
                values[valid_indices],
            )
            result[frame_indices] = values.astype(np.float32)
    return result


def constrain_object_candidate(
    raw_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    modal: np.ndarray,
    hidden: np.ndarray,
    *,
    full_candidate_error: float = 12.0,
    zero_candidate_error: float = 45.0,
    minimum_modal_pixels: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject generated hand colours using nearby observed object colours.

    E2FGVI can continue the human-hand texture because it surrounds the hole.
    When a usable modal object is present, generated RGB is kept only while it
    agrees with the nearest observed object pixel.  Collapsed modal tracks fail
    open to the temporal candidate rather than trusting one or two pixels.
    """
    raw = np.asarray(raw_rgb, dtype=np.uint8)
    candidate = np.asarray(candidate_rgb, dtype=np.uint8)
    visible = np.asarray(modal, dtype=bool)
    inferred = np.asarray(hidden, dtype=bool)
    if raw.shape != candidate.shape or raw.shape != visible.shape + (3,):
        raise ValueError("RGB/modal shapes differ")
    if inferred.shape != visible.shape:
        raise ValueError("modal and hidden shapes differ")
    if not (0 <= full_candidate_error < zero_candidate_error):
        raise ValueError("candidate error thresholds are invalid")
    output = candidate.copy()
    weights = np.ones(visible.shape, dtype=np.float32)
    if not np.any(inferred) or int(visible.sum()) < minimum_modal_pixels:
        return output, weights
    support = visible | inferred
    rows, columns = np.nonzero(support)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    visible_crop = visible[y0:y1, x0:x1]
    hidden_crop = inferred[y0:y1, x0:x1]
    _distance, indices = distance_transform_edt(
        ~visible_crop,
        return_distances=True,
        return_indices=True,
    )
    source_rows, source_columns = indices
    raw_crop = raw[y0:y1, x0:x1]
    candidate_crop = candidate[y0:y1, x0:x1]
    nearest = raw_crop[source_rows, source_columns].astype(np.float32)
    error = np.mean(
        np.abs(candidate_crop.astype(np.float32) - nearest),
        axis=2,
    )
    weight = np.clip(
        (zero_candidate_error - error)
        / (zero_candidate_error - full_candidate_error),
        0.0,
        1.0,
    )
    constrained = np.clip(
        weight[..., None] * candidate_crop.astype(np.float32)
        + (1.0 - weight[..., None]) * nearest,
        0,
        255,
    ).astype(np.uint8)
    output_crop = output[y0:y1, x0:x1]
    output_crop[hidden_crop] = constrained[hidden_crop]
    output[y0:y1, x0:x1] = output_crop
    weight_crop = weights[y0:y1, x0:x1]
    weight_crop[hidden_crop] = weight[hidden_crop]
    weights[y0:y1, x0:x1] = weight_crop
    return output, weights


def _read_resized_frames(
    path: Path,
    *,
    output_height: int,
) -> tuple[list[Image.Image], VideoMetadata]:
    metadata = probe_video(path)
    if output_height <= 0 or output_height > metadata.height:
        raise ValueError("inpaint height must be in (0, source height]")
    output_width = int(round(metadata.width * output_height / metadata.height))
    capture = cv2.VideoCapture(str(path))
    frames: list[Image.Image] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (output_height, output_width):
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if len(frames) != metadata.frame_count:
        raise RuntimeError(
            f"video decoded {len(frames)} frames, expected {metadata.frame_count}"
        )
    return frames, metadata


def _resize_masks_for_inpaint(
    masks: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[Image.Image]:
    result: list[Image.Image] = []
    for frame in masks:
        small = cv2.resize(
            np.asarray(frame, dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        result.append(Image.fromarray(small * 255))
    return result


def _open_writer(path: Path, metadata: VideoMetadata) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create video writer: {path}")
    return writer


def _transcode_h264(source: Path, target: Path) -> None:
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ),
        check=True,
    )
    source.unlink(missing_ok=True)


def _paint_completion_evidence(
    frame: np.ndarray,
    hidden: np.ndarray,
    contact_support: np.ndarray | None = None,
) -> np.ndarray:
    output = np.asarray(frame, dtype=np.uint8).copy()
    if contact_support is not None:
        contact = np.asarray(contact_support, dtype=bool)
        if contact.shape != output.shape[:2]:
            raise ValueError("contact support and debug frame shapes differ")
        if np.any(contact):
            colour = np.asarray((0, 255, 255), dtype=np.float32)
            output[contact] = np.clip(
                output[contact].astype(np.float32) * 0.45
                + colour * 0.55,
                0,
                255,
            ).astype(np.uint8)
    mask = np.asarray(hidden, dtype=bool)
    if np.any(mask):
        colour = np.asarray((255, 0, 255), dtype=np.float32)
        output[mask] = np.clip(
            output[mask].astype(np.float32) * 0.35 + colour * 0.65,
            0,
            255,
        ).astype(np.uint8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--modal_mask", type=Path, required=True)
    parser.add_argument("--arm_mask", type=Path, required=True)
    parser.add_argument("--hand_support", type=Path, required=True)
    parser.add_argument("--surface_depth", type=Path, required=True)
    parser.add_argument("--labels_json", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--inpaint_height", type=int, default=360)
    parser.add_argument("--hand_dilate_px", type=int, default=16)
    parser.add_argument("--modal_hand_exclusion_px", type=int, default=16)
    parser.add_argument("--arm_dilate_px", type=int, default=5)
    parser.add_argument("--object_inpaint_dilate_px", type=int, default=8)
    parser.add_argument("--max_modal_distance_px", type=float, default=64.0)
    parser.add_argument("--reliable_minimum_ratio", type=float, default=0.40)
    parser.add_argument("--reliable_maximum_ratio", type=float, default=1.80)
    parser.add_argument("--full_candidate_error", type=float, default=12.0)
    parser.add_argument("--zero_candidate_error", type=float, default=45.0)
    parser.add_argument("--colour_donor_hand_dilate_px", type=int, default=16)
    parser.add_argument("--contact_dir", type=Path)
    parser.add_argument("--aux_contact_dir", type=Path)
    parser.add_argument("--hawor_npz", type=Path)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--aux_side",
        choices=("left", "right"),
        default="left",
    )
    parser.add_argument(
        "--aux_frame_offset",
        type=int,
        default=0,
        help=(
            "Auxiliary-view alignment using aux_index = primary_index + "
            "offset; out-of-range frames use primary evidence only"
        ),
    )
    parser.add_argument("--haco_modal_hand_exclusion_px", type=int, default=4)
    parser.add_argument("--haco_temporal_grace_frames", type=int, default=2)
    parser.add_argument("--haco_contact_score_threshold", type=float, default=0.72)
    parser.add_argument("--haco_contact_point_threshold", type=float, default=0.78)
    parser.add_argument("--haco_contact_top_fraction", type=float, default=0.25)
    parser.add_argument("--haco_min_contact_points", type=int, default=6)
    parser.add_argument("--haco_contact_radius_px", type=int, default=22)
    parser.add_argument("--haco_point_probe_radius_px", type=int, default=4)
    parser.add_argument("--haco_hidden_fraction_on", type=float, default=0.42)
    parser.add_argument("--haco_hidden_fraction_off", type=float, default=0.22)
    parser.add_argument("--haco_min_on_frames", type=int, default=2)
    parser.add_argument("--haco_hold_frames", type=int, default=3)
    args = parser.parse_args()

    paths = {
        "source": args.source.expanduser().resolve(),
        "modal_mask": args.modal_mask.expanduser().resolve(),
        "arm_mask": args.arm_mask.expanduser().resolve(),
        "hand_support": args.hand_support.expanduser().resolve(),
        "surface_depth": args.surface_depth.expanduser().resolve(),
        "labels_json": args.labels_json.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    haco_enabled = args.contact_dir is not None or args.hawor_npz is not None
    if haco_enabled and (args.contact_dir is None or args.hawor_npz is None):
        parser.error("HaCo mode requires both --contact_dir and --hawor_npz")
    if args.aux_contact_dir is not None and not haco_enabled:
        parser.error("--aux_contact_dir requires HaCo mode")
    contact_dir = (
        args.contact_dir.expanduser().resolve()
        if args.contact_dir is not None
        else None
    )
    auxiliary_contact_dir = (
        args.aux_contact_dir.expanduser().resolve()
        if args.aux_contact_dir is not None
        else None
    )
    hawor_npz = (
        args.hawor_npz.expanduser().resolve()
        if args.hawor_npz is not None
        else None
    )
    for directory in (contact_dir, auxiliary_contact_dir):
        if directory is not None and not directory.is_dir():
            raise NotADirectoryError(directory)
    if hawor_npz is not None and not hawor_npz.is_file():
        raise FileNotFoundError(hawor_npz)
    if args.hand_dilate_px < 0 or args.arm_dilate_px < 0:
        parser.error("mask dilation values must be non-negative")
    if args.modal_hand_exclusion_px < 0:
        parser.error("--modal_hand_exclusion_px must be non-negative")
    if args.hand_dilate_px < args.modal_hand_exclusion_px:
        parser.error(
            "--hand_dilate_px must cover --modal_hand_exclusion_px"
        )
    if args.object_inpaint_dilate_px < 0:
        parser.error("--object_inpaint_dilate_px must be non-negative")
    if args.max_modal_distance_px <= 0:
        parser.error("--max_modal_distance_px must be positive")
    if not (
        0 < args.reliable_minimum_ratio < 1.0
        and args.reliable_maximum_ratio > 1.0
    ):
        parser.error("reliable area ratios must straddle 1.0")
    if not (0 <= args.full_candidate_error < args.zero_candidate_error):
        parser.error("candidate colour-error thresholds are invalid")
    if args.colour_donor_hand_dilate_px < 0:
        parser.error("--colour_donor_hand_dilate_px must be non-negative")
    if args.haco_modal_hand_exclusion_px < 0:
        parser.error("--haco_modal_hand_exclusion_px must be non-negative")
    if args.haco_temporal_grace_frames < 0:
        parser.error("--haco_temporal_grace_frames must be non-negative")
    if args.haco_min_contact_points <= 0 or args.haco_contact_radius_px <= 0:
        parser.error("HaCo contact counts/radii must be positive")

    metadata = probe_video(paths["source"])
    modal = np.load(paths["modal_mask"], mmap_mode="r", allow_pickle=False)
    arm = np.load(paths["arm_mask"], mmap_mode="r", allow_pickle=False)
    hand = np.load(paths["hand_support"], mmap_mode="r", allow_pickle=False)
    surface = np.load(paths["surface_depth"], mmap_mode="r", allow_pickle=False)
    expected_shape = (metadata.frame_count, metadata.height, metadata.width)
    for name, value in (("modal", modal), ("arm", arm), ("hand", hand)):
        if value.shape != expected_shape or value.dtype != np.bool_:
            raise ValueError(f"{name} mask must be bool {expected_shape}")
    if surface.shape != expected_shape or not np.issubdtype(
        surface.dtype,
        np.floating,
    ):
        raise ValueError(f"surface depth must be floating {expected_shape}")
    segments = load_segments(paths["labels_json"], metadata.frame_count)

    output_dir = args.out_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".object_completion.", dir=output_dir.parent)
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)

    haco_evidence: dict[str, object] | None = None
    contact_support: np.ndarray | None = None
    direct_haco_frames = np.zeros(metadata.frame_count, dtype=bool)
    temporal_haco_frames = np.zeros(metadata.frame_count, dtype=bool)
    if haco_enabled:
        assert contact_dir is not None and hawor_npz is not None
        print("[1/6] building dual-view HaCo contact evidence", flush=True)
        haco_evidence = build_dual_haco_contact_support(
            contact_dir=contact_dir,
            auxiliary_contact_dir=auxiliary_contact_dir,
            hawor_npz=hawor_npz,
            modal=modal,
            metadata=metadata,
            side=args.side,
            auxiliary_side=args.aux_side,
            auxiliary_frame_offset=args.aux_frame_offset,
            contact_score_threshold=args.haco_contact_score_threshold,
            contact_point_threshold=args.haco_contact_point_threshold,
            contact_top_fraction=args.haco_contact_top_fraction,
            min_contact_points=args.haco_min_contact_points,
            contact_radius_px=args.haco_contact_radius_px,
            point_probe_radius_px=args.haco_point_probe_radius_px,
            hidden_fraction_on=args.haco_hidden_fraction_on,
            hidden_fraction_off=args.haco_hidden_fraction_off,
            min_on_frames=args.haco_min_on_frames,
            hold_frames=args.haco_hold_frames,
        )
        contact_support = np.asarray(haco_evidence["support"], dtype=bool)
        trusted_modal, contested_modal = clean_modal_observations_with_contact(
            modal,
            hand,
            contact_support,
            hand_dilate_px=args.haco_modal_hand_exclusion_px,
        )
    else:
        print(
            "[1/5] cleaning modal observations and building amodal support",
            flush=True,
        )
        trusted_modal, contested_modal = clean_modal_observations(
            modal,
            hand,
            hand_dilate_px=args.modal_hand_exclusion_px,
        )

    if haco_enabled:
        print("[2/6] building HaCo-selected amodal object support", flush=True)
    (
        amodal,
        hidden,
        object_inpaint,
        hand_removal,
        segment_reports,
    ) = build_completion_masks(
        trusted_modal,
        hand,
        arm,
        segments,
        hand_dilate_px=args.hand_dilate_px,
        arm_dilate_px=args.arm_dilate_px,
        object_inpaint_dilate_px=args.object_inpaint_dilate_px,
        max_modal_distance_px=args.max_modal_distance_px,
        reliable_minimum_ratio=args.reliable_minimum_ratio,
        reliable_maximum_ratio=args.reliable_maximum_ratio,
    )
    raw_hidden = hidden
    if haco_enabled:
        assert contact_support is not None
        hidden, direct_haco_frames, temporal_haco_frames = (
            select_haco_hidden_components(
                raw_hidden,
                contact_support,
                segments,
                temporal_grace_frames=args.haco_temporal_grace_frames,
            )
        )
        amodal = trusted_modal | hidden
        object_inpaint = np.zeros_like(hidden)
        for frame_index in range(metadata.frame_count):
            inpaint_region = dilate_mask(
                hidden[frame_index],
                args.object_inpaint_dilate_px,
            )
            inpaint_region &= dilate_mask(
                np.asarray(hand[frame_index], dtype=bool),
                args.hand_dilate_px + args.object_inpaint_dilate_px,
            )
            inpaint_region &= ~trusted_modal[frame_index]
            object_inpaint[frame_index] = inpaint_region
        for segment, segment_report in zip(segments, segment_reports):
            raw_count = int(
                raw_hidden[segment.start : segment.end + 1].sum()
            )
            selected_count = int(
                hidden[segment.start : segment.end + 1].sum()
            )
            segment_report["raw_hidden_pixels"] = raw_count
            segment_report["hidden_pixels"] = selected_count
            segment_report["haco_rejected_hidden_pixels"] = (
                raw_count - selected_count
            )
    hand_removal |= contested_modal
    np.save(staging / "object_mask_observed_clean.npy", trusted_modal)
    np.save(staging / "object_mask_amodal.npy", amodal)
    if haco_enabled:
        assert contact_support is not None and haco_evidence is not None
        np.save(staging / "haco_contact_support.npy", contact_support)
        auxiliary_scores = haco_evidence["auxiliary_scores"]
        if auxiliary_scores is None:
            auxiliary_scores = np.full_like(
                np.asarray(haco_evidence["primary_scores"]),
                np.nan,
            )
        gates = haco_evidence["gates"]
        assert isinstance(gates, dict)
        np.savez_compressed(
            staging / "haco_evidence.npz",
            finger_names=np.asarray(haco_evidence["finger_names"]),
            primary_scores=np.asarray(haco_evidence["primary_scores"]),
            auxiliary_scores=np.asarray(auxiliary_scores),
            auxiliary_frame_indices=np.asarray(
                haco_evidence["auxiliary_frame_indices"],
                dtype=np.int64,
            ),
            fused_scores=np.asarray(haco_evidence["fused_scores"]),
            hidden_fraction=np.asarray(haco_evidence["hidden_fraction"]),
            primary_active=np.asarray(haco_evidence["primary_active"]),
            active=np.asarray(haco_evidence["active"]),
            projected_active=np.asarray(haco_evidence["projected_active"]),
            auxiliary_qualified=np.asarray(gates["auxiliary_qualified"]),
            seed_pixels_by_finger=np.asarray(
                haco_evidence["seed_pixels_by_finger"]
            ),
            direct_selected_frames=direct_haco_frames,
            temporal_fallback_frames=temporal_haco_frames,
        )

    stage_prefix = "3/6" if haco_enabled else "2/5"
    print(
        f"[{stage_prefix}] extending visible object surface into inferred pixels",
        flush=True,
    )
    surface_fallback = temporal_surface_fallbacks(
        surface,
        trusted_modal,
        segments,
    )
    completed_surface = np.lib.format.open_memmap(
        staging / "object_surface_depth_completed.npy",
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )
    missing_depth_pixels = 0
    for frame_index in range(metadata.frame_count):
        extended = extend_surface_frame(
            np.asarray(surface[frame_index], dtype=np.float32),
            trusted_modal[frame_index],
            hidden[frame_index],
        )
        missing = hidden[frame_index] & ~(
            np.isfinite(extended) & (extended > 0.02) & (extended < 5.0)
        )
        if np.any(missing) and np.isfinite(surface_fallback[frame_index]):
            extended[missing] = surface_fallback[frame_index]
        extended[~amodal[frame_index]] = 0.0
        completed_surface[frame_index] = extended.astype(np.float16)
        missing_depth_pixels += int(
            np.sum(hidden[frame_index] & ~(np.isfinite(extended) & (extended > 0.02)))
        )
    completed_surface.flush()
    del completed_surface

    stage_prefix = "4/6" if haco_enabled else "3/5"
    print(
        f"[{stage_prefix}] loading E2FGVI and removing the human arm",
        flush=True,
    )
    frames, decoded_metadata = _read_resized_frames(
        paths["source"],
        output_height=args.inpaint_height,
    )
    if decoded_metadata != metadata:
        raise RuntimeError("source metadata changed while decoding")
    small_width, small_height = frames[0].size
    removal_masks = _resize_masks_for_inpaint(
        hand_removal,
        width=small_width,
        height=small_height,
    )
    object_masks = _resize_masks_for_inpaint(
        object_inpaint,
        width=small_width,
        height=small_height,
    )

    # Delay the heavyweight model import until all geometry inputs have passed
    # validation.  This also keeps pure helper functions unit-testable on CPU.
    sys.path.insert(0, str(Path(__file__).parent))
    from _paths import E2FGVI_CHECKPOINT, ensure_e2fgvi_importable

    ensure_e2fgvi_importable()
    import torch
    from model.e2fgvi_hq import InpaintGenerator
    from inpaint_hands import _inpaint_video

    checkpoint = Path(E2FGVI_CHECKPOINT)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InpaintGenerator().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    baseline_small = _inpaint_video(model, device, frames, removal_masks)

    stage_prefix = "5/6" if haco_enabled else "4/5"
    print(
        f"[{stage_prefix}] completing only the hand-hidden object support",
        flush=True,
    )
    object_small = _inpaint_video(model, device, frames, object_masks)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stage_prefix = "6/6" if haco_enabled else "5/5"
    print(
        f"[{stage_prefix}] composing native-resolution object layers",
        flush=True,
    )
    baseline_temp = staging / ".video_hand_removed_modal_only.mp4v.mp4"
    completed_temp = staging / ".video_object_completed.mp4v.mp4"
    debug_temp = staging / ".debug_object_completion.mp4v.mp4"
    baseline_writer = _open_writer(baseline_temp, metadata)
    completed_writer = _open_writer(completed_temp, metadata)
    debug_writer = _open_writer(debug_temp, metadata)
    capture = cv2.VideoCapture(str(paths["source"]))
    modal_changed = 0
    outside_hidden_changed = 0
    candidate_weight_sum = 0.0
    candidate_weight_pixels = 0
    try:
        for frame_index in range(metadata.frame_count):
            ok, raw_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"source video ended at frame {frame_index}")
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            baseline_rgb = cv2.resize(
                baseline_small[frame_index],
                (metadata.width, metadata.height),
                interpolation=cv2.INTER_LINEAR,
            )
            object_rgb = cv2.resize(
                object_small[frame_index],
                (metadata.width, metadata.height),
                interpolation=cv2.INTER_LINEAR,
            )
            modal_frame = trusted_modal[frame_index]
            hidden_frame = hidden[frame_index]
            baseline_rgb[modal_frame] = raw_rgb[modal_frame]
            completed_rgb = baseline_rgb.copy()
            if np.any(hidden_frame):
                # Shape ownership and colour ownership are deliberately
                # separate.  The modal SAM track has priority for preserving
                # observed RGB, but contact-boundary pixels can include skin.
                # Do not use HaWoR-hand pixels as colour donors for completion.
                colour_donor = modal_frame & ~dilate_mask(
                    np.asarray(hand[frame_index], dtype=bool),
                    args.colour_donor_hand_dilate_px,
                )
                object_rgb, candidate_weights = constrain_object_candidate(
                    raw_rgb,
                    object_rgb,
                    colour_donor,
                    hidden_frame,
                    full_candidate_error=args.full_candidate_error,
                    zero_candidate_error=args.zero_candidate_error,
                )
                candidate_weight_sum += float(
                    candidate_weights[hidden_frame].sum()
                )
                candidate_weight_pixels += int(hidden_frame.sum())
                alpha = cv2.GaussianBlur(
                    hidden_frame.astype(np.float32),
                    (0, 0),
                    sigmaX=1.2,
                    sigmaY=1.2,
                )
                alpha = np.clip(alpha * 1.8, 0.0, 1.0)
                alpha *= hidden_frame.astype(np.float32)
                completed_rgb = np.clip(
                    alpha[..., None] * object_rgb.astype(np.float32)
                    + (1.0 - alpha[..., None]) * completed_rgb.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
            completed_rgb[modal_frame] = raw_rgb[modal_frame]
            modal_changed += int(
                np.count_nonzero(completed_rgb[modal_frame] != raw_rgb[modal_frame])
            )
            outside_hidden = ~hidden_frame
            outside_hidden_changed += int(
                np.count_nonzero(
                    completed_rgb[outside_hidden] != baseline_rgb[outside_hidden]
                )
            )
            baseline_bgr = cv2.cvtColor(baseline_rgb, cv2.COLOR_RGB2BGR)
            completed_bgr = cv2.cvtColor(completed_rgb, cv2.COLOR_RGB2BGR)
            baseline_writer.write(baseline_bgr)
            completed_writer.write(completed_bgr)
            debug_writer.write(
                _paint_completion_evidence(
                    completed_bgr,
                    hidden_frame,
                    (
                        contact_support[frame_index]
                        if contact_support is not None
                        else None
                    ),
                )
            )
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[compose] {frame_index + 1}/{metadata.frame_count}",
                    flush=True,
                )
    finally:
        capture.release()
        baseline_writer.release()
        completed_writer.release()
        debug_writer.release()
    if modal_changed:
        raise RuntimeError(f"completion changed {modal_changed} observed RGB values")
    if outside_hidden_changed:
        raise RuntimeError(
            f"completion changed {outside_hidden_changed} values outside hidden support"
        )
    _transcode_h264(
        baseline_temp,
        staging / "video_hand_removed_modal_only.mp4",
    )
    _transcode_h264(
        completed_temp,
        staging / "video_object_completed.mp4",
    )
    _transcode_h264(
        debug_temp,
        staging / "debug_object_completion.mp4",
    )

    per_frame_hidden = hidden.reshape(metadata.frame_count, -1).sum(axis=1)
    per_frame_raw_hidden = raw_hidden.reshape(metadata.frame_count, -1).sum(axis=1)
    per_frame_inpaint = object_inpaint.reshape(metadata.frame_count, -1).sum(axis=1)
    np.savez_compressed(
        staging / "completion_evidence.npz",
        hidden_pixels_per_frame=per_frame_hidden.astype(np.int64),
        raw_hidden_pixels_per_frame=per_frame_raw_hidden.astype(np.int64),
        object_inpaint_pixels_per_frame=per_frame_inpaint.astype(np.int64),
        segment_labels=np.asarray([segment.label for segment in segments]),
    )
    report_sources = {name: str(path) for name, path in paths.items()}
    if haco_enabled:
        assert contact_dir is not None and hawor_npz is not None
        report_sources.update(
            {
                "contact_dir": str(contact_dir),
                "aux_contact_dir": (
                    str(auxiliary_contact_dir)
                    if auxiliary_contact_dir is not None
                    else None
                ),
                "hawor_npz": str(hawor_npz),
            }
        )
    method = (
        "dual_haco_selected_hand_cleaned_object_constrained_e2fgvi"
        if haco_enabled
        else "hand_cleaned_modal_object_constrained_e2fgvi"
    )
    texture_source = (
        "E2FGVI temporal candidate constrained by nearest trusted MH object "
        "colour; projected MH HaCo contact selects eligible hidden components"
        if haco_enabled
        else "E2FGVI temporal candidate constrained by nearest trusted MH "
        "object colour after HaWoR hand-overlap removal"
    )
    report_config: dict[str, object] = {
        "hand_dilate_px": args.hand_dilate_px,
        "modal_hand_exclusion_px": args.modal_hand_exclusion_px,
        "arm_dilate_px": args.arm_dilate_px,
        "object_inpaint_dilate_px": args.object_inpaint_dilate_px,
        "max_modal_distance_px": args.max_modal_distance_px,
        "reliable_minimum_ratio": args.reliable_minimum_ratio,
        "reliable_maximum_ratio": args.reliable_maximum_ratio,
        "full_candidate_error": args.full_candidate_error,
        "zero_candidate_error": args.zero_candidate_error,
        "colour_donor_excludes_hawor_hand": True,
        "colour_donor_hand_dilate_px": args.colour_donor_hand_dilate_px,
        "missing_surface_policy": (
            "nearest_valid_spatial_then_segmentwise_temporal_median_interpolation"
        ),
    }
    if haco_enabled:
        report_config.update(
            {
                "side": args.side,
                "aux_side": args.aux_side,
                "primary_hawor_focal_px": float(
                    haco_evidence["primary_focal_px"]
                ),
                "aux_frame_offset": args.aux_frame_offset,
                "aux_frame_mapping": "aux_index = primary_index + offset",
                "aux_out_of_range_policy": "primary_evidence_only",
                "haco_modal_hand_exclusion_px": (
                    args.haco_modal_hand_exclusion_px
                ),
                "haco_temporal_grace_frames": (
                    args.haco_temporal_grace_frames
                ),
                "haco_contact_score_threshold": (
                    args.haco_contact_score_threshold
                ),
                "haco_contact_point_threshold": (
                    args.haco_contact_point_threshold
                ),
                "haco_contact_top_fraction": args.haco_contact_top_fraction,
                "haco_min_contact_points": args.haco_min_contact_points,
                "haco_contact_radius_px": args.haco_contact_radius_px,
                "haco_point_probe_radius_px": (
                    args.haco_point_probe_radius_px
                ),
                "haco_hidden_fraction_on": args.haco_hidden_fraction_on,
                "haco_hidden_fraction_off": args.haco_hidden_fraction_off,
                "haco_min_on_frames": args.haco_min_on_frames,
                "haco_hold_frames": args.haco_hold_frames,
                "haco_component_connectivity": 8,
                "haco_role": "contact-connected hidden-component selector only",
                "primary_view_owns_contact_projection": True,
                "auxiliary_view_role": "same-finger confidence only",
            }
        )
    report_counts: dict[str, object] = {
        "input_modal_pixels": int(np.asarray(modal, dtype=bool).sum()),
        "trusted_modal_pixels": int(trusted_modal.sum()),
        "hand_contested_modal_pixels": int(contested_modal.sum()),
        "hidden_completed_pixels": int(hidden.sum()),
        "frames_with_hidden_completion": int((per_frame_hidden > 0).sum()),
        "object_inpaint_pixels": int(object_inpaint.sum()),
        "hand_removal_pixels": int(hand_removal.sum()),
        "hidden_pixels_without_completed_depth": missing_depth_pixels,
        "mean_e2fgvi_candidate_weight": (
            candidate_weight_sum / max(candidate_weight_pixels, 1)
        ),
    }
    if haco_enabled:
        assert contact_support is not None and haco_evidence is not None
        gates = haco_evidence["gates"]
        assert isinstance(gates, dict)
        report_counts.update(
            {
                "raw_hidden_candidate_pixels": int(raw_hidden.sum()),
                "haco_selected_hidden_pixels": int(hidden.sum()),
                "haco_rejected_hidden_pixels": int(
                    raw_hidden.sum() - hidden.sum()
                ),
                "haco_contact_support_pixels": int(contact_support.sum()),
                "haco_contact_support_frames": int(
                    np.any(contact_support.reshape(metadata.frame_count, -1), axis=1).sum()
                ),
                "haco_active_finger_frames": int(
                    np.asarray(haco_evidence["active"], dtype=bool).sum()
                ),
                "haco_projected_active_finger_frames": int(
                    np.asarray(
                        haco_evidence["projected_active"],
                        dtype=bool,
                    ).sum()
                ),
                "auxiliary_qualified_finger_frames": int(
                    np.asarray(gates["auxiliary_qualified"], dtype=bool).sum()
                ),
                "auxiliary_out_of_range_frames": int(
                    np.sum(
                        np.asarray(
                            haco_evidence["auxiliary_frame_indices"],
                            dtype=np.int64,
                        )
                        < 0
                    )
                ),
                "direct_haco_selected_frames": int(direct_haco_frames.sum()),
                "temporal_haco_fallback_frames": int(
                    temporal_haco_frames.sum()
                ),
            }
        )
    report = {
        "schema_version": 1,
        "method": method,
        "object_representation": "inferred_amodal_2d_support_plus_completed_camera_z",
        "texture_source": texture_source,
        "generated_texture": True,
        "physical_geometry_guarantee": False,
        "sources": report_sources,
        "metadata": {
            "frames": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "inpaint_height": args.inpaint_height,
            "inpaint_width": small_width,
        },
        "config": report_config,
        "counts": report_counts,
        "segments": segment_reports,
        "invariants": {
            "trusted_modal_subset_input_modal": bool(
                not np.any(trusted_modal & ~np.asarray(modal, dtype=bool))
            ),
            "trusted_modal_subset_amodal": bool(
                not np.any(trusted_modal & ~amodal)
            ),
            "hand_contested_disjoint_trusted_modal": bool(
                not np.any(contested_modal & trusted_modal)
            ),
            "hidden_disjoint_trusted_modal": bool(
                not np.any(hidden & trusted_modal)
            ),
            "preencode_trusted_modal_rgb_values_changed": modal_changed,
            "preencode_values_changed_outside_hidden": outside_hidden_changed,
            "trusted_modal_rgb_has_priority": True,
            "hand_contested_input_modal_is_not_rgb_protected": True,
            "trajectory_arrays_unchanged": True,
            "haco_selected_hidden_subset_raw_hidden": bool(
                not np.any(hidden & ~raw_hidden)
            ),
            "haco_does_not_measure_object_rgb_or_depth": bool(haco_enabled),
            "primary_view_owns_haco_projection": bool(haco_enabled),
            "auxiliary_haco_is_confidence_only": bool(haco_enabled),
            "auxiliary_geometry_used": False,
        },
        "outputs": {
            "baseline_video": "video_hand_removed_modal_only.mp4",
            "completed_video": "video_object_completed.mp4",
            "debug_video": "debug_object_completion.mp4",
            "clean_modal_mask": "object_mask_observed_clean.npy",
            "amodal_mask": "object_mask_amodal.npy",
            "completed_surface_depth": "object_surface_depth_completed.npy",
            "evidence": "completion_evidence.npz",
            **(
                {
                    "haco_contact_support": "haco_contact_support.npy",
                    "haco_evidence": "haco_evidence.npz",
                }
                if haco_enabled
                else {}
            ),
        },
        "provenance_warning": (
            "Pixels outside the trusted modal mask are inferred, not measured. "
            + (
                "HaCo only selects contact-connected hidden support; it does "
                "not provide object RGB, depth, or front/back direction. "
                if haco_enabled
                else "Input-modal pixels overlapping the HaWoR hand are "
                "deliberately treated as occluded rather than protected RGB. "
            )
            + "The completed depth is nearest-valid camera-Z support for visual "
            "occlusion and is not a watertight object mesh or physical SDF."
        ),
    }
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    publish_directory(str(staging), str(output_dir))
    print(
        f"[ok] object-aware completion: {output_dir} "
        f"hidden={int(hidden.sum())}px/{int((per_frame_hidden > 0).sum())}f",
        flush=True,
    )


if __name__ == "__main__":
    main()
