"""Stereo visibility-conditioned, finger-only RB5 + XHand compositing.

Camera 2 is always the output view.  Camera 1 is used only as independent
visibility evidence: a finger becomes a *behind-object* candidate when the
hand is confidently observed in camera 1 and not observed in camera 2.  A
verified modal object mask limits the decision spatially in camera 2.

Three nested modes are produced in one run:

``visibility``
    Stereo visibility evidence and the camera-2 modal object mask.
``visibility_depth``
    The visibility decision plus camera-2-primary registered object depth;
    camera 1 is accepted only when the two estimates agree.
``visibility_depth_haco``
    Keeps strong stereo+depth decisions and uses the maximum HaCo score from
    the available camera views only to admit or stabilize ambiguous stereo
    evidence.

Opt-in outputs leave the established three-mode comparison unchanged:
``--include_haco_only`` adds max-fused HaCo layering, while supplying both
``--camera1_metric_depth_npz`` and ``--camera2_metric_depth_npz`` adds
``metric_depth_order``.  The latter classifies local hand/object metric depth
with camera 2 authoritative, uses stereo visibility only for ambiguity, and
uses temporally active HaCo only as a contact-finger selector.
``--include_haco_priority`` requires those metric inputs and instead treats
active max-fused HaCo as primary: depth vetoes hiding only when the resolved
order confidently says that the hand is in front.

``--include_ablation_modes`` requires the same native metric inputs and emits
four diagnostic outputs without changing the established comparison video:
an unoccluded baseline, camera-2 metric-depth only, a conservative fixed-panel
2-of-3 vote, and a continuous confidence ensemble.  The ensemble uses HaCo as
contact confidence only; signed depth separation and stereo visibility are
the sole front/back directional cues.  A separate 2x2 ablation comparison is
written so the legacy three-panel layout stays byte-for-byte compatible.

Missing contact evidence always fails open.  Ambiguous depth also fails open
in the established modes, while the explicitly requested HaCo-priority mode
lets active contact win ambiguity by design.  Isaac finger semantics guarantee
that no palm or arm pixel is removed.  PNG and JPEG source-frame directories
are accepted directly, so the synchronized RealSense export does not need to
be transcoded before compositing.

Example::

    python src/inpainting/composite_rb5_stereo_occlusion.py \
      --camera1_rgb_dir /data/case01/camera_1/rgb \
      --camera2_rgb_dir /data/case01/camera_2/rgb \
      --camera1_hawor /work/camera1/rgb_hawor/retarget_input.npz \
      --camera2_hawor /work/camera2/rgb_hawor/retarget_input.npz \
      --camera1_tracks /work/camera1/rgb_hawor/tracks_0_643/model_tracks.npy \
      --camera2_tracks /work/camera2/rgb_hawor/tracks_0_643/model_tracks.npy \
      --camera1_visible_mask /work/camera1/masks_arm.npy \
      --camera2_visible_mask /work/camera2/masks_arm.npy \
      --camera1_contact_dir /work/camera1/contact \
      --contact_dir /work/camera2/contact \
      --background /work/processed/inpaint_processor/video_human_inpaint.mkv \
      --overlay_dir /work/processed/overlay_processor \
      --object_mask /work/processed/object_layer/object_mask_modal.npy \
      --object_restore_mask \
        /work/processed/object_layer/object_mask_observed_clean.npy \
      --scene_depth_camera1 /work/depth/camera1_to_camera2_color.npy \
      --scene_depth /work/depth/camera2_to_camera2_color.npy \
      --out_dir /work/processed/stereo_occlusion
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from atomic_directory_publish import publish_directory
from composite_rb5_contact_occlusion import (
    FINGER_NAMES,
    FINGER_PARTS,
    _open_writer,
    _resize_mask,
    _resize_overlay_frame,
    _true_runs,
    composite_frame,
    estimate_object_depth_track,
    project_camera_points,
    sample_local_fraction,
    suppress_short_runs,
    temporal_hysteresis,
)


MODE_NAMES = (
    "visibility",
    "visibility_depth",
    "visibility_depth_haco",
)
HACO_ONLY_MODE = "haco_only"
VISIBILITY_HACO_MODE = "visibility_haco"
METRIC_DEPTH_ORDER_MODE = "metric_depth_order"
HACO_PRIORITY_MODE = "haco_priority"
NO_OCCLUSION_MODE = "no_occlusion"
CAMERA2_DEPTH_ONLY_MODE = "camera2_depth_only"
VOTE_2OF3_MODE = "vote_2of3"
CONFIDENCE_ENSEMBLE_MODE = "confidence_ensemble"
ABLATION_MODE_NAMES = (
    NO_OCCLUSION_MODE,
    CAMERA2_DEPTH_ONLY_MODE,
    VOTE_2OF3_MODE,
    CONFIDENCE_ENSEMBLE_MODE,
)
DEPTH_ORDER_AMBIGUOUS = np.uint8(0)
DEPTH_ORDER_HAND_FRONT = np.uint8(1)
DEPTH_ORDER_OBJECT_FRONT = np.uint8(2)
DEPTH_ORDER_LABELS = np.asarray(
    ("ambiguous", "hand_front", "object_front"),
)
DEPTH_ORDER_SOURCE_AMBIGUOUS = np.uint8(0)
DEPTH_ORDER_SOURCE_CAMERA2_METRIC = np.uint8(1)
DEPTH_ORDER_SOURCE_STEREO_VISIBILITY = np.uint8(2)
DEPTH_ORDER_SOURCE_STEREO_CONTRADICTION = np.uint8(3)
DEPTH_ORDER_SOURCE_LABELS = np.asarray(
    (
        "ambiguous",
        "camera2_metric",
        "stereo_visibility_assist",
        "stereo_overrides_c2_hand_front",
    ),
)
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
DEPTH_SOURCE_NONE = np.uint8(0)
DEPTH_SOURCE_CAMERA1_UNSUPPORTED = np.uint8(1)
# Backward-compatible constant name; camera-1-only depth is deliberately not
# used for a camera-2 output view because it cannot be cross-view validated.
DEPTH_SOURCE_CAMERA1 = DEPTH_SOURCE_CAMERA1_UNSUPPORTED
DEPTH_SOURCE_CAMERA2 = np.uint8(2)
DEPTH_SOURCE_BOTH = np.uint8(3)
DEPTH_SOURCE_CAMERA2_REJECTED_CAMERA1 = np.uint8(4)
DEPTH_SOURCE_LABELS = np.asarray(
    (
        "none",
        "camera1_unsupported",
        "camera2_only",
        "both_agree_max",
        "camera2_reject_disagreement",
    ),
)


def validate_visibility_haco_inputs(
    *,
    enabled: bool,
    camera1_visible_mask: Path | None,
    camera2_visible_mask: Path | None,
    camera1_contact_dir: Path | None,
) -> None:
    """Require true dual-view, finger-specific evidence for the RGB mode."""

    if not enabled:
        return
    missing = []
    if camera1_visible_mask is None:
        missing.append("--camera1_visible_mask")
    if camera2_visible_mask is None:
        missing.append("--camera2_visible_mask")
    if camera1_contact_dir is None:
        missing.append("--camera1_contact_dir")
    if missing:
        raise ValueError(
            "--include_visibility_haco requires dual-view finger evidence: "
            + ", ".join(missing)
        )


def validate_visibility_haco_coverage(
    *,
    enabled: bool,
    frame_count: int,
    camera1_missing_frames: int,
    camera2_missing_frames: int,
) -> None:
    """Prevent the dual-view mode from silently degrading to one view."""

    if not enabled:
        return
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    missing = {
        "camera1": int(camera1_missing_frames),
        "camera2": int(camera2_missing_frames),
    }
    if any(value < 0 or value > frame_count for value in missing.values()):
        raise ValueError(f"invalid missing-contact counts: {missing}")
    if any(missing.values()):
        raise ValueError(
            "--include_visibility_haco requires complete per-frame HaCo "
            f"coverage in both views; missing={missing}"
        )


def align_camera1_to_camera2(
    values: np.ndarray,
    offset_frames: int,
    *,
    fill_value: float | int | bool,
) -> np.ndarray:
    """Align camera-1 evidence onto the camera-2/output time axis.

    The source lookup is ``camera1_index = camera2_index + offset_frames``.
    For example, ``offset_frames=-1`` maps MH/output frame ``k`` to SH frame
    ``k-1``.  Evidence outside the camera-1 sequence is filled explicitly so
    callers can keep directional decisions fail-open at the boundary.
    """

    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError("camera-1 evidence must have a frame axis")
    frame_count = len(array)
    if frame_count <= 0:
        raise ValueError("camera-1 evidence must contain at least one frame")
    if not isinstance(offset_frames, (int, np.integer)):
        raise TypeError("offset_frames must be an integer")
    offset = int(offset_frames)
    if abs(offset) >= frame_count:
        raise ValueError(
            f"camera-1 offset {offset} is outside a {frame_count}-frame sequence"
        )

    aligned = np.full(array.shape, fill_value, dtype=array.dtype)
    source_start = max(0, offset)
    destination_start = max(0, -offset)
    usable = min(
        frame_count - source_start,
        frame_count - destination_start,
    )
    aligned[destination_start : destination_start + usable] = array[
        source_start : source_start + usable
    ]
    return aligned


def resolve_output_modes(
    *,
    include_haco_only: bool,
    include_haco_priority: bool,
    camera1_metric_depth_npz: Path | None,
    camera2_metric_depth_npz: Path | None,
    include_visibility_haco: bool = False,
    include_ablation_modes: bool = False,
) -> tuple[tuple[str, ...], bool]:
    """Validate opt-in dependencies and return ordered output modes.

    The established modes retain their order.  Supplying both metric files
    keeps the existing implicit ``metric_depth_order`` behavior, and the new
    HaCo-priority output is appended only when explicitly requested.
    """

    metric_depth_paths = (
        camera1_metric_depth_npz,
        camera2_metric_depth_npz,
    )
    supplied_metric_paths = sum(path is not None for path in metric_depth_paths)
    if supplied_metric_paths == 1:
        raise ValueError(
            "--camera1_metric_depth_npz and --camera2_metric_depth_npz "
            "must be supplied together"
        )
    metric_depth_enabled = supplied_metric_paths == 2
    if include_haco_priority and not metric_depth_enabled:
        raise ValueError(
            "--include_haco_priority requires --camera1_metric_depth_npz and "
            "--camera2_metric_depth_npz"
        )
    if include_ablation_modes and not metric_depth_enabled:
        raise ValueError(
            "--include_ablation_modes requires --camera1_metric_depth_npz and "
            "--camera2_metric_depth_npz"
        )
    output_modes = MODE_NAMES + (
        (HACO_ONLY_MODE,) if include_haco_only else ()
    ) + (
        (VISIBILITY_HACO_MODE,) if include_visibility_haco else ()
    ) + ((METRIC_DEPTH_ORDER_MODE,) if metric_depth_enabled else ()) + (
        (HACO_PRIORITY_MODE,) if include_haco_priority else ()
    ) + (ABLATION_MODE_NAMES if include_ablation_modes else ())
    return output_modes, metric_depth_enabled


@dataclass(frozen=True)
class StereoOcclusionConfig:
    """Thresholds shared by the evidence pass and compositor."""

    visibility_on: float = 0.55
    visibility_off: float = 0.28
    assisted_visibility_on: float = 0.30
    assisted_visibility_off: float = 0.15
    visibility_min_on_frames: int = 2
    visibility_hold_frames: int = 3
    haco_on: float = 0.72
    haco_off: float = 0.55
    haco_min_on_frames: int = 2
    haco_hold_frames: int = 3
    haco_top_fraction: float = 0.25
    haco_min_points: int = 6
    visible_probe_radius_px: int = 3
    visible_point_support: float = 0.20
    visible_min_projected_vertices: int = 8
    depth_margin_m: float = 0.030
    depth_agreement_tolerance_m: float = 0.020
    metric_depth_separation_margin_m: float = 0.020
    metric_depth_visibility_assist_threshold: float = 0.55
    metric_depth_min_hand_samples: int = 6
    metric_depth_min_object_samples: int = 20
    object_depth_erode_px: int = 12
    min_occlusion_run_frames: int = 2
    robot_edge_sigma_px: float = 0.6
    occlusion_edge_sigma_px: float = 0.0

    def validate(self) -> None:
        probabilities = (
            self.visibility_on,
            self.visibility_off,
            self.assisted_visibility_on,
            self.assisted_visibility_off,
            self.haco_on,
            self.haco_off,
            self.haco_top_fraction,
            self.visible_point_support,
            self.metric_depth_visibility_assist_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("probability thresholds must be in [0, 1]")
        if self.visibility_off > self.visibility_on:
            raise ValueError("visibility_off must not exceed visibility_on")
        if self.assisted_visibility_off > self.assisted_visibility_on:
            raise ValueError(
                "assisted_visibility_off must not exceed assisted_visibility_on"
            )
        if self.assisted_visibility_on > self.visibility_on:
            raise ValueError(
                "assisted_visibility_on must not exceed strong visibility_on"
            )
        if self.haco_off > self.haco_on:
            raise ValueError("haco_off must not exceed haco_on")
        if self.visibility_min_on_frames <= 0 or self.haco_min_on_frames <= 0:
            raise ValueError("minimum-on frame counts must be positive")
        if self.visibility_hold_frames < 0 or self.haco_hold_frames < 0:
            raise ValueError("hold frame counts must be non-negative")
        if self.haco_min_points <= 0:
            raise ValueError("haco_min_points must be positive")
        if self.visible_probe_radius_px < 0:
            raise ValueError("visible_probe_radius_px must be non-negative")
        if self.visible_min_projected_vertices <= 0:
            raise ValueError("visible_min_projected_vertices must be positive")
        if (
            self.depth_margin_m < 0.0
            or not np.isfinite(self.depth_agreement_tolerance_m)
            or self.depth_agreement_tolerance_m < 0.0
            or not np.isfinite(self.metric_depth_separation_margin_m)
            or self.metric_depth_separation_margin_m < 0.0
            or self.object_depth_erode_px < 0
        ):
            raise ValueError("depth settings must be non-negative")
        if (
            self.metric_depth_min_hand_samples <= 0
            or self.metric_depth_min_object_samples <= 0
        ):
            raise ValueError("metric depth sample minimums must be positive")
        if self.min_occlusion_run_frames <= 0:
            raise ValueError("min_occlusion_run_frames must be positive")
        if self.robot_edge_sigma_px < 0.0 or self.occlusion_edge_sigma_px < 0.0:
            raise ValueError("edge sigmas must be non-negative")


@dataclass(frozen=True)
class MetricFingerDepthEvidence:
    """Validated local metric-depth evidence for one RGB camera view."""

    hand_depth_m_raw: np.ndarray
    object_depth_m_raw: np.ndarray
    hand_sample_count: np.ndarray
    object_sample_count: np.ndarray
    hand_depth_m: np.ndarray
    object_depth_m: np.ndarray


@dataclass(frozen=True)
class AblationConfig:
    """Opt-in confidence-ensemble parameters.

    The directional score is a normalized weighted mean of signed camera-2
    metric-depth separation and positive stereo visibility support.  HaCo is
    then a multiplicative contact-confidence gain, so it cannot create an
    object-front decision when both directional cues are zero.
    """

    confidence_depth_weight: float = 0.65
    confidence_stereo_weight: float = 0.35
    depth_separation_margin_m: float = 0.010
    confidence_depth_saturation_m: float = 0.025
    confidence_stereo_start: float = 0.30
    confidence_stereo_saturation: float = 0.55
    confidence_contact_floor: float = 0.50
    confidence_score_threshold: float = 0.18

    def validate(self, *, depth_deadzone_m: float | None = None) -> None:
        deadzone = (
            self.depth_separation_margin_m
            if depth_deadzone_m is None
            else depth_deadzone_m
        )
        weights = (
            self.confidence_depth_weight,
            self.confidence_stereo_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError(
                "confidence ensemble weights must be finite and non-negative"
            )
        if sum(weights) <= 0.0:
            raise ValueError("at least one confidence ensemble weight must be positive")
        if not np.isfinite(deadzone) or deadzone < 0.0:
            raise ValueError(
                "ablation depth separation margin must be finite and non-negative"
            )
        if (
            not np.isfinite(self.confidence_depth_saturation_m)
            or self.confidence_depth_saturation_m <= deadzone
        ):
            raise ValueError(
                "confidence_depth_saturation_m must exceed the metric-depth "
                "separation margin"
            )
        for name, value in (
            ("confidence_stereo_start", self.confidence_stereo_start),
            ("confidence_contact_floor", self.confidence_contact_floor),
            ("confidence_score_threshold", self.confidence_score_threshold),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if (
            not np.isfinite(self.confidence_stereo_saturation)
            or not 0.0 <= self.confidence_stereo_saturation <= 1.0
            or self.confidence_stereo_saturation
            <= self.confidence_stereo_start
        ):
            raise ValueError(
                "confidence_stereo_saturation must be in [0,1] and exceed "
                "confidence_stereo_start"
            )


@dataclass(frozen=True)
class Vote2Of3Evidence:
    """Per-finger fixed-panel vote diagnostics.

    ``depth_vote_state`` is -1 for explicit camera-2 ``hand_front``, 0 for an
    ambiguous/missing abstention, and +1 for ``object_front``.
    """

    selected: np.ndarray
    positive_count: np.ndarray
    haco_positive: np.ndarray
    depth_vote_state: np.ndarray
    stereo_positive: np.ndarray


@dataclass(frozen=True)
class ConfidenceEnsembleEvidence:
    """Continuous confidence-ensemble diagnostics, all shaped ``(T,5)``."""

    selected: np.ndarray
    score: np.ndarray
    direction_score: np.ndarray
    depth_direction: np.ndarray
    stereo_direction: np.ndarray
    contact_gain: np.ndarray


def load_metric_finger_depth_evidence(
    path: Path,
    *,
    frame_count: int,
    min_hand_samples: int,
    min_object_samples: int,
) -> MetricFingerDepthEvidence:
    """Load and count-gate per-finger local depth estimates from one view.

    Required NPZ keys are ``hand_depth_m``, ``object_depth_m``,
    ``hand_sample_count`` and ``object_sample_count``, all shaped ``(T,5)``.
    Depth is camera-z in metres in that view's native RGB coordinate system.
    An optional ``finger_names`` key must match :data:`FINGER_NAMES`.
    """

    if frame_count <= 0 or min_hand_samples <= 0 or min_object_samples <= 0:
        raise ValueError("frame count and metric-depth sample minimums must be positive")
    required = (
        "hand_depth_m",
        "object_depth_m",
        "hand_sample_count",
        "object_sample_count",
    )
    with np.load(path, allow_pickle=False) as loaded:
        missing = [key for key in required if key not in loaded.files]
        if missing:
            raise ValueError(f"metric depth NPZ is missing keys {missing}: {path}")
        if "finger_names" in loaded.files:
            names = tuple(str(value) for value in loaded["finger_names"].tolist())
            if names != tuple(FINGER_NAMES):
                raise ValueError(
                    f"metric depth finger_names {names} != {tuple(FINGER_NAMES)}"
                )
        hand_raw = np.asarray(loaded["hand_depth_m"], dtype=np.float32)
        object_raw = np.asarray(loaded["object_depth_m"], dtype=np.float32)
        raw_hand_count = np.asarray(loaded["hand_sample_count"])
        raw_object_count = np.asarray(loaded["object_sample_count"])

    expected = (frame_count, len(FINGER_NAMES))
    arrays = {
        "hand_depth_m": hand_raw,
        "object_depth_m": object_raw,
        "hand_sample_count": raw_hand_count,
        "object_sample_count": raw_object_count,
    }
    for name, array in arrays.items():
        if array.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {array.shape}: {path}"
            )
    for name, counts in (
        ("hand_sample_count", raw_hand_count),
        ("object_sample_count", raw_object_count),
    ):
        if not np.issubdtype(counts.dtype, np.integer):
            raise TypeError(f"{name} must use an integer dtype: {path}")
        if np.any(counts < 0):
            raise ValueError(f"{name} must be non-negative: {path}")
    hand_count = raw_hand_count.astype(np.int32, copy=False)
    object_count = raw_object_count.astype(np.int32, copy=False)
    hand = hand_raw.copy()
    foreground = object_raw.copy()
    hand_valid = (
        np.isfinite(hand)
        & (hand > 0.0)
        & (hand_count >= min_hand_samples)
    )
    object_valid = (
        np.isfinite(foreground)
        & (foreground > 0.0)
        & (object_count >= min_object_samples)
    )
    hand[~hand_valid] = np.nan
    foreground[~object_valid] = np.nan
    return MetricFingerDepthEvidence(
        hand_depth_m_raw=hand_raw,
        object_depth_m_raw=object_raw,
        hand_sample_count=hand_count,
        object_sample_count=object_count,
        hand_depth_m=hand,
        object_depth_m=foreground,
    )


def stereo_visibility_evidence(
    camera1_observed: np.ndarray,
    camera2_observed: np.ndarray,
) -> np.ndarray:
    """Return continuous ``C1 visible AND C2 hidden`` evidence.

    Both arrays may be ``(T,)`` or ``(T,F)``.  NaN means an unknown cue, not
    a negative observation; unknown values therefore fail open by producing
    zero behind-object evidence.
    """

    first = np.asarray(camera1_observed, dtype=np.float32)
    second = np.asarray(camera2_observed, dtype=np.float32)
    if first.shape != second.shape or first.ndim not in (1, 2):
        raise ValueError(
            "camera observation arrays must share shape (T,) or (T,F), "
            f"got {first.shape} and {second.shape}"
        )
    known = np.isfinite(first) & np.isfinite(second)
    first = np.clip(np.nan_to_num(first, nan=0.0), 0.0, 1.0)
    # Treat unknown C2 visibility as visible so it cannot remove robot pixels.
    second = np.clip(np.nan_to_num(second, nan=1.0), 0.0, 1.0)
    return np.where(known, first * (1.0 - second), 0.0).astype(np.float32)


def fuse_haco_scores(
    camera1_scores: np.ndarray | None,
    camera2_scores: np.ndarray | None,
) -> np.ndarray:
    """Fuse independent per-finger HaCo scores with a view-wise maximum.

    A missing view has no effect.  Non-finite samples are treated as missing,
    and a sample missing in every supplied view fails open with score zero.
    The one-view case deliberately returns the same values as the legacy
    camera-2-only path.
    """

    supplied = [
        np.asarray(scores, dtype=np.float32)
        for scores in (camera1_scores, camera2_scores)
        if scores is not None
    ]
    if not supplied:
        raise ValueError("at least one HaCo score array is required")
    expected = supplied[0].shape
    if len(expected) != 2 or expected[1] != len(FINGER_NAMES):
        raise ValueError(
            "HaCo scores must have shape (T,F), got "
            f"{expected}"
        )
    if any(scores.shape != expected for scores in supplied[1:]):
        raise ValueError("camera HaCo score arrays must share one shape")

    fused = np.zeros(expected, dtype=np.float32)
    known = np.zeros(expected, dtype=bool)
    for scores in supplied:
        finite = np.isfinite(scores)
        fused = np.where(
            finite & known,
            np.maximum(fused, np.nan_to_num(scores, nan=0.0)),
            np.where(finite, scores, fused),
        )
        known |= finite
    return np.clip(np.where(known, fused, 0.0), 0.0, 1.0).astype(np.float32)


def fuse_object_depth_tracks(
    camera1_depth_m: np.ndarray,
    camera2_depth_m: np.ndarray,
    *,
    agreement_tolerance_m: float = 0.020,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse registered object-depth tracks with camera 2 as the authority.

    Camera-1 reprojection can place disoccluded background samples inside the
    camera-2 modal object mask.  Its farther estimate is therefore accepted
    only when both scalar estimates agree within ``agreement_tolerance_m``.
    An agreeing pair uses the farther (maximum) estimate as a conservative
    depth gate; a disagreement uses camera 2 unchanged.  Camera-1-only depth
    is unsupported for a camera-2 output view and fails open with NaN.

    The returned uint8 source codes index :data:`DEPTH_SOURCE_LABELS`.
    """

    first = np.asarray(camera1_depth_m, dtype=np.float32)
    second = np.asarray(camera2_depth_m, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError(
            "camera object-depth tracks must share shape (T,), got "
            f"{first.shape} and {second.shape}"
        )
    if not np.isfinite(agreement_tolerance_m) or agreement_tolerance_m < 0.0:
        raise ValueError("agreement_tolerance_m must be finite and non-negative")
    first_valid = np.isfinite(first) & (first > 0.0)
    second_valid = np.isfinite(second) & (second > 0.0)
    both = first_valid & second_valid
    agree = both & (
        np.abs(first - second) <= float(agreement_tolerance_m)
    )
    disagree = both & ~agree
    fused = np.full(first.shape, np.nan, dtype=np.float32)
    fused[second_valid & ~first_valid] = second[second_valid & ~first_valid]
    fused[agree] = np.maximum(first[agree], second[agree])
    fused[disagree] = second[disagree]

    source = np.full(first.shape, DEPTH_SOURCE_NONE, dtype=np.uint8)
    source[first_valid & ~second_valid] = DEPTH_SOURCE_CAMERA1_UNSUPPORTED
    source[second_valid & ~first_valid] = DEPTH_SOURCE_CAMERA2
    source[agree] = DEPTH_SOURCE_BOTH
    source[disagree] = DEPTH_SOURCE_CAMERA2_REJECTED_CAMERA1
    return fused, source


def classify_hand_object_depth_order(
    hand_depth_m: np.ndarray,
    object_depth_m: np.ndarray,
    *,
    separation_margin_m: float = 0.020,
) -> np.ndarray:
    """Classify metric hand/object ordering independently in one camera.

    Inputs are corresponding local depth estimates, normally shaped ``(T,5)``
    after sampling each finger neighbourhood.  Smaller camera-z is closer.
    Missing/non-positive estimates and separations inside the uncertainty
    margin remain ambiguous.
    """

    hand = np.asarray(hand_depth_m, dtype=np.float32)
    foreground = np.asarray(object_depth_m, dtype=np.float32)
    if hand.shape != foreground.shape or hand.ndim not in (1, 2):
        raise ValueError(
            "hand/object depth arrays must share shape (T,) or (T,F), got "
            f"{hand.shape} and {foreground.shape}"
        )
    if not np.isfinite(separation_margin_m) or separation_margin_m < 0.0:
        raise ValueError("separation_margin_m must be finite and non-negative")
    known = (
        np.isfinite(hand)
        & np.isfinite(foreground)
        & (hand > 0.0)
        & (foreground > 0.0)
    )
    delta = hand - foreground
    order = np.full(hand.shape, DEPTH_ORDER_AMBIGUOUS, dtype=np.uint8)
    order[known & (delta < -float(separation_margin_m))] = (
        DEPTH_ORDER_HAND_FRONT
    )
    order[known & (delta > float(separation_margin_m))] = (
        DEPTH_ORDER_OBJECT_FRONT
    )
    return order


def resolve_camera2_depth_order(
    camera1_order: np.ndarray,
    camera2_order: np.ndarray,
    stereo_object_front_evidence: np.ndarray,
    *,
    visibility_assist_threshold: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve final-view order with C2 metric depth as the authority.

    C2 ``object_front`` is always authoritative.  C2 ``hand_front`` is also
    authoritative unless C1 independently measures ``hand_front`` while
    strong stereo visibility says the finger is visible in C1 but hidden in
    C2.  That precise contradiction indicates that C2 sampled a visible
    proximal hand patch rather than the hidden finger, so it resolves to
    ``object_front``.  When C2 is already ambiguous, strong C1-visible/C2-
    hidden evidence alone promotes ``object_front``; requiring a C1 metric
    order there would make missing C1 hand depth veto otherwise valid stereo
    evidence.  All other ambiguous cases fail open.

    Returns the order and a uint8 source array indexing
    :data:`DEPTH_ORDER_SOURCE_LABELS`.
    """

    first = np.asarray(camera1_order, dtype=np.uint8)
    second = np.asarray(camera2_order, dtype=np.uint8)
    evidence = np.asarray(stereo_object_front_evidence, dtype=np.float32)
    if first.shape != second.shape or first.shape != evidence.shape:
        raise ValueError("camera depth orders and visibility evidence must align")
    if first.ndim not in (1, 2):
        raise ValueError("depth-order arrays must have shape (T,) or (T,F)")
    if np.any(first > DEPTH_ORDER_OBJECT_FRONT) or np.any(
        second > DEPTH_ORDER_OBJECT_FRONT
    ):
        raise ValueError("depth-order array contains an unknown class ID")
    if not 0.0 <= visibility_assist_threshold <= 1.0:
        raise ValueError("visibility_assist_threshold must be in [0,1]")

    resolved = second.copy()
    source = np.full(
        second.shape,
        DEPTH_ORDER_SOURCE_AMBIGUOUS,
        dtype=np.uint8,
    )
    visibility_support = (
        np.isfinite(evidence)
        & (evidence >= float(visibility_assist_threshold))
    )
    camera2_object_front = second == DEPTH_ORDER_OBJECT_FRONT
    camera2_hand_front = second == DEPTH_ORDER_HAND_FRONT
    contradicted_hand_front = (
        camera2_hand_front
        & (first == DEPTH_ORDER_HAND_FRONT)
        & visibility_support
    )
    camera2_decisive = camera2_object_front | (
        camera2_hand_front & ~contradicted_hand_front
    )
    source[camera2_decisive] = DEPTH_ORDER_SOURCE_CAMERA2_METRIC
    assisted_ambiguity = (
        (second == DEPTH_ORDER_AMBIGUOUS)
        & visibility_support
    )
    resolved[assisted_ambiguity] = DEPTH_ORDER_OBJECT_FRONT
    source[assisted_ambiguity] = DEPTH_ORDER_SOURCE_STEREO_VISIBILITY
    resolved[contradicted_hand_front] = DEPTH_ORDER_OBJECT_FRONT
    source[contradicted_hand_front] = DEPTH_ORDER_SOURCE_STEREO_CONTRADICTION
    return resolved, source


def select_camera2_depth_only_fingers(camera2_depth_order: np.ndarray) -> np.ndarray:
    """Select C2 ``object_front`` fingers without consulting C1 or HaCo."""

    order = np.asarray(camera2_depth_order, dtype=np.uint8)
    if order.ndim not in (1, 2) or order.shape[-1] != len(FINGER_NAMES):
        raise ValueError(
            "camera2_depth_order must have shape (F,) or (T,F), got "
            f"{order.shape}"
        )
    if np.any(order > DEPTH_ORDER_OBJECT_FRONT):
        raise ValueError("camera2_depth_order contains an unknown class ID")
    return order == DEPTH_ORDER_OBJECT_FRONT


def compute_vote_2of3_evidence(
    *,
    haco_active: np.ndarray,
    camera2_depth_order: np.ndarray,
    strong_stereo_active: np.ndarray,
) -> Vote2Of3Evidence:
    """Return a conservative fixed-denominator 2-of-3 decision.

    The three panel members are (1) temporally active, dual-view max-fused
    HaCo, (2) native camera-2 metric order, and (3) temporally active strong
    C1-visible/C2-hidden evidence.  C2 ``object_front`` is +1,
    ``hand_front`` is an explicit negative, and ambiguous/missing metric depth
    abstains.  Abstentions and missing/inactive cues contribute zero but never
    shrink the denominator: selection always requires at least two positive
    votes out of the original three members.  This prevents one remaining cue
    from becoming a majority merely because another cue is unavailable.
    """

    contact = np.asarray(haco_active, dtype=bool)
    order = np.asarray(camera2_depth_order, dtype=np.uint8)
    stereo = np.asarray(strong_stereo_active, dtype=bool)
    if contact.shape != order.shape or contact.shape != stereo.shape:
        raise ValueError("2-of-3 evidence arrays must share one shape")
    if contact.ndim not in (1, 2) or contact.shape[-1] != len(FINGER_NAMES):
        raise ValueError(
            "2-of-3 evidence must have shape (F,) or (T,F), got "
            f"{contact.shape}"
        )
    if np.any(order > DEPTH_ORDER_OBJECT_FRONT):
        raise ValueError("camera2_depth_order contains an unknown class ID")

    depth_state = np.zeros(order.shape, dtype=np.int8)
    depth_state[order == DEPTH_ORDER_HAND_FRONT] = -1
    depth_state[order == DEPTH_ORDER_OBJECT_FRONT] = 1
    positive_count = (
        contact.astype(np.uint8)
        + (depth_state > 0).astype(np.uint8)
        + stereo.astype(np.uint8)
    )
    return Vote2Of3Evidence(
        selected=positive_count >= 2,
        positive_count=positive_count,
        haco_positive=contact,
        depth_vote_state=depth_state,
        stereo_positive=stereo,
    )


def compute_confidence_ensemble_evidence(
    *,
    camera2_hand_depth_m: np.ndarray,
    camera2_object_depth_m: np.ndarray,
    stereo_visibility: np.ndarray,
    haco_confidence: np.ndarray,
    depth_deadzone_m: float,
    config: AblationConfig,
) -> ConfidenceEnsembleEvidence:
    """Fuse directional evidence while keeping HaCo direction-neutral.

    Let ``delta = hand_z - object_z`` (positive means the object is in front).
    The signed depth direction is zero inside ``depth_deadzone_m`` and ramps
    linearly to +/-1 at ``confidence_depth_saturation_m``.  Stereo visibility
    is a positive-only directional cue which ramps from zero at
    ``confidence_stereo_start`` to one at
    ``confidence_stereo_saturation``.  Their configured
    weights are normalized before averaging::

        direction = (w_d * depth_direction + w_s * stereo_direction)
                    / (w_d + w_s)
        contact_gain = contact_floor + (1 - contact_floor) * HaCo
        score = max(direction, 0) * contact_gain

    Selection is ``score >= confidence_score_threshold``.  Missing directional
    inputs map to zero; missing HaCo maps to the contact floor.  Consequently,
    even HaCo=1 cannot assert object-front when depth and stereo provide no
    positive direction.
    """

    config.validate(depth_deadzone_m=depth_deadzone_m)
    hand = np.asarray(camera2_hand_depth_m, dtype=np.float32)
    foreground = np.asarray(camera2_object_depth_m, dtype=np.float32)
    stereo = np.asarray(stereo_visibility, dtype=np.float32)
    contact = np.asarray(haco_confidence, dtype=np.float32)
    if not (
        hand.shape == foreground.shape == stereo.shape == contact.shape
    ):
        raise ValueError("confidence ensemble evidence arrays must share one shape")
    if hand.ndim not in (1, 2) or hand.shape[-1] != len(FINGER_NAMES):
        raise ValueError(
            "confidence ensemble evidence must have shape (F,) or (T,F), got "
            f"{hand.shape}"
        )
    if not np.isfinite(depth_deadzone_m) or depth_deadzone_m < 0.0:
        raise ValueError("depth_deadzone_m must be finite and non-negative")

    valid_depth = (
        np.isfinite(hand)
        & np.isfinite(foreground)
        & (hand > 0.0)
        & (foreground > 0.0)
    )
    delta = hand - foreground
    depth_denominator = (
        config.confidence_depth_saturation_m - float(depth_deadzone_m)
    )
    depth_magnitude = np.clip(
        (np.abs(delta) - float(depth_deadzone_m)) / depth_denominator,
        0.0,
        1.0,
    )
    depth_direction = np.where(
        valid_depth,
        np.sign(delta) * depth_magnitude,
        0.0,
    ).astype(np.float32)

    stereo_known = np.isfinite(stereo)
    stereo_clipped = np.clip(np.nan_to_num(stereo, nan=0.0), 0.0, 1.0)
    stereo_direction = np.clip(
        (stereo_clipped - config.confidence_stereo_start)
        / (
            config.confidence_stereo_saturation
            - config.confidence_stereo_start
        ),
        0.0,
        1.0,
    )
    stereo_direction = np.where(
        stereo_known,
        stereo_direction,
        0.0,
    ).astype(np.float32)

    weight_sum = (
        config.confidence_depth_weight + config.confidence_stereo_weight
    )
    direction = (
        config.confidence_depth_weight * depth_direction
        + config.confidence_stereo_weight * stereo_direction
    ) / weight_sum
    direction = np.clip(direction, -1.0, 1.0).astype(np.float32)
    contact_clipped = np.clip(
        np.nan_to_num(contact, nan=0.0),
        0.0,
        1.0,
    )
    contact_gain = (
        config.confidence_contact_floor
        + (1.0 - config.confidence_contact_floor) * contact_clipped
    ).astype(np.float32)
    score = (np.maximum(direction, 0.0) * contact_gain).astype(np.float32)
    return ConfidenceEnsembleEvidence(
        selected=score >= config.confidence_score_threshold,
        score=score,
        direction_score=direction,
        depth_direction=depth_direction,
        stereo_direction=stereo_direction,
        contact_gain=contact_gain,
    )


def temporal_mode_decisions(
    camera1_observed: np.ndarray,
    camera2_observed: np.ndarray,
    haco_scores: np.ndarray,
    config: StereoOcclusionConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Build stable per-finger tracks with visibility primary, HaCo secondary.

    The visibility/depth baselines use the lower assisted threshold so their
    videos expose ambiguous stereo candidates.  The third mode always keeps
    strong stereo evidence, even if HaCo misses, and requires HaCo only for
    candidates between the assisted and strong visibility thresholds.
    """

    config.validate()
    first = np.asarray(camera1_observed, dtype=np.float32)
    second = np.asarray(camera2_observed, dtype=np.float32)
    contact = np.asarray(haco_scores, dtype=np.float32)
    expected_rank = 2
    if (
        first.shape != second.shape
        or first.shape != contact.shape
        or first.ndim != expected_rank
    ):
        raise ValueError(
            "camera visibility and HaCo arrays must share shape (T,F), "
            f"got {first.shape}, {second.shape}, {contact.shape}"
        )
    if first.shape[1] != len(FINGER_NAMES):
        raise ValueError(
            f"expected {len(FINGER_NAMES)} fingers, got {first.shape[1]}"
        )

    visibility = stereo_visibility_evidence(first, second)
    visibility_active = np.zeros_like(visibility, dtype=bool)
    strong_visibility_active = np.zeros_like(visibility, dtype=bool)
    haco_active = np.zeros_like(visibility, dtype=bool)
    for finger_index in range(len(FINGER_NAMES)):
        visibility_active[:, finger_index] = temporal_hysteresis(
            visibility[:, finger_index],
            on_threshold=config.assisted_visibility_on,
            off_threshold=config.assisted_visibility_off,
            min_on_frames=config.visibility_min_on_frames,
            hold_frames=config.visibility_hold_frames,
        )
        strong_visibility_active[:, finger_index] = temporal_hysteresis(
            visibility[:, finger_index],
            on_threshold=config.visibility_on,
            off_threshold=config.visibility_off,
            min_on_frames=config.visibility_min_on_frames,
            hold_frames=config.visibility_hold_frames,
        )
        haco_active[:, finger_index] = temporal_hysteresis(
            np.clip(
                np.nan_to_num(contact[:, finger_index], nan=0.0),
                0.0,
                1.0,
            ),
            on_threshold=config.haco_on,
            off_threshold=config.haco_off,
            min_on_frames=config.haco_min_on_frames,
            hold_frames=config.haco_hold_frames,
        )

    active = {
        "visibility": visibility_active,
        "visibility_depth": visibility_active.copy(),
        "visibility_depth_haco": (
            strong_visibility_active
            | (visibility_active & haco_active)
        ),
    }
    return active, visibility, strong_visibility_active, haco_active


def compute_mode_occlusion_masks(
    *,
    robot_mask: np.ndarray,
    finger_labels: np.ndarray,
    robot_depth: np.ndarray,
    object_mask: np.ndarray,
    active_fingers: dict[str, np.ndarray],
    object_depth_m: float | None,
    depth_margin_m: float,
) -> dict[str, np.ndarray]:
    """Compute nested, finger-only pixel decisions for one camera-2 frame."""

    robot = np.asarray(robot_mask, dtype=bool)
    labels = np.asarray(finger_labels, dtype=np.uint8)
    depth = np.asarray(robot_depth, dtype=np.float32)
    foreground = np.asarray(object_mask, dtype=bool)
    if not (robot.shape == labels.shape == depth.shape == foreground.shape):
        raise ValueError("pixelwise compositor inputs must share one shape")
    if depth_margin_m < 0.0:
        raise ValueError("depth_margin_m must be non-negative")
    if np.any(labels > len(FINGER_NAMES)):
        raise ValueError("finger_labels contains an unknown semantic ID")
    missing_modes = set(MODE_NAMES) - set(active_fingers)
    if missing_modes:
        raise ValueError(f"missing active-finger modes: {sorted(missing_modes)}")

    active_lookup: dict[str, np.ndarray] = {}
    for mode in MODE_NAMES:
        values = np.asarray(active_fingers[mode], dtype=bool)
        if values.shape != (len(FINGER_NAMES),):
            raise ValueError(
                f"{mode} active fingers must have shape "
                f"({len(FINGER_NAMES)},), got {values.shape}"
            )
        lookup = np.r_[False, values]
        active_lookup[mode] = lookup[labels]

    base = robot & (labels > 0) & foreground
    visibility = base & active_lookup["visibility"]
    if object_depth_m is None or not np.isfinite(object_depth_m):
        depth_gate = np.zeros_like(robot)
    else:
        depth_gate = (
            np.isfinite(depth)
            & (depth > float(object_depth_m) + float(depth_margin_m))
        )
    visibility_depth = (
        base
        & active_lookup["visibility_depth"]
        & depth_gate
    )
    visibility_depth_haco = (
        visibility_depth
        & active_lookup["visibility_depth_haco"]
    )
    return {
        "visibility": visibility,
        "visibility_depth": visibility_depth,
        "visibility_depth_haco": visibility_depth_haco,
    }


def compute_selected_finger_occlusion_mask(
    *,
    robot_mask: np.ndarray,
    finger_labels: np.ndarray,
    object_mask: np.ndarray,
    selected_fingers: np.ndarray,
) -> np.ndarray:
    """Apply a per-finger decision within C2 object/robot semantic support."""

    robot = np.asarray(robot_mask, dtype=bool)
    labels = np.asarray(finger_labels, dtype=np.uint8)
    foreground = np.asarray(object_mask, dtype=bool)
    selected = np.asarray(selected_fingers, dtype=bool)
    if not (robot.shape == labels.shape == foreground.shape):
        raise ValueError("selected-finger pixel inputs must share one shape")
    if selected.shape != (len(FINGER_NAMES),):
        raise ValueError(
            "selected_fingers must have shape "
            f"({len(FINGER_NAMES)},), got {selected.shape}"
        )
    if np.any(labels > len(FINGER_NAMES)):
        raise ValueError("finger_labels contains an unknown semantic ID")
    selected_at_pixel = np.r_[False, selected][labels]
    return robot & (labels > 0) & foreground & selected_at_pixel


def compute_no_occlusion_mask(robot_mask: np.ndarray) -> np.ndarray:
    """Return the explicit no-occlusion baseline mask."""

    robot = np.asarray(robot_mask, dtype=bool)
    if robot.ndim != 2:
        raise ValueError(f"robot_mask must be 2D, got {robot.shape}")
    return np.zeros_like(robot, dtype=bool)


def compute_haco_only_occlusion_mask(
    *,
    robot_mask: np.ndarray,
    finger_labels: np.ndarray,
    object_mask: np.ndarray,
    haco_active_fingers: np.ndarray,
) -> np.ndarray:
    """Hide active HaCo fingers inside the object mask, with no other cue.

    This intentionally accepts neither visibility nor depth inputs.  The
    semantic finger labels ensure that HaCo-only layering cannot remove palm
    or arm pixels even if their rendered geometry overlaps the object.
    """

    return compute_selected_finger_occlusion_mask(
        robot_mask=robot_mask,
        finger_labels=finger_labels,
        object_mask=object_mask,
        selected_fingers=haco_active_fingers,
    )


def compute_visibility_haco_occlusion_mask(
    *,
    robot_mask: np.ndarray,
    finger_labels: np.ndarray,
    object_mask: np.ndarray,
    visibility_haco_active_fingers: np.ndarray,
) -> np.ndarray:
    """Hide MH robot fingers selected by directional stereo + dual HaCo.

    The selector is the existing temporally stable decision
    ``strong_stereo OR (assisted_stereo AND fused_HaCo)``.  Unlike the
    historical ``visibility_depth_haco`` output, this opt-in mode deliberately
    has no metric-depth gate, so SH visibility can provide direction when the
    dataset has RGB cameras only.  Pixel removal remains restricted to the MH
    modal object mask and rendered semantic fingers.
    """

    return compute_selected_finger_occlusion_mask(
        robot_mask=robot_mask,
        finger_labels=finger_labels,
        object_mask=object_mask,
        selected_fingers=visibility_haco_active_fingers,
    )


def compute_depth_order_occlusion_mask(
    *,
    robot_mask: np.ndarray,
    finger_labels: np.ndarray,
    object_mask: np.ndarray,
    camera2_depth_order: np.ndarray,
    haco_active_fingers: np.ndarray,
) -> np.ndarray:
    """Hide only contact-selected fingers classified behind the C2 object.

    HaCo is strictly a selector here: it cannot create or change a depth-order
    class.  ``hand_front`` and ``ambiguous`` stay visible, so uncertain metric
    evidence fails open.  The shared semantic helper preserves the finger-only
    invariant.
    """

    order = np.asarray(camera2_depth_order, dtype=np.uint8)
    contact = np.asarray(haco_active_fingers, dtype=bool)
    expected = (len(FINGER_NAMES),)
    if order.shape != expected or contact.shape != expected:
        raise ValueError(
            f"depth order and HaCo selector must have shape {expected}"
        )
    if np.any(order > DEPTH_ORDER_OBJECT_FRONT):
        raise ValueError("camera2_depth_order contains an unknown class ID")
    selected_object_front = contact & (order == DEPTH_ORDER_OBJECT_FRONT)
    return compute_haco_only_occlusion_mask(
        robot_mask=robot_mask,
        finger_labels=finger_labels,
        object_mask=object_mask,
        haco_active_fingers=selected_object_front,
    )


def select_haco_priority_fingers(
    resolved_depth_order: np.ndarray,
    haco_active_fingers: np.ndarray,
) -> np.ndarray:
    """Select active-contact fingers unless metric order proves hand-front.

    This is intentionally asymmetric.  Fused, temporally stable HaCo is the
    primary cue, so both ``object_front`` and ``ambiguous`` depth allow an
    active finger to be hidden.  Only a confidently resolved ``hand_front``
    class vetoes HaCo.  Missing HaCo is represented by an inactive selector
    and therefore still fails open.
    """

    order = np.asarray(resolved_depth_order, dtype=np.uint8)
    contact = np.asarray(haco_active_fingers, dtype=bool)
    expected = (len(FINGER_NAMES),)
    if order.shape != expected or contact.shape != expected:
        raise ValueError(
            f"resolved depth order and HaCo selector must have shape {expected}"
        )
    if np.any(order > DEPTH_ORDER_OBJECT_FRONT):
        raise ValueError("resolved_depth_order contains an unknown class ID")
    return contact & (order != DEPTH_ORDER_HAND_FRONT)


def compute_haco_priority_occlusion_mask(
    *,
    robot_mask: np.ndarray,
    finger_labels: np.ndarray,
    object_mask: np.ndarray,
    resolved_depth_order: np.ndarray,
    haco_active_fingers: np.ndarray,
) -> np.ndarray:
    """Hide HaCo-active semantic fingers unless depth proves hand-front.

    Pixel removal remains limited to the camera-2 modal object mask and the
    rendered robot's semantic finger labels.  Thus neither an ambiguous depth
    sample nor active HaCo can remove palm or arm geometry.
    """

    selected = select_haco_priority_fingers(
        resolved_depth_order,
        haco_active_fingers,
    )
    return compute_haco_only_occlusion_mask(
        robot_mask=robot_mask,
        finger_labels=finger_labels,
        object_mask=object_mask,
        haco_active_fingers=selected,
    )


def _natural_key(path: Path) -> list[int | str]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def discover_images(directory: Path) -> list[Path]:
    """Discover PNG/JPEG frames using natural filename order."""

    if not directory.is_dir():
        raise NotADirectoryError(directory)
    images = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )
    if not images:
        raise FileNotFoundError(f"no PNG/JPEG frames in {directory}")
    return images


def resolve_model_tracks(path: Path | None, hawor_npz: Path) -> Path | None:
    """Resolve a ``model_tracks.npy`` file from a file or HaWoR directory."""

    roots = [path] if path is not None else [hawor_npz.parent]
    matches: list[Path] = []
    for root in roots:
        if root is None:
            continue
        root = root.resolve()
        if root.is_file():
            if root.name != "model_tracks.npy":
                raise ValueError(f"expected model_tracks.npy, got {root}")
            matches.append(root)
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)
        direct = root / "model_tracks.npy"
        if direct.is_file():
            matches.append(direct)
        matches.extend(sorted(root.glob("tracks_*/model_tracks.npy")))
    unique = list(dict.fromkeys(candidate.resolve() for candidate in matches))
    if not unique:
        return None
    if len(unique) != 1:
        raise ValueError(f"expected one model_tracks.npy, got {unique}")
    return unique[0]


def _side_index(side: str) -> int:
    if side not in {"left", "right"}:
        raise ValueError(f"invalid hand side {side!r}")
    return 0 if side == "left" else 1


def load_hawor_valid(
    hawor_path: Path,
    *,
    side: str,
    frame_count: int,
) -> np.ndarray:
    """Load the side-specific HaWoR validity track."""

    with np.load(hawor_path) as hawor:
        if "valid" not in hawor.files:
            raise ValueError(f"HaWoR file is missing valid: {hawor_path}")
        valid = np.asarray(hawor["valid"], dtype=bool)
        if valid.shape == (2, frame_count):
            result = valid[_side_index(side)]
        elif valid.shape == (frame_count, 2):
            result = valid[:, _side_index(side)]
        else:
            raise ValueError(
                f"invalid HaWoR valid shape {valid.shape}; expected "
                f"(2,{frame_count}) or ({frame_count},2)"
            )
        verts_key = f"verts_{side}"
        if verts_key not in hawor.files or hawor[verts_key].shape[:2] != (
            frame_count,
            778,
        ):
            raise ValueError(
                f"invalid or missing {verts_key} in {hawor_path}"
            )
    return result.astype(np.float32)


def load_track_observation_confidence(
    track_path: Path,
    *,
    side: str,
    frame_count: int,
) -> np.ndarray:
    """Convert HaWoR detector tracks to a dense observed-confidence track."""

    loaded = np.load(track_path, allow_pickle=True)
    try:
        tracks = loaded.item()
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid HaWoR model tracks: {track_path}") from exc
    if not isinstance(tracks, dict):
        raise ValueError(f"model tracks must contain a dict: {track_path}")

    desired = _side_index(side)
    confidence = np.zeros(frame_count, dtype=np.float32)
    for entries_value in tracks.values():
        entries = list(entries_value)
        handedness: list[int] = []
        for entry in entries:
            if not isinstance(entry, dict) or not bool(entry.get("det", False)):
                continue
            raw = np.asarray(entry.get("det_handedness", []), dtype=np.float32)
            if raw.size:
                handedness.append(int(float(raw.reshape(-1)[0]) >= 0.5))
        if not handedness:
            continue
        track_side = int(np.mean(handedness) >= 0.5)
        if track_side != desired:
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not bool(entry.get("det", False)):
                continue
            frame_index = int(entry.get("frame", -1))
            if frame_index < 0 or frame_index >= frame_count:
                continue
            box = np.asarray(entry.get("det_box", []), dtype=np.float32)
            score = 1.0
            if box.size and box.shape[-1] >= 5:
                score = float(box.reshape(-1, box.shape[-1])[0, 4])
            if not np.isfinite(score):
                continue
            confidence[frame_index] = max(
                confidence[frame_index],
                float(np.clip(score, 0.0, 1.0)),
            )
    return confidence


def build_observation_confidence(
    *,
    hawor_path: Path,
    track_path: Path | None,
    side: str,
    frame_count: int,
) -> tuple[np.ndarray, str]:
    """Prefer detector observations, falling back to non-infilled validity."""

    hawor_valid = load_hawor_valid(
        hawor_path,
        side=side,
        frame_count=frame_count,
    )
    if track_path is None:
        return hawor_valid, "retarget_valid_fallback"
    observed = load_track_observation_confidence(
        track_path,
        side=side,
        frame_count=frame_count,
    )
    # A detector observation without a usable HaWoR pose is too ambiguous for
    # finger-specific layering.  This intersection remains fail-open.
    return observed * hawor_valid, "model_tracks_confidence_and_hawor_valid"


def projected_finger_visible_fractions(
    *,
    vertices_camera: np.ndarray,
    valid_frames: np.ndarray,
    visible_masks: np.ndarray,
    finger_parts: np.ndarray,
    focal_px: float,
    image_width: int,
    image_height: int,
    probe_radius_px: int,
    point_support_threshold: float,
    min_projected_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure modal visibility independently for each MANO finger.

    A projected vertex counts as observed when the local SAM mask fraction in
    a small probe window reaches ``point_support_threshold``.  Fractions stay
    NaN when HaWoR geometry is invalid or too few vertices project into the
    image; callers must treat NaN as an unknown cue and fail open.

    The function is pure: all geometry and masks are supplied as arrays and
    it performs no file-system access.
    """

    vertices = np.asarray(vertices_camera, dtype=np.float32)
    valid = np.asarray(valid_frames, dtype=bool)
    masks = np.asarray(visible_masks)
    parts = np.asarray(finger_parts, dtype=np.int32)
    if vertices.ndim != 3 or vertices.shape[1:] != (778, 3):
        raise ValueError(
            f"vertices_camera must have shape (T,778,3), got {vertices.shape}"
        )
    frame_count = len(vertices)
    if valid.shape != (frame_count,):
        raise ValueError(f"valid_frames must have shape ({frame_count},)")
    if masks.ndim != 3 or len(masks) != frame_count:
        raise ValueError(
            f"visible_masks must have shape ({frame_count},H,W), got {masks.shape}"
        )
    if parts.shape != (778,):
        raise ValueError(f"finger_parts must have shape (778,), got {parts.shape}")
    if not np.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError("focal_px must be finite and positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if probe_radius_px < 0 or min_projected_vertices <= 0:
        raise ValueError("invalid projected-visibility sampling settings")
    if not 0.0 <= point_support_threshold <= 1.0:
        raise ValueError("point_support_threshold must be in [0,1]")
    mask_height, mask_width = masks.shape[1:]
    if not np.isclose(
        mask_width / mask_height,
        image_width / image_height,
        atol=1e-3,
    ):
        raise ValueError(
            "visible mask/source RGB aspect mismatch: "
            f"{mask_width}x{mask_height} vs {image_width}x{image_height}"
        )

    fractions = np.full(
        (frame_count, len(FINGER_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    projected_counts = np.zeros_like(fractions, dtype=np.int32)
    for frame_index in range(frame_count):
        if not valid[frame_index]:
            continue
        frame_mask = np.asarray(masks[frame_index], dtype=np.uint8)
        if frame_mask.shape != (image_height, image_width):
            frame_mask = cv2.resize(
                frame_mask,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        frame_mask = frame_mask > 0
        frame_vertices = vertices[frame_index]
        for finger_index, finger in enumerate(FINGER_NAMES):
            selected_vertices = frame_vertices[
                np.isin(parts, FINGER_PARTS[finger])
            ]
            points_uv, valid_depth = project_camera_points(
                selected_vertices,
                focal_px=float(focal_px),
                image_width=image_width,
                image_height=image_height,
            )
            in_frame = (
                valid_depth
                & (points_uv[:, 0] >= 0.0)
                & (points_uv[:, 0] < image_width)
                & (points_uv[:, 1] >= 0.0)
                & (points_uv[:, 1] < image_height)
            )
            points_uv = points_uv[in_frame]
            projected_counts[frame_index, finger_index] = len(points_uv)
            if len(points_uv) < min_projected_vertices:
                continue
            local_support = sample_local_fraction(
                frame_mask,
                points_uv,
                probe_radius_px,
            )
            fractions[frame_index, finger_index] = float(
                (local_support >= point_support_threshold).mean()
            )
    return fractions, projected_counts


def fuse_visible_fraction_with_detector(
    detector_confidence: np.ndarray,
    visible_fraction: np.ndarray,
) -> np.ndarray:
    """Fuse SAM modal visibility with conservative HaWoR confidence.

    Known SAM fractions are confidence-weighted.  An unknown projected
    fraction stays unknown: whole-hand detector confidence is not
    finger-specific and therefore cannot safely say that one finger was
    hidden.  ``stereo_visibility_evidence`` treats that NaN as fail-open.
    """

    detector = np.asarray(detector_confidence, dtype=np.float32)
    fraction = np.asarray(visible_fraction, dtype=np.float32)
    if detector.ndim != 1 or fraction.ndim != 2:
        raise ValueError("detector must be (T,) and visible_fraction must be (T,F)")
    if fraction.shape != (len(detector), len(FINGER_NAMES)):
        raise ValueError(
            "visible_fraction must have shape "
            f"({len(detector)},{len(FINGER_NAMES)}), got {fraction.shape}"
        )
    detector = np.clip(np.nan_to_num(detector, nan=0.0), 0.0, 1.0)
    confidence = np.repeat(detector[:, None], len(FINGER_NAMES), axis=1)
    known = np.isfinite(fraction)
    weighted = confidence * np.clip(np.nan_to_num(fraction, nan=0.0), 0.0, 1.0)
    return np.where(known, weighted, np.nan).astype(np.float32)


def load_projected_visible_fractions(
    *,
    hawor_path: Path,
    visible_mask_path: Path,
    side: str,
    frame_count: int,
    finger_parts: np.ndarray,
    image_width: int,
    image_height: int,
    config: StereoOcclusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load HaWoR geometry/SAM masks and compute per-finger visibility."""

    visible_masks = np.load(visible_mask_path, mmap_mode="r")
    if visible_masks.ndim != 3 or len(visible_masks) != frame_count:
        raise ValueError(
            f"visible mask must have shape ({frame_count},H,W), "
            f"got {visible_masks.shape}"
        )
    with np.load(hawor_path) as hawor:
        vertices = np.asarray(hawor[f"verts_{side}"], dtype=np.float32)
        valid_raw = np.asarray(hawor["valid"], dtype=bool)
        if valid_raw.shape == (2, frame_count):
            valid = valid_raw[_side_index(side)]
        elif valid_raw.shape == (frame_count, 2):
            valid = valid_raw[:, _side_index(side)]
        else:
            raise ValueError(f"invalid HaWoR valid shape: {valid_raw.shape}")
        frame_is_camera = bool(np.asarray(hawor["frame_is_cam_space"]).item())
        if frame_is_camera:
            vertices_camera = vertices
        else:
            if "R_c2w" not in hawor.files or "t_c2w" not in hawor.files:
                raise ValueError(
                    "world-space HaWoR visibility projection requires R_c2w/t_c2w; "
                    "rerun HaWoR with --skip_slam for direct camera-space geometry"
                )
            rotation = np.asarray(hawor["R_c2w"], dtype=np.float32)
            translation = np.asarray(hawor["t_c2w"], dtype=np.float32)
            if rotation.shape != (frame_count, 3, 3) or translation.shape != (
                frame_count,
                3,
            ):
                raise ValueError("HaWoR camera transforms are not frame-aligned")
            vertices_camera = np.einsum(
                "tvi,tij->tvj",
                vertices - translation[:, None, :],
                rotation,
            )
        focal_px = float(np.asarray(hawor["img_focal"]).item())
    return projected_finger_visible_fractions(
        vertices_camera=vertices_camera,
        valid_frames=valid,
        visible_masks=visible_masks,
        finger_parts=finger_parts,
        focal_px=focal_px,
        image_width=image_width,
        image_height=image_height,
        probe_radius_px=config.visible_probe_radius_px,
        point_support_threshold=config.visible_point_support,
        min_projected_vertices=config.visible_min_projected_vertices,
    )


def contact_scores_from_probabilities(
    probability: np.ndarray,
    contact_mask: np.ndarray,
    finger_parts: np.ndarray,
    palmar_mask: np.ndarray,
    *,
    top_fraction: float,
    min_points: int,
) -> np.ndarray:
    """Reduce 778 HaCo vertex probabilities to five finger scores."""

    values = np.asarray(probability, dtype=np.float32)
    filtered = np.asarray(contact_mask, dtype=bool)
    parts = np.asarray(finger_parts, dtype=np.int32)
    palmar = np.asarray(palmar_mask, dtype=bool)
    if not (
        values.shape == filtered.shape == parts.shape == palmar.shape == (778,)
    ):
        raise ValueError("HaCo probability/mask/assets must all have shape (778,)")
    if not 0.0 <= top_fraction <= 1.0 or min_points <= 0:
        raise ValueError("invalid HaCo reduction settings")

    result = np.zeros(len(FINGER_NAMES), dtype=np.float32)
    for finger_index, finger in enumerate(FINGER_NAMES):
        eligible = palmar & np.isin(parts, FINGER_PARTS[finger]) & filtered
        selected = values[eligible]
        if len(selected) < min_points:
            continue
        count = max(min_points, int(math.ceil(len(selected) * top_fraction)))
        count = min(count, len(selected))
        result[finger_index] = float(np.sort(selected)[-count:].mean())
    return result


def load_haco_scores(
    *,
    contact_dir: Path,
    camera2_images: list[Path],
    side: str,
    finger_parts: np.ndarray,
    palmar_mask: np.ndarray,
    config: StereoOcclusionConfig,
    require_complete: bool = False,
) -> tuple[np.ndarray, int]:
    """Load frame-aligned HaCo scores; missing frames stay zero.

    ``camera2_images`` retains its historical parameter name for callers, but
    the paths may belong to either synchronized camera view.
    """

    scores = np.zeros(
        (len(camera2_images), len(FINGER_NAMES)),
        dtype=np.float32,
    )
    missing = 0
    for frame_index, image_path in enumerate(camera2_images):
        contact_path = contact_dir / f"{image_path.stem}.npz"
        if not contact_path.is_file():
            missing += 1
            if require_complete:
                raise FileNotFoundError(
                    f"missing required HaCo frame: {contact_path}"
                )
            continue
        with np.load(contact_path) as contact:
            valid_key = f"{side}_valid"
            probability_key = f"{side}_contact_probability"
            mask_key = f"{side}_contact_mask"
            if require_complete:
                required = {valid_key, probability_key, mask_key}
                absent = sorted(required - set(contact.files))
                if absent:
                    raise ValueError(
                        f"{contact_path}: missing required keys {absent}"
                    )
            if valid_key in contact.files and not bool(contact[valid_key]):
                continue
            if probability_key in contact.files:
                probability = np.asarray(
                    contact[probability_key],
                    dtype=np.float32,
                )
            elif mask_key in contact.files:
                probability = np.asarray(contact[mask_key], dtype=np.float32)
            else:
                continue
            filtered = (
                np.asarray(contact[mask_key], dtype=bool)
                if mask_key in contact.files
                else probability >= config.haco_on
            )
        scores[frame_index] = contact_scores_from_probabilities(
            probability,
            filtered,
            finger_parts,
            palmar_mask,
            top_fraction=config.haco_top_fraction,
            min_points=config.haco_min_points,
        )
    return scores, missing


class FrameSource:
    """Sequential reader for either a video or a PNG/JPEG directory."""

    def __init__(self, path: Path, *, fps_if_images: float) -> None:
        self.path = path.resolve()
        self._capture: cv2.VideoCapture | None = None
        self._images: list[Path] | None = None
        self._index = 0
        if self.path.is_dir():
            self._images = discover_images(self.path)
            sample = cv2.imread(str(self._images[0]))
            if sample is None:
                raise RuntimeError(f"failed to read {self._images[0]}")
            self.height, self.width = sample.shape[:2]
            self.frame_count = len(self._images)
            self.fps = float(fps_if_images)
        else:
            self._capture = cv2.VideoCapture(str(self.path))
            if not self._capture.isOpened():
                raise FileNotFoundError(self.path)
            self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(
                round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
            )
            self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or fps_if_images)
        if (
            self.width <= 0
            or self.height <= 0
            or self.frame_count <= 0
            or self.fps <= 0
        ):
            raise ValueError(f"invalid frame source metadata: {self.path}")

    def read(self) -> np.ndarray:
        if self._images is not None:
            if self._index >= len(self._images):
                raise EOFError(f"image sequence ended at {self._index}: {self.path}")
            image_path = self._images[self._index]
            frame = cv2.imread(str(image_path))
            self._index += 1
            if frame is None:
                raise RuntimeError(f"failed to read {image_path}")
            return frame
        assert self._capture is not None
        ok, frame = self._capture.read()
        self._index += 1
        if not ok:
            raise EOFError(f"video ended at {self._index - 1}: {self.path}")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _infer_side(overlay_dir: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    manifest_path = overlay_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("--side is required when overlay manifest.json is missing")
    manifest = json.loads(manifest_path.read_text())
    side = str(manifest.get("side", ""))
    if side not in {"left", "right"}:
        raise ValueError(f"could not infer hand side from {manifest_path}")
    return side


def _label_panel(
    frame: np.ndarray,
    *,
    title: str,
    frame_index: int,
    evidence_text: str,
) -> np.ndarray:
    panel = np.asarray(frame, dtype=np.uint8).copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 68), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"frame {frame_index:04d} | {evidence_text}",
        (14, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return panel


def _mode_statistics(
    mask: np.ndarray,
    labels: np.ndarray,
    *,
    output_shape: tuple[int, int],
) -> dict[str, Any]:
    frame_count = len(mask)
    height, width = output_shape
    frame_track = np.zeros(frame_count, dtype=bool)
    finger_track = np.zeros((frame_count, len(FINGER_NAMES)), dtype=bool)
    finger_pixels = np.zeros(len(FINGER_NAMES), dtype=np.int64)
    total_pixels = 0
    for frame_index in range(frame_count):
        frame_mask = np.asarray(mask[frame_index], dtype=bool)
        frame_labels = np.asarray(labels[frame_index], dtype=np.uint8)
        if frame_labels.shape != (height, width):
            frame_labels = cv2.resize(
                frame_labels,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        pixels = int(frame_mask.sum())
        total_pixels += pixels
        frame_track[frame_index] = pixels > 0
        for finger_index in range(len(FINGER_NAMES)):
            selected = frame_mask & (frame_labels == finger_index + 1)
            count = int(selected.sum())
            finger_pixels[finger_index] += count
            finger_track[frame_index, finger_index] = count > 0
    return {
        "pixels": total_pixels,
        "frames": int(frame_track.sum()),
        "runs": _true_runs(frame_track),
        "per_finger": {
            finger: {
                "pixels": int(finger_pixels[index]),
                "frames": int(finger_track[:, index].sum()),
                "runs": _true_runs(finger_track[:, index]),
            }
            for index, finger in enumerate(FINGER_NAMES)
        },
    }


def validate_object_restore_mask(
    object_mask: np.ndarray,
    object_restore_mask: np.ndarray,
    *,
    frame_count: int,
    chunk_frames: int = 8,
) -> None:
    """Validate a raw-RGB restore mask against the modal geometry mask.

    The restore mask is intentionally stricter than the geometry mask: it must
    be exactly frame/pixel aligned, use the same dtype, and be a subset of the
    modal object support. Chunked subset checks keep validation bounded for
    full-resolution video masks.
    """

    modal = np.asanyarray(object_mask)
    restore = np.asanyarray(object_restore_mask)
    if modal.ndim != 3 or len(modal) != frame_count:
        raise ValueError(
            f"object mask must have shape ({frame_count},H,W), got {modal.shape}"
        )
    if restore.ndim != 3 or len(restore) != frame_count:
        raise ValueError(
            "object restore mask must be frame-aligned with shape "
            f"({frame_count},H,W), got {restore.shape}"
        )
    if restore.shape != modal.shape:
        raise ValueError(
            "object restore mask shape must exactly match object mask: "
            f"{restore.shape} != {modal.shape}"
        )
    if restore.dtype != modal.dtype:
        raise ValueError(
            "object restore mask dtype must match object mask: "
            f"{restore.dtype} != {modal.dtype}"
        )
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    for start in range(0, frame_count, chunk_frames):
        end = min(start + chunk_frames, frame_count)
        modal_chunk = np.asarray(modal[start:end], dtype=bool)
        restore_chunk = np.asarray(restore[start:end], dtype=bool)
        outside = restore_chunk & ~modal_chunk
        if np.any(outside):
            first_flat = int(np.argmax(outside))
            first = np.unravel_index(first_flat, outside.shape)
            frame_index = start + int(first[0])
            raise ValueError(
                "object restore mask must be a subset of the modal object "
                f"mask; first violation is in frame {frame_index}"
            )


def restore_camera2_object_pixels(
    background: np.ndarray,
    raw: np.ndarray,
    object_restore_mask: np.ndarray,
) -> np.ndarray:
    """Restore camera-2 RGB only where the verified restore mask permits it."""

    background_array = np.asarray(background)
    raw_array = np.asarray(raw)
    restore = np.asarray(object_restore_mask, dtype=bool)
    if background_array.shape != raw_array.shape:
        raise ValueError("background and raw camera-2 frames must share one shape")
    if background_array.ndim != 3 or background_array.shape[2] != 3:
        raise ValueError("camera-2 frames must have shape (H,W,3)")
    if restore.shape != background_array.shape[:2]:
        raise ValueError("object restore mask must match camera-2 frame geometry")
    composite_background = background_array.copy()
    composite_background[restore] = raw_array[restore]
    return composite_background


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera1_rgb_dir", type=Path, required=True)
    parser.add_argument("--camera2_rgb_dir", type=Path, required=True)
    parser.add_argument("--camera1_hawor", type=Path, required=True)
    parser.add_argument("--camera2_hawor", type=Path, required=True)
    parser.add_argument(
        "--camera1_tracks",
        type=Path,
        default=None,
        help="model_tracks.npy or its HaWoR parent; auto-discovered if omitted",
    )
    parser.add_argument(
        "--camera2_tracks",
        type=Path,
        default=None,
        help="model_tracks.npy or its HaWoR parent; auto-discovered if omitted",
    )
    parser.add_argument(
        "--camera1_visible_mask",
        type=Path,
        default=None,
        help="Optional camera-1 SAM masks_arm.npy used for finger visibility",
    )
    parser.add_argument(
        "--camera2_visible_mask",
        type=Path,
        default=None,
        help="Optional camera-2 SAM masks_arm.npy used for finger visibility",
    )
    parser.add_argument(
        "--camera1_contact_dir",
        type=Path,
        default=None,
        help="Optional camera-1 HaCo directory; scores are max-fused by finger",
    )
    parser.add_argument(
        "--contact_dir",
        type=Path,
        required=True,
        help="Camera-2 HaCo directory (legacy argument retained)",
    )
    parser.add_argument(
        "--background",
        type=Path,
        required=True,
        help="Camera-2 inpainted video or image directory",
    )
    parser.add_argument("--overlay_dir", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument(
        "--object_restore_mask",
        type=Path,
        default=None,
        help=(
            "Optional clean subset of --object_mask used only to restore raw "
            "camera-2 RGB. Geometry and occlusion continue to use the modal "
            "--object_mask. Defaults to --object_mask for legacy behavior."
        ),
    )
    parser.add_argument(
        "--object_depth_mask",
        type=Path,
        default=None,
        help="Optional mask used only for robust sensor-depth sampling",
    )
    parser.add_argument(
        "--scene_depth_camera1",
        type=Path,
        default=None,
        help=(
            "Optional metric camera-1 depth already registered into camera-2 "
            "output coordinates (T,H,W)"
        ),
    )
    parser.add_argument(
        "--scene_depth",
        type=Path,
        default=None,
        help=(
            "Optional registered metric camera-2 depth array (T,H,W); legacy "
            "argument retained"
        ),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default=None)
    parser.add_argument(
        "--camera1_side",
        choices=("left", "right"),
        default=None,
        help="Override if camera 1 handedness is mirrored",
    )
    parser.add_argument(
        "--camera2_side",
        choices=("left", "right"),
        default=None,
        help="Override if camera 2 handedness is mirrored",
    )
    parser.add_argument(
        "--camera1_frame_offset",
        type=int,
        default=0,
        help=(
            "Temporal lookup offset on the camera-2/output axis: camera-1 "
            "source index = camera-2 frame k + offset. For example, -1 "
            "maps MH k to SH k-1; out-of-range evidence fails open."
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--visibility_on", type=float, default=0.55)
    parser.add_argument("--visibility_off", type=float, default=0.28)
    parser.add_argument("--assisted_visibility_on", type=float, default=0.30)
    parser.add_argument("--assisted_visibility_off", type=float, default=0.15)
    parser.add_argument("--visibility_min_on_frames", type=int, default=2)
    parser.add_argument("--visibility_hold_frames", type=int, default=3)
    parser.add_argument("--haco_on", type=float, default=0.72)
    parser.add_argument("--haco_off", type=float, default=0.55)
    parser.add_argument("--haco_min_on_frames", type=int, default=2)
    parser.add_argument("--haco_hold_frames", type=int, default=3)
    parser.add_argument("--visible_probe_radius_px", type=int, default=3)
    parser.add_argument("--visible_point_support", type=float, default=0.20)
    parser.add_argument(
        "--visible_min_projected_vertices",
        type=int,
        default=8,
    )
    parser.add_argument("--depth_margin_m", type=float, default=0.030)
    parser.add_argument(
        "--depth_agreement_tolerance_m",
        type=float,
        default=0.020,
        help=(
            "Maximum C1/C2 object-depth difference for accepting C1; "
            "disagreement keeps C2 and C1-only depth fails open"
        ),
    )
    parser.add_argument("--object_depth_erode_px", type=int, default=12)
    parser.add_argument("--min_occlusion_run_frames", type=int, default=2)
    parser.add_argument("--robot_edge_sigma_px", type=float, default=0.6)
    parser.add_argument("--occlusion_edge_sigma_px", type=float, default=0.0)
    parser.add_argument(
        "--include_haco_only",
        action="store_true",
        help=(
            "Also emit a C1+C2 max-HaCo-only mode; it ignores visibility and "
            "depth and hides only active semantic fingers inside object_mask"
        ),
    )
    parser.add_argument(
        "--include_visibility_haco",
        action="store_true",
        help=(
            "Also emit RGB-only directional stereo + dual-HaCo layering: "
            "strong SH-visible/MH-hidden evidence is kept, while ambiguous "
            "stereo evidence requires max-fused HaCo. Requires both visible "
            "masks and --camera1_contact_dir"
        ),
    )
    parser.add_argument(
        "--include_haco_priority",
        action="store_true",
        help=(
            "Also emit HaCo-primary layering: max-fused active contact hides "
            "semantic fingers inside object_mask unless resolved metric depth "
            "confidently says hand_front; requires both metric-depth NPZs"
        ),
    )
    parser.add_argument(
        "--include_ablation_modes",
        action="store_true",
        help=(
            "Also emit no_occlusion, camera2_depth_only, vote_2of3, and "
            "confidence_ensemble plus a separate 2x2 comparison; requires "
            "both native metric-depth NPZs"
        ),
    )
    parser.add_argument(
        "--camera1_metric_depth_npz",
        type=Path,
        default=None,
        help=(
            "Optional C1 native-RGB local hand/object metric depth NPZ; must "
            "be supplied together with --camera2_metric_depth_npz"
        ),
    )
    parser.add_argument(
        "--camera2_metric_depth_npz",
        type=Path,
        default=None,
        help=(
            "Optional C2 native-RGB local hand/object metric depth NPZ; "
            "enables the metric_depth_order output mode with the C1 NPZ"
        ),
    )
    parser.add_argument(
        "--metric_depth_separation_margin_m",
        type=float,
        default=0.020,
    )
    parser.add_argument(
        "--metric_depth_visibility_assist_threshold",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--metric_depth_min_hand_samples",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--metric_depth_min_object_samples",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--ablation_depth_separation_margin_m",
        type=float,
        default=0.010,
        help=(
            "Ablation-only C2 depth-order margin/deadzone; leaves the "
            "established metric-depth mode threshold unchanged"
        ),
    )
    parser.add_argument(
        "--confidence_depth_weight",
        type=float,
        default=0.65,
        help="Directional weight for signed C2 metric-depth separation",
    )
    parser.add_argument(
        "--confidence_stereo_weight",
        type=float,
        default=0.35,
        help="Directional weight for C1-visible/C2-hidden evidence",
    )
    parser.add_argument(
        "--confidence_depth_saturation_m",
        type=float,
        default=0.025,
        help="Absolute depth separation where signed depth confidence is 1",
    )
    parser.add_argument(
        "--confidence_stereo_start",
        type=float,
        default=0.30,
        help="Stereo evidence where positive directional confidence starts",
    )
    parser.add_argument(
        "--confidence_stereo_saturation",
        type=float,
        default=0.55,
        help="Stereo evidence where positive directional confidence reaches 1",
    )
    parser.add_argument(
        "--confidence_contact_floor",
        type=float,
        default=0.50,
        help="Contact gain when HaCo confidence is zero or missing",
    )
    parser.add_argument(
        "--confidence_score_threshold",
        type=float,
        default=0.18,
        help="Minimum final confidence-ensemble score for hiding a finger",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = StereoOcclusionConfig(
        visibility_on=args.visibility_on,
        visibility_off=args.visibility_off,
        assisted_visibility_on=args.assisted_visibility_on,
        assisted_visibility_off=args.assisted_visibility_off,
        visibility_min_on_frames=args.visibility_min_on_frames,
        visibility_hold_frames=args.visibility_hold_frames,
        haco_on=args.haco_on,
        haco_off=args.haco_off,
        haco_min_on_frames=args.haco_min_on_frames,
        haco_hold_frames=args.haco_hold_frames,
        visible_probe_radius_px=args.visible_probe_radius_px,
        visible_point_support=args.visible_point_support,
        visible_min_projected_vertices=args.visible_min_projected_vertices,
        depth_margin_m=args.depth_margin_m,
        depth_agreement_tolerance_m=args.depth_agreement_tolerance_m,
        metric_depth_separation_margin_m=(
            args.metric_depth_separation_margin_m
        ),
        metric_depth_visibility_assist_threshold=(
            args.metric_depth_visibility_assist_threshold
        ),
        metric_depth_min_hand_samples=args.metric_depth_min_hand_samples,
        metric_depth_min_object_samples=args.metric_depth_min_object_samples,
        object_depth_erode_px=args.object_depth_erode_px,
        min_occlusion_run_frames=args.min_occlusion_run_frames,
        robot_edge_sigma_px=args.robot_edge_sigma_px,
        occlusion_edge_sigma_px=args.occlusion_edge_sigma_px,
    )
    config.validate()
    ablation_config = AblationConfig(
        confidence_depth_weight=args.confidence_depth_weight,
        confidence_stereo_weight=args.confidence_stereo_weight,
        depth_separation_margin_m=(
            args.ablation_depth_separation_margin_m
        ),
        confidence_depth_saturation_m=args.confidence_depth_saturation_m,
        confidence_stereo_start=args.confidence_stereo_start,
        confidence_stereo_saturation=(
            args.confidence_stereo_saturation
        ),
        confidence_contact_floor=args.confidence_contact_floor,
        confidence_score_threshold=args.confidence_score_threshold,
    )
    if args.include_ablation_modes:
        ablation_config.validate()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    validate_visibility_haco_inputs(
        enabled=args.include_visibility_haco,
        camera1_visible_mask=args.camera1_visible_mask,
        camera2_visible_mask=args.camera2_visible_mask,
        camera1_contact_dir=args.camera1_contact_dir,
    )
    output_modes, metric_depth_enabled = resolve_output_modes(
        include_haco_only=args.include_haco_only,
        include_visibility_haco=args.include_visibility_haco,
        include_haco_priority=args.include_haco_priority,
        include_ablation_modes=args.include_ablation_modes,
        camera1_metric_depth_npz=args.camera1_metric_depth_npz,
        camera2_metric_depth_npz=args.camera2_metric_depth_npz,
    )
    # Keep the established three-panel comparison byte-layout compatible.
    # HaCo-only is emitted as its own video and mask when requested.
    comparison_modes = MODE_NAMES

    camera1_rgb = args.camera1_rgb_dir.resolve()
    camera2_rgb = args.camera2_rgb_dir.resolve()
    camera1_images = discover_images(camera1_rgb)
    camera2_images = discover_images(camera2_rgb)
    if len(camera1_images) != len(camera2_images):
        raise ValueError(
            "stereo RGB sequences must already be synchronized: "
            f"{len(camera1_images)} vs {len(camera2_images)} frames"
        )
    frame_count = len(camera2_images)
    camera1_source_frame_indices = align_camera1_to_camera2(
        np.arange(frame_count, dtype=np.int64),
        args.camera1_frame_offset,
        fill_value=-1,
    )
    if args.camera1_frame_offset and args.scene_depth_camera1 is not None:
        raise ValueError(
            "--camera1_frame_offset is not supported with legacy "
            "--scene_depth_camera1; pre-align that registered depth source "
            "or use native --camera1_metric_depth_npz evidence"
        )

    overlay_dir = args.overlay_dir.resolve()
    rendered_side = _infer_side(overlay_dir, args.side)
    camera1_side = args.camera1_side or rendered_side
    camera2_side = args.camera2_side or rendered_side
    camera1_hawor = args.camera1_hawor.resolve()
    camera2_hawor = args.camera2_hawor.resolve()
    camera1_tracks = resolve_model_tracks(args.camera1_tracks, camera1_hawor)
    camera2_tracks = resolve_model_tracks(args.camera2_tracks, camera2_hawor)

    camera1_sample = cv2.imread(str(camera1_images[0]))
    camera2_sample = cv2.imread(str(camera2_images[0]))
    if camera1_sample is None or camera2_sample is None:
        raise RuntimeError("failed to read the first synchronized stereo frame")
    camera1_height, camera1_width = camera1_sample.shape[:2]
    camera2_height, camera2_width = camera2_sample.shape[:2]

    assets = Path(__file__).resolve().parents[1] / "retargeting" / "assets"
    camera1_finger_parts = np.load(
        assets / f"finger_part_{camera1_side}.npy"
    ).astype(np.int32)
    camera2_finger_parts = np.load(
        assets / f"finger_part_{camera2_side}.npy"
    ).astype(np.int32)

    camera1_observed_hand, camera1_observation_source = (
        build_observation_confidence(
            hawor_path=camera1_hawor,
            track_path=camera1_tracks,
            side=camera1_side,
            frame_count=frame_count,
        )
    )
    camera2_observed_hand, camera2_observation_source = (
        build_observation_confidence(
            hawor_path=camera2_hawor,
            track_path=camera2_tracks,
            side=camera2_side,
            frame_count=frame_count,
        )
    )
    camera1_visible_fraction = np.full(
        (frame_count, len(FINGER_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    camera2_visible_fraction = camera1_visible_fraction.copy()
    camera1_projected_counts = np.zeros_like(
        camera1_visible_fraction,
        dtype=np.int32,
    )
    camera2_projected_counts = np.zeros_like(
        camera2_visible_fraction,
        dtype=np.int32,
    )
    if args.camera1_visible_mask is not None:
        camera1_visible_fraction, camera1_projected_counts = (
            load_projected_visible_fractions(
                hawor_path=camera1_hawor,
                visible_mask_path=args.camera1_visible_mask.resolve(),
                side=camera1_side,
                frame_count=frame_count,
                finger_parts=camera1_finger_parts,
                image_width=camera1_width,
                image_height=camera1_height,
                config=config,
            )
        )
        camera1_observed = fuse_visible_fraction_with_detector(
            camera1_observed_hand,
            camera1_visible_fraction,
        )
        camera1_observation_source += "+sam_projected_finger_fraction"
    else:
        camera1_observed = np.repeat(
            camera1_observed_hand[:, None],
            len(FINGER_NAMES),
            axis=1,
        )
    if args.camera2_visible_mask is not None:
        camera2_visible_fraction, camera2_projected_counts = (
            load_projected_visible_fractions(
                hawor_path=camera2_hawor,
                visible_mask_path=args.camera2_visible_mask.resolve(),
                side=camera2_side,
                frame_count=frame_count,
                finger_parts=camera2_finger_parts,
                image_width=camera2_width,
                image_height=camera2_height,
                config=config,
            )
        )
        camera2_observed = fuse_visible_fraction_with_detector(
            camera2_observed_hand,
            camera2_visible_fraction,
        )
        camera2_observation_source += "+sam_projected_finger_fraction"
    else:
        camera2_observed = np.repeat(
            camera2_observed_hand[:, None],
            len(FINGER_NAMES),
            axis=1,
        )

    # Camera 2 (MH for 08_04) is the immutable output/GT time axis.  Shift
    # only camera-1 evidence; never reorder either source sequence.  NaN at a
    # boundary is intentional so SH cannot manufacture a directional cue.
    camera1_observed = align_camera1_to_camera2(
        camera1_observed,
        args.camera1_frame_offset,
        fill_value=np.nan,
    )
    camera1_visible_fraction = align_camera1_to_camera2(
        camera1_visible_fraction,
        args.camera1_frame_offset,
        fill_value=np.nan,
    )
    camera1_projected_counts = align_camera1_to_camera2(
        camera1_projected_counts,
        args.camera1_frame_offset,
        fill_value=0,
    )
    if args.camera1_frame_offset:
        camera1_observation_source += (
            f"+camera2_axis_offset({args.camera1_frame_offset:+d})"
        )

    camera2_palmar_mask = np.load(
        assets / f"palmar_mask_{camera2_side}.npy"
    ).astype(bool)
    camera2_haco_scores, camera2_missing_contact_frames = load_haco_scores(
        contact_dir=args.contact_dir.resolve(),
        camera2_images=camera2_images,
        side=camera2_side,
        finger_parts=camera2_finger_parts,
        palmar_mask=camera2_palmar_mask,
        config=config,
        require_complete=args.include_visibility_haco,
    )
    if args.camera1_contact_dir is not None:
        camera1_palmar_mask = np.load(
            assets / f"palmar_mask_{camera1_side}.npy"
        ).astype(bool)
        camera1_haco_scores, camera1_missing_contact_frames = load_haco_scores(
            contact_dir=args.camera1_contact_dir.resolve(),
            camera2_images=camera1_images,
            side=camera1_side,
            finger_parts=camera1_finger_parts,
            palmar_mask=camera1_palmar_mask,
            config=config,
            require_complete=args.include_visibility_haco,
        )
        camera1_haco_scores = align_camera1_to_camera2(
            camera1_haco_scores,
            args.camera1_frame_offset,
            fill_value=np.nan,
        )
    else:
        camera1_haco_scores = np.full_like(camera2_haco_scores, np.nan)
        camera1_missing_contact_frames = frame_count
    validate_visibility_haco_coverage(
        enabled=args.include_visibility_haco,
        frame_count=frame_count,
        camera1_missing_frames=camera1_missing_contact_frames,
        camera2_missing_frames=camera2_missing_contact_frames,
    )
    haco_scores = fuse_haco_scores(
        camera1_haco_scores if args.camera1_contact_dir is not None else None,
        camera2_haco_scores,
    )
    (
        active_tracks,
        visibility_evidence,
        strong_visibility_active,
        haco_active,
    ) = temporal_mode_decisions(
        camera1_observed,
        camera2_observed,
        haco_scores,
        config,
    )

    camera1_metric_depth: MetricFingerDepthEvidence | None = None
    camera2_metric_depth: MetricFingerDepthEvidence | None = None
    camera1_metric_order: np.ndarray | None = None
    camera2_metric_order: np.ndarray | None = None
    metric_depth_order: np.ndarray | None = None
    metric_depth_order_source: np.ndarray | None = None
    haco_priority_selected: np.ndarray | None = None
    haco_priority_depth_vetoed: np.ndarray | None = None
    camera2_ablation_order: np.ndarray | None = None
    camera2_depth_only_selected: np.ndarray | None = None
    vote_2of3_evidence: Vote2Of3Evidence | None = None
    confidence_ensemble_evidence: ConfidenceEnsembleEvidence | None = None
    if metric_depth_enabled:
        assert args.camera1_metric_depth_npz is not None
        assert args.camera2_metric_depth_npz is not None
        camera1_metric_depth = load_metric_finger_depth_evidence(
            args.camera1_metric_depth_npz.resolve(),
            frame_count=frame_count,
            min_hand_samples=config.metric_depth_min_hand_samples,
            min_object_samples=config.metric_depth_min_object_samples,
        )
        camera1_metric_depth = MetricFingerDepthEvidence(
            hand_depth_m_raw=align_camera1_to_camera2(
                camera1_metric_depth.hand_depth_m_raw,
                args.camera1_frame_offset,
                fill_value=np.nan,
            ),
            object_depth_m_raw=align_camera1_to_camera2(
                camera1_metric_depth.object_depth_m_raw,
                args.camera1_frame_offset,
                fill_value=np.nan,
            ),
            hand_sample_count=align_camera1_to_camera2(
                camera1_metric_depth.hand_sample_count,
                args.camera1_frame_offset,
                fill_value=0,
            ),
            object_sample_count=align_camera1_to_camera2(
                camera1_metric_depth.object_sample_count,
                args.camera1_frame_offset,
                fill_value=0,
            ),
            hand_depth_m=align_camera1_to_camera2(
                camera1_metric_depth.hand_depth_m,
                args.camera1_frame_offset,
                fill_value=np.nan,
            ),
            object_depth_m=align_camera1_to_camera2(
                camera1_metric_depth.object_depth_m,
                args.camera1_frame_offset,
                fill_value=np.nan,
            ),
        )
        camera2_metric_depth = load_metric_finger_depth_evidence(
            args.camera2_metric_depth_npz.resolve(),
            frame_count=frame_count,
            min_hand_samples=config.metric_depth_min_hand_samples,
            min_object_samples=config.metric_depth_min_object_samples,
        )
        camera1_metric_order = classify_hand_object_depth_order(
            camera1_metric_depth.hand_depth_m,
            camera1_metric_depth.object_depth_m,
            separation_margin_m=config.metric_depth_separation_margin_m,
        )
        camera2_metric_order = classify_hand_object_depth_order(
            camera2_metric_depth.hand_depth_m,
            camera2_metric_depth.object_depth_m,
            separation_margin_m=config.metric_depth_separation_margin_m,
        )
        metric_depth_order, metric_depth_order_source = (
            resolve_camera2_depth_order(
                camera1_metric_order,
                camera2_metric_order,
                visibility_evidence,
                visibility_assist_threshold=(
                    config.metric_depth_visibility_assist_threshold
                ),
            )
        )
        if args.include_haco_priority:
            haco_priority_selected = haco_active & (
                metric_depth_order != DEPTH_ORDER_HAND_FRONT
            )
            haco_priority_depth_vetoed = haco_active & (
                metric_depth_order == DEPTH_ORDER_HAND_FRONT
            )
        if args.include_ablation_modes:
            camera2_ablation_order = classify_hand_object_depth_order(
                camera2_metric_depth.hand_depth_m,
                camera2_metric_depth.object_depth_m,
                separation_margin_m=(
                    ablation_config.depth_separation_margin_m
                ),
            )
            camera2_depth_only_selected = select_camera2_depth_only_fingers(
                camera2_ablation_order
            )
            vote_2of3_evidence = compute_vote_2of3_evidence(
                haco_active=haco_active,
                camera2_depth_order=camera2_ablation_order,
                strong_stereo_active=strong_visibility_active,
            )
            confidence_ensemble_evidence = (
                compute_confidence_ensemble_evidence(
                    camera2_hand_depth_m=camera2_metric_depth.hand_depth_m,
                    camera2_object_depth_m=(
                        camera2_metric_depth.object_depth_m
                    ),
                    stereo_visibility=visibility_evidence,
                    haco_confidence=haco_scores,
                    depth_deadzone_m=(
                        ablation_config.depth_separation_margin_m
                    ),
                    config=ablation_config,
                )
            )

    robot_rgb = np.load(overlay_dir / "robot_rgb.npy", mmap_mode="r")
    robot_depth = np.load(overlay_dir / "robot_depth.npy", mmap_mode="r")
    robot_mask = np.load(overlay_dir / "robot_mask.npy", mmap_mode="r")
    finger_labels = np.load(
        overlay_dir / "robot_finger_labels.npy",
        mmap_mode="r",
    )
    if robot_mask.ndim != 3:
        raise ValueError(f"robot_mask must have shape (T,H,W), got {robot_mask.shape}")
    overlay_height, overlay_width = robot_mask.shape[1:]
    expected_overlay_shapes = {
        "robot_rgb": (frame_count, overlay_height, overlay_width, 3),
        "robot_depth": (frame_count, overlay_height, overlay_width),
        "robot_mask": (frame_count, overlay_height, overlay_width),
        "robot_finger_labels": (frame_count, overlay_height, overlay_width),
    }
    for name, expected in expected_overlay_shapes.items():
        array = {
            "robot_rgb": robot_rgb,
            "robot_depth": robot_depth,
            "robot_mask": robot_mask,
            "robot_finger_labels": finger_labels,
        }[name]
        if array.shape != expected:
            raise ValueError(f"{name} shape mismatch: {array.shape} != {expected}")

    with FrameSource(args.background, fps_if_images=args.fps) as metadata_source:
        width = metadata_source.width
        height = metadata_source.height
        fps = metadata_source.fps
        if metadata_source.frame_count != frame_count:
            raise ValueError(
                "background/frame count mismatch: "
                f"{metadata_source.frame_count} vs {frame_count}"
            )
    if not np.isclose(
        overlay_width / overlay_height,
        width / height,
        atol=1e-3,
    ):
        raise ValueError(
            "Isaac/background aspect mismatch: "
            f"{overlay_width}x{overlay_height} vs {width}x{height}"
        )

    object_mask_path = args.object_mask.resolve()
    object_restore_mask_path = (
        args.object_restore_mask.resolve()
        if args.object_restore_mask is not None
        else object_mask_path
    )
    object_mask = np.load(object_mask_path, mmap_mode="r", allow_pickle=False)
    object_restore_mask = (
        object_mask
        if object_restore_mask_path == object_mask_path
        else np.load(
            object_restore_mask_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    )
    validate_object_restore_mask(
        object_mask,
        object_restore_mask,
        frame_count=frame_count,
    )
    object_depth_mask = (
        np.load(args.object_depth_mask.resolve(), mmap_mode="r")
        if args.object_depth_mask is not None
        else object_mask
    )
    scene_depth_camera1 = (
        np.load(args.scene_depth_camera1.resolve(), mmap_mode="r")
        if args.scene_depth_camera1 is not None
        else None
    )
    scene_depth_camera2 = (
        np.load(args.scene_depth.resolve(), mmap_mode="r")
        if args.scene_depth is not None
        else None
    )
    if object_depth_mask.ndim != 3 or len(object_depth_mask) != frame_count:
        raise ValueError("object depth mask is not frame-aligned")
    for camera_name, depth_array in (
        ("camera-1 scene depth", scene_depth_camera1),
        ("camera-2 scene depth", scene_depth_camera2),
    ):
        if depth_array is not None and (
            depth_array.ndim != 3 or len(depth_array) != frame_count
        ):
            raise ValueError(f"{camera_name} is not frame-aligned")
    if scene_depth_camera1 is not None:
        camera1_object_depth_track = estimate_object_depth_track(
            scene_depth_camera1,
            object_depth_mask,
            output_shape=(height, width),
            erode_px=config.object_depth_erode_px,
        )
    else:
        camera1_object_depth_track = np.full(
            frame_count,
            np.nan,
            dtype=np.float32,
        )
    if scene_depth_camera2 is not None:
        camera2_object_depth_track = estimate_object_depth_track(
            scene_depth_camera2,
            object_depth_mask,
            output_shape=(height, width),
            erode_px=config.object_depth_erode_px,
        )
    else:
        camera2_object_depth_track = np.full(
            frame_count,
            np.nan,
            dtype=np.float32,
        )
    object_depth_track, object_depth_source = fuse_object_depth_tracks(
        camera1_object_depth_track,
        camera2_object_depth_track,
        agreement_tolerance_m=config.depth_agreement_tolerance_m,
    )
    camera1_depth_valid = np.isfinite(camera1_object_depth_track)
    camera2_depth_valid = np.isfinite(camera2_object_depth_track)
    object_depth_both_valid = camera1_depth_valid & camera2_depth_valid
    object_depth_disagreement_m = np.full(
        frame_count,
        np.nan,
        dtype=np.float32,
    )
    object_depth_disagreement_m[object_depth_both_valid] = (
        camera1_object_depth_track[object_depth_both_valid]
        - camera2_object_depth_track[object_depth_both_valid]
    )
    object_depth_cameras_agree = object_depth_both_valid & (
        np.abs(object_depth_disagreement_m)
        <= config.depth_agreement_tolerance_m
    )

    output_dir = args.out_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".stereo_occlusion.",
            dir=output_dir.parent,
        )
    )
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    mask_buffers = {
        mode: np.lib.format.open_memmap(
            staging / f"occluded_finger_mask_{mode}.npy",
            mode="w+",
            dtype=bool,
            shape=(frame_count, height, width),
        )
        for mode in output_modes
    }
    candidate_presence = {
        mode: np.zeros((frame_count, len(FINGER_NAMES)), dtype=bool)
        for mode in output_modes
    }

    for frame_index in range(frame_count):
        core_object = _resize_mask(object_mask[frame_index], width, height)
        (
            _frame_robot_rgb,
            frame_robot_depth,
            frame_robot_mask,
            frame_finger_mask,
            frame_finger_labels,
        ) = _resize_overlay_frame(
            robot_rgb[frame_index],
            robot_depth[frame_index],
            robot_mask[frame_index],
            finger_labels[frame_index],
            width=width,
            height=height,
        )
        frame_active = {
            mode: active_tracks[mode][frame_index]
            for mode in MODE_NAMES
        }
        frame_masks = compute_mode_occlusion_masks(
            robot_mask=frame_robot_mask,
            finger_labels=frame_finger_labels,
            robot_depth=frame_robot_depth,
            object_mask=core_object,
            active_fingers=frame_active,
            object_depth_m=float(object_depth_track[frame_index]),
            depth_margin_m=config.depth_margin_m,
        )
        if args.include_haco_only:
            frame_masks[HACO_ONLY_MODE] = compute_haco_only_occlusion_mask(
                robot_mask=frame_robot_mask,
                finger_labels=frame_finger_labels,
                object_mask=core_object,
                haco_active_fingers=haco_active[frame_index],
            )
        if args.include_visibility_haco:
            frame_masks[VISIBILITY_HACO_MODE] = (
                compute_visibility_haco_occlusion_mask(
                    robot_mask=frame_robot_mask,
                    finger_labels=frame_finger_labels,
                    object_mask=core_object,
                    visibility_haco_active_fingers=(
                        active_tracks["visibility_depth_haco"][frame_index]
                    ),
                )
            )
        if metric_depth_enabled:
            assert metric_depth_order is not None
            frame_masks[METRIC_DEPTH_ORDER_MODE] = (
                compute_depth_order_occlusion_mask(
                    robot_mask=frame_robot_mask,
                    finger_labels=frame_finger_labels,
                    object_mask=core_object,
                    camera2_depth_order=metric_depth_order[frame_index],
                    haco_active_fingers=haco_active[frame_index],
                )
            )
        if args.include_haco_priority:
            assert metric_depth_order is not None
            frame_masks[HACO_PRIORITY_MODE] = (
                compute_haco_priority_occlusion_mask(
                    robot_mask=frame_robot_mask,
                    finger_labels=frame_finger_labels,
                    object_mask=core_object,
                    resolved_depth_order=metric_depth_order[frame_index],
                    haco_active_fingers=haco_active[frame_index],
                )
            )
        if args.include_ablation_modes:
            assert camera2_ablation_order is not None
            assert camera2_depth_only_selected is not None
            assert vote_2of3_evidence is not None
            assert confidence_ensemble_evidence is not None
            frame_masks[NO_OCCLUSION_MODE] = compute_no_occlusion_mask(
                frame_robot_mask
            )
            frame_masks[CAMERA2_DEPTH_ONLY_MODE] = (
                compute_selected_finger_occlusion_mask(
                    robot_mask=frame_robot_mask,
                    finger_labels=frame_finger_labels,
                    object_mask=core_object,
                    selected_fingers=(
                        camera2_depth_only_selected[frame_index]
                    ),
                )
            )
            frame_masks[VOTE_2OF3_MODE] = (
                compute_selected_finger_occlusion_mask(
                    robot_mask=frame_robot_mask,
                    finger_labels=frame_finger_labels,
                    object_mask=core_object,
                    selected_fingers=(
                        vote_2of3_evidence.selected[frame_index]
                    ),
                )
            )
            frame_masks[CONFIDENCE_ENSEMBLE_MODE] = (
                compute_selected_finger_occlusion_mask(
                    robot_mask=frame_robot_mask,
                    finger_labels=frame_finger_labels,
                    object_mask=core_object,
                    selected_fingers=(
                        confidence_ensemble_evidence.selected[frame_index]
                    ),
                )
            )
        for mode, frame_mask in frame_masks.items():
            if np.any(frame_mask & ~frame_finger_mask):
                raise RuntimeError(
                    f"{mode} removed a non-finger pixel at frame {frame_index}"
                )
            mask_buffers[mode][frame_index] = frame_mask
            for finger_index in range(len(FINGER_NAMES)):
                candidate_presence[mode][frame_index, finger_index] = bool(
                    np.any(frame_mask & (frame_finger_labels == finger_index + 1))
                )
        if (frame_index + 1) % 100 == 0:
            print(f"[stereo-mask] {frame_index + 1}/{frame_count}", flush=True)
    for buffer in mask_buffers.values():
        buffer.flush()

    stable_presence: dict[str, np.ndarray] = {}
    for mode in output_modes:
        stable_presence[mode] = suppress_short_runs(
            candidate_presence[mode],
            min_frames=config.min_occlusion_run_frames,
        )
    for frame_index in range(frame_count):
        labels = np.asarray(finger_labels[frame_index], dtype=np.uint8)
        if labels.shape != (height, width):
            labels = cv2.resize(
                labels,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        for mode in output_modes:
            frame_mask = np.asarray(mask_buffers[mode][frame_index], dtype=bool).copy()
            for finger_index in range(len(FINGER_NAMES)):
                if not stable_presence[mode][frame_index, finger_index]:
                    frame_mask[labels == finger_index + 1] = False
            mask_buffers[mode][frame_index] = frame_mask
    for buffer in mask_buffers.values():
        buffer.flush()

    individual_writers = {
        mode: _open_writer(
            staging / f"video_overlay_{mode}.mp4",
            fps,
            (width, height),
        )
        for mode in output_modes
    }
    comparison_writer = _open_writer(
        staging / "video_compare_stereo_modes.mp4",
        fps,
        (width * len(comparison_modes), height),
    )
    ablation_comparison_writer = (
        _open_writer(
            staging / "video_compare_ablation_modes_2x2.mp4",
            fps,
            (width * 2, height * 2),
        )
        if args.include_ablation_modes
        else None
    )
    titles = {
        "visibility": "C1 visible + C2 hidden",
        "visibility_depth": "+ registered depth",
        "visibility_depth_haco": "+ depth + HaCo assist",
        HACO_ONLY_MODE: "C1+C2 HaCo only",
        VISIBILITY_HACO_MODE: "Stereo direction + C1+C2 HaCo",
        METRIC_DEPTH_ORDER_MODE: "Metric hand/object order",
        HACO_PRIORITY_MODE: "HaCo priority + depth veto",
        NO_OCCLUSION_MODE: "No occlusion baseline",
        CAMERA2_DEPTH_ONLY_MODE: "Camera 2 metric depth only",
        VOTE_2OF3_MODE: "Fixed-panel 2-of-3 vote",
        CONFIDENCE_ENSEMBLE_MODE: "Confidence ensemble",
    }
    modal_object_pixel_count = np.zeros(frame_count, dtype=np.int64)
    raw_object_pixel_count = np.zeros(frame_count, dtype=np.int64)
    with FrameSource(args.background, fps_if_images=args.fps) as background_source, FrameSource(
        camera2_rgb,
        fps_if_images=args.fps,
    ) as raw_source:
        if raw_source.frame_count != frame_count:
            raise ValueError("camera-2 RGB frame count changed during run")
        try:
            for frame_index in range(frame_count):
                background = background_source.read()
                raw = raw_source.read()
                if background.shape[:2] != (height, width):
                    background = cv2.resize(
                        background,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )
                if raw.shape[:2] != (height, width):
                    raw = cv2.resize(
                        raw,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )
                core_object = _resize_mask(object_mask[frame_index], width, height)
                restore_object = (
                    core_object
                    if object_restore_mask is object_mask
                    else _resize_mask(
                        object_restore_mask[frame_index],
                        width,
                        height,
                    )
                )
                if np.any(restore_object & ~core_object):
                    raise RuntimeError(
                        "resized object restore mask escaped modal object support"
                    )
                composite_background = restore_camera2_object_pixels(
                    background,
                    raw,
                    restore_object,
                )
                modal_object_pixel_count[frame_index] = int(core_object.sum())
                raw_object_pixel_count[frame_index] = int(restore_object.sum())
                (
                    frame_robot_rgb,
                    _frame_robot_depth,
                    frame_robot_mask,
                    frame_finger_mask,
                    _frame_finger_labels,
                ) = _resize_overlay_frame(
                    robot_rgb[frame_index],
                    robot_depth[frame_index],
                    robot_mask[frame_index],
                    finger_labels[frame_index],
                    width=width,
                    height=height,
                )
                panels: list[np.ndarray] = []
                ablation_panels: list[np.ndarray] = []
                for mode in output_modes:
                    final, _robot_only, _alpha = composite_frame(
                        composite_background,
                        frame_robot_rgb,
                        frame_robot_mask,
                        frame_finger_mask,
                        np.asarray(mask_buffers[mode][frame_index], dtype=bool),
                        robot_edge_sigma_px=config.robot_edge_sigma_px,
                        occlusion_edge_sigma_px=config.occlusion_edge_sigma_px,
                    )
                    individual_writers[mode].write(final)
                    active_names = [
                        FINGER_NAMES[index]
                        for index in range(len(FINGER_NAMES))
                        if stable_presence[mode][frame_index, index]
                    ]
                    decision_text = (
                        ",".join(active_names) if active_names else "fail-open"
                    )
                    if mode == HACO_ONLY_MODE:
                        evidence_text = (
                            f"{decision_text} | max-HaCo="
                            f"{haco_scores[frame_index].mean():.2f}"
                        )
                    elif mode == VISIBILITY_HACO_MODE:
                        evidence_text = (
                            f"{decision_text} | stereo="
                            f"{visibility_evidence[frame_index].mean():.2f} "
                            f"max-HaCo={haco_scores[frame_index].mean():.2f}"
                        )
                    elif mode == METRIC_DEPTH_ORDER_MODE:
                        assert metric_depth_order is not None
                        object_front_count = int(
                            np.count_nonzero(
                                metric_depth_order[frame_index]
                                == DEPTH_ORDER_OBJECT_FRONT
                            )
                        )
                        evidence_text = (
                            f"{decision_text} | metric object-front="
                            f"{object_front_count}/5"
                        )
                    elif mode == HACO_PRIORITY_MODE:
                        assert haco_priority_selected is not None
                        assert haco_priority_depth_vetoed is not None
                        selected_count = int(
                            np.count_nonzero(
                                haco_priority_selected[frame_index]
                            )
                        )
                        vetoed_count = int(
                            np.count_nonzero(
                                haco_priority_depth_vetoed[frame_index]
                            )
                        )
                        evidence_text = (
                            f"{decision_text} | HaCo-selected={selected_count}/5 "
                            f"depth-veto={vetoed_count}/5"
                        )
                    elif mode == NO_OCCLUSION_MODE:
                        evidence_text = "baseline | object never hides robot"
                    elif mode == CAMERA2_DEPTH_ONLY_MODE:
                        assert camera2_depth_only_selected is not None
                        selected_count = int(
                            np.count_nonzero(
                                camera2_depth_only_selected[frame_index]
                            )
                        )
                        evidence_text = (
                            f"{decision_text} | C2 object-front="
                            f"{selected_count}/5"
                        )
                    elif mode == VOTE_2OF3_MODE:
                        assert vote_2of3_evidence is not None
                        vote_counts = vote_2of3_evidence.positive_count[
                            frame_index
                        ]
                        evidence_text = (
                            f"{decision_text} | positive votes="
                            f"{int(vote_counts.max())}/3 max"
                        )
                    elif mode == CONFIDENCE_ENSEMBLE_MODE:
                        assert confidence_ensemble_evidence is not None
                        confidence = confidence_ensemble_evidence.score[
                            frame_index
                        ]
                        evidence_text = (
                            f"{decision_text} | confidence="
                            f"{float(confidence.max()):.2f} max"
                        )
                    else:
                        evidence_text = (
                            f"{decision_text} | "
                            f"C1={camera1_observed[frame_index].mean():.2f} "
                            f"C2={camera2_observed[frame_index].mean():.2f}"
                        )
                    if mode in comparison_modes:
                        panels.append(
                            _label_panel(
                                final,
                                title=titles[mode],
                                frame_index=frame_index,
                                evidence_text=evidence_text,
                            )
                        )
                    if mode in ABLATION_MODE_NAMES:
                        ablation_panels.append(
                            _label_panel(
                                final,
                                title=titles[mode],
                                frame_index=frame_index,
                                evidence_text=evidence_text,
                            )
                        )
                comparison_writer.write(np.concatenate(panels, axis=1))
                if ablation_comparison_writer is not None:
                    if len(ablation_panels) != len(ABLATION_MODE_NAMES):
                        raise RuntimeError(
                            "ablation comparison requires exactly four panels"
                        )
                    ablation_comparison_writer.write(
                        np.concatenate(
                            (
                                np.concatenate(ablation_panels[:2], axis=1),
                                np.concatenate(ablation_panels[2:], axis=1),
                            ),
                            axis=0,
                        )
                    )
                if (frame_index + 1) % 100 == 0:
                    print(
                        f"[stereo-composite] {frame_index + 1}/{frame_count}",
                        flush=True,
                    )
        finally:
            for writer in individual_writers.values():
                writer.release()
            comparison_writer.release()
            if ablation_comparison_writer is not None:
                ablation_comparison_writer.release()

    evidence_payload: dict[str, np.ndarray] = {
        "finger_names": np.asarray(FINGER_NAMES),
        "camera1_frame_offset": np.asarray(
            args.camera1_frame_offset,
            dtype=np.int64,
        ),
        "camera1_source_frame_index": camera1_source_frame_indices,
        "camera1_observed": camera1_observed,
        "camera2_observed": camera2_observed,
        "camera1_visible_fraction": camera1_visible_fraction,
        "camera2_visible_fraction": camera2_visible_fraction,
        "camera1_projected_vertex_count": camera1_projected_counts,
        "camera2_projected_vertex_count": camera2_projected_counts,
        "visibility_evidence": visibility_evidence,
        # ``haco_scores`` and ``object_depth_m`` are retained as fused legacy
        # aliases so downstream one-camera readers remain compatible.
        "haco_scores": haco_scores,
        "haco_scores_camera1": camera1_haco_scores,
        "haco_scores_camera2": camera2_haco_scores,
        "haco_scores_fused": haco_scores,
        "haco_active": haco_active,
        "visibility_active": active_tracks["visibility"],
        "strong_visibility_active": strong_visibility_active,
        "visibility_depth_haco_active": active_tracks[
            "visibility_depth_haco"
        ],
        "visibility_haco_active": active_tracks["visibility_depth_haco"],
        "object_depth_m": object_depth_track,
        "object_depth_m_camera1": camera1_object_depth_track,
        "object_depth_m_camera2": camera2_object_depth_track,
        "object_depth_m_fused": object_depth_track,
        "object_depth_fusion_source": object_depth_source,
        "object_depth_fusion_source_labels": DEPTH_SOURCE_LABELS,
        "object_depth_both_valid": object_depth_both_valid,
        "object_depth_cameras_agree": object_depth_cameras_agree,
        "object_depth_disagreement_m": object_depth_disagreement_m,
    }
    if metric_depth_enabled:
        assert camera1_metric_depth is not None
        assert camera2_metric_depth is not None
        assert camera1_metric_order is not None
        assert camera2_metric_order is not None
        assert metric_depth_order is not None
        assert metric_depth_order_source is not None
        evidence_payload.update(
            {
                "metric_camera1_hand_depth_m_raw": (
                    camera1_metric_depth.hand_depth_m_raw
                ),
                "metric_camera1_object_depth_m_raw": (
                    camera1_metric_depth.object_depth_m_raw
                ),
                "metric_camera1_hand_depth_m": (
                    camera1_metric_depth.hand_depth_m
                ),
                "metric_camera1_object_depth_m": (
                    camera1_metric_depth.object_depth_m
                ),
                "metric_camera1_hand_sample_count": (
                    camera1_metric_depth.hand_sample_count
                ),
                "metric_camera1_object_sample_count": (
                    camera1_metric_depth.object_sample_count
                ),
                "metric_camera2_hand_depth_m_raw": (
                    camera2_metric_depth.hand_depth_m_raw
                ),
                "metric_camera2_object_depth_m_raw": (
                    camera2_metric_depth.object_depth_m_raw
                ),
                "metric_camera2_hand_depth_m": (
                    camera2_metric_depth.hand_depth_m
                ),
                "metric_camera2_object_depth_m": (
                    camera2_metric_depth.object_depth_m
                ),
                "metric_camera2_hand_sample_count": (
                    camera2_metric_depth.hand_sample_count
                ),
                "metric_camera2_object_sample_count": (
                    camera2_metric_depth.object_sample_count
                ),
                "metric_camera1_depth_order": camera1_metric_order,
                "metric_camera2_depth_order": camera2_metric_order,
                "metric_depth_order_resolved": metric_depth_order,
                "metric_depth_order_labels": DEPTH_ORDER_LABELS,
                "metric_depth_order_source": metric_depth_order_source,
                "metric_depth_order_source_labels": DEPTH_ORDER_SOURCE_LABELS,
                "metric_depth_order_haco_selected": (
                    haco_active
                    & (metric_depth_order == DEPTH_ORDER_OBJECT_FRONT)
                ),
            }
        )
        if args.include_haco_priority:
            assert haco_priority_selected is not None
            assert haco_priority_depth_vetoed is not None
            evidence_payload.update(
                {
                    "haco_priority_selected": haco_priority_selected,
                    "haco_priority_depth_vetoed": (
                        haco_priority_depth_vetoed
                    ),
                    "haco_priority_selected_ambiguous": (
                        haco_active
                        & (metric_depth_order == DEPTH_ORDER_AMBIGUOUS)
                    ),
                    "haco_priority_selected_object_front": (
                        haco_active
                        & (metric_depth_order == DEPTH_ORDER_OBJECT_FRONT)
                    ),
                }
            )
        if args.include_ablation_modes:
            assert camera2_ablation_order is not None
            assert camera2_depth_only_selected is not None
            assert vote_2of3_evidence is not None
            assert confidence_ensemble_evidence is not None
            evidence_payload.update(
                {
                    "ablation_no_occlusion_selected": np.zeros_like(
                        camera2_depth_only_selected,
                        dtype=bool,
                    ),
                    "ablation_camera2_depth_only_selected": (
                        camera2_depth_only_selected
                    ),
                    "ablation_camera2_depth_order": camera2_ablation_order,
                    "ablation_camera2_depth_order_labels": DEPTH_ORDER_LABELS,
                    "ablation_vote_2of3_selected": (
                        vote_2of3_evidence.selected
                    ),
                    "ablation_vote_2of3_positive_count": (
                        vote_2of3_evidence.positive_count
                    ),
                    "ablation_vote_2of3_haco_positive": (
                        vote_2of3_evidence.haco_positive
                    ),
                    "ablation_vote_2of3_depth_vote_state": (
                        vote_2of3_evidence.depth_vote_state
                    ),
                    "ablation_vote_2of3_stereo_positive": (
                        vote_2of3_evidence.stereo_positive
                    ),
                    "ablation_confidence_ensemble_selected": (
                        confidence_ensemble_evidence.selected
                    ),
                    "ablation_confidence_ensemble_score": (
                        confidence_ensemble_evidence.score
                    ),
                    "ablation_confidence_direction_score": (
                        confidence_ensemble_evidence.direction_score
                    ),
                    "ablation_confidence_depth_direction": (
                        confidence_ensemble_evidence.depth_direction
                    ),
                    "ablation_confidence_stereo_direction": (
                        confidence_ensemble_evidence.stereo_direction
                    ),
                    "ablation_confidence_contact_gain": (
                        confidence_ensemble_evidence.contact_gain
                    ),
                }
            )
    np.savez_compressed(
        staging / "stereo_evidence.npz",
        **evidence_payload,
    )

    mode_statistics = {
        mode: _mode_statistics(
            mask_buffers[mode],
            finger_labels,
            output_shape=(height, width),
        )
        for mode in output_modes
    }
    subset_violations = {
        "visibility_depth_not_in_visibility": 0,
        "visibility_depth_haco_not_in_visibility_depth": 0,
    }
    for frame_index in range(frame_count):
        visibility_mask = np.asarray(mask_buffers["visibility"][frame_index])
        depth_mask = np.asarray(mask_buffers["visibility_depth"][frame_index])
        combined_mask = np.asarray(
            mask_buffers["visibility_depth_haco"][frame_index]
        )
        subset_violations["visibility_depth_not_in_visibility"] += int(
            np.logical_and(depth_mask, ~visibility_mask).sum()
        )
        subset_violations[
            "visibility_depth_haco_not_in_visibility_depth"
        ] += int(np.logical_and(combined_mask, ~depth_mask).sum())

    metric_depth_summary: dict[str, Any] = {"enabled": False}
    if metric_depth_enabled:
        assert camera1_metric_depth is not None
        assert camera2_metric_depth is not None
        assert camera1_metric_order is not None
        assert camera2_metric_order is not None
        assert metric_depth_order is not None
        assert metric_depth_order_source is not None
        selected = haco_active & (
            metric_depth_order == DEPTH_ORDER_OBJECT_FRONT
        )
        metric_depth_summary = {
            "enabled": True,
            "camera1_npz": str(args.camera1_metric_depth_npz.resolve()),
            "camera2_npz": str(args.camera2_metric_depth_npz.resolve()),
            "input_contract": {
                "shape": [frame_count, len(FINGER_NAMES)],
                "depth_unit": "meter",
                "depth_axis": "native RGB camera z; smaller is closer",
                "required_keys": [
                    "hand_depth_m",
                    "object_depth_m",
                    "hand_sample_count",
                    "object_sample_count",
                ],
            },
            "valid_local_depth_pairs": {
                "camera1": int(
                    np.count_nonzero(
                        np.isfinite(camera1_metric_depth.hand_depth_m)
                        & np.isfinite(camera1_metric_depth.object_depth_m)
                    )
                ),
                "camera2": int(
                    np.count_nonzero(
                        np.isfinite(camera2_metric_depth.hand_depth_m)
                        & np.isfinite(camera2_metric_depth.object_depth_m)
                    )
                ),
            },
            "order_counts": {
                "camera1": {
                    label: int(np.count_nonzero(camera1_metric_order == code))
                    for code, label in enumerate(DEPTH_ORDER_LABELS.tolist())
                },
                "camera2": {
                    label: int(np.count_nonzero(camera2_metric_order == code))
                    for code, label in enumerate(DEPTH_ORDER_LABELS.tolist())
                },
                "resolved": {
                    label: int(np.count_nonzero(metric_depth_order == code))
                    for code, label in enumerate(DEPTH_ORDER_LABELS.tolist())
                },
            },
            "source_counts": {
                label: int(np.count_nonzero(metric_depth_order_source == code))
                for code, label in enumerate(
                    DEPTH_ORDER_SOURCE_LABELS.tolist()
                )
            },
            "haco_selected_object_front_finger_frames": int(selected.sum()),
            "haco_selected_object_front_frames": int(
                np.any(selected, axis=1).sum()
            ),
            "per_finger_selected_frames": {
                finger: int(selected[:, index].sum())
                for index, finger in enumerate(FINGER_NAMES)
            },
        }

    haco_priority_summary: dict[str, Any] = {"enabled": False}
    if args.include_haco_priority:
        assert metric_depth_order is not None
        assert haco_priority_selected is not None
        assert haco_priority_depth_vetoed is not None
        selected_ambiguous = haco_active & (
            metric_depth_order == DEPTH_ORDER_AMBIGUOUS
        )
        selected_object_front = haco_active & (
            metric_depth_order == DEPTH_ORDER_OBJECT_FRONT
        )
        haco_priority_summary = {
            "enabled": True,
            "policy": (
                "max-fused temporally active HaCo is primary; resolved "
                "hand_front vetoes, object_front and ambiguous are selected"
            ),
            "provenance": {
                "contact_fusion": (
                    "per-finger maximum of finite camera1/camera2 HaCo scores"
                ),
                "camera1_contact_dir": (
                    str(args.camera1_contact_dir.resolve())
                    if args.camera1_contact_dir is not None
                    else None
                ),
                "camera2_contact_dir": str(args.contact_dir.resolve()),
                "resolved_metric_depth": "metric_depth_order_resolved",
                "camera1_metric_depth_npz": str(
                    args.camera1_metric_depth_npz.resolve()
                ),
                "camera2_metric_depth_npz": str(
                    args.camera2_metric_depth_npz.resolve()
                ),
                "object_mask": str(args.object_mask.resolve()),
                "robot_semantics": str(
                    (overlay_dir / "robot_finger_labels.npy").resolve()
                ),
            },
            "contact_active_finger_frames": int(haco_active.sum()),
            "selected_finger_frames": int(haco_priority_selected.sum()),
            "selected_frames": int(
                np.any(haco_priority_selected, axis=1).sum()
            ),
            "selected_by_depth_order": {
                "ambiguous": int(selected_ambiguous.sum()),
                "object_front": int(selected_object_front.sum()),
            },
            "depth_vetoed_hand_front_finger_frames": int(
                haco_priority_depth_vetoed.sum()
            ),
            "depth_vetoed_hand_front_frames": int(
                np.any(haco_priority_depth_vetoed, axis=1).sum()
            ),
            "per_finger": {
                finger: {
                    "contact_active_frames": int(haco_active[:, index].sum()),
                    "selected_frames": int(
                        haco_priority_selected[:, index].sum()
                    ),
                    "selected_ambiguous_frames": int(
                        selected_ambiguous[:, index].sum()
                    ),
                    "selected_object_front_frames": int(
                        selected_object_front[:, index].sum()
                    ),
                    "depth_vetoed_hand_front_frames": int(
                        haco_priority_depth_vetoed[:, index].sum()
                    ),
                }
                for index, finger in enumerate(FINGER_NAMES)
            },
        }

    ablation_summary: dict[str, Any] = {"enabled": False}
    if args.include_ablation_modes:
        assert camera2_ablation_order is not None
        assert camera2_depth_only_selected is not None
        assert vote_2of3_evidence is not None
        assert confidence_ensemble_evidence is not None
        confidence_score = confidence_ensemble_evidence.score
        ablation_summary = {
            "enabled": True,
            "comparison_video": "video_compare_ablation_modes_2x2.mp4",
            "comparison_layout": [
                [NO_OCCLUSION_MODE, CAMERA2_DEPTH_ONLY_MODE],
                [VOTE_2OF3_MODE, CONFIDENCE_ENSEMBLE_MODE],
            ],
            "camera2_depth_only": {
                "depth_separation_margin_m": (
                    ablation_config.depth_separation_margin_m
                ),
                "order_counts": {
                    label: int(
                        np.count_nonzero(camera2_ablation_order == code)
                    )
                    for code, label in enumerate(DEPTH_ORDER_LABELS.tolist())
                },
                "selected_finger_frames": int(
                    camera2_depth_only_selected.sum()
                ),
                "selected_frames": int(
                    np.any(camera2_depth_only_selected, axis=1).sum()
                ),
            },
            "vote_2of3": {
                "denominator_policy": (
                    "fixed denominator of 3; ambiguous/missing depth abstains "
                    "as zero without shrinking the denominator; inactive or "
                    "missing HaCo/stereo is non-positive"
                ),
                "required_positive_votes": 2,
                "panel_members": [
                    "dual-camera max-fused temporal HaCo active",
                    "camera2 native metric depth object_front",
                    "strong temporal C1-visible/C2-hidden active",
                ],
                "selected_finger_frames": int(
                    vote_2of3_evidence.selected.sum()
                ),
                "selected_frames": int(
                    np.any(vote_2of3_evidence.selected, axis=1).sum()
                ),
                "positive_count_histogram": {
                    str(count): int(
                        np.count_nonzero(
                            vote_2of3_evidence.positive_count == count
                        )
                    )
                    for count in range(4)
                },
                "depth_vote_state_counts": {
                    "hand_front_negative": int(
                        np.count_nonzero(
                            vote_2of3_evidence.depth_vote_state == -1
                        )
                    ),
                    "ambiguous_or_missing_abstain": int(
                        np.count_nonzero(
                            vote_2of3_evidence.depth_vote_state == 0
                        )
                    ),
                    "object_front_positive": int(
                        np.count_nonzero(
                            vote_2of3_evidence.depth_vote_state == 1
                        )
                    ),
                },
                "dataset_caveat": {
                    "all_positive_depth_or_stereo_have_active_haco": bool(
                        not np.any(
                            (
                                (vote_2of3_evidence.depth_vote_state > 0)
                                | vote_2of3_evidence.stereo_positive
                            )
                            & ~vote_2of3_evidence.haco_positive
                        )
                    ),
                    "interpretation_if_true": (
                        "On this dataset the 2-of-3 result numerically "
                        "collapses to depth-positive OR stereo-positive, "
                        "because active HaCo accompanies every positive "
                        "directional cue; this is a dataset observation, not "
                        "a change to the fixed-panel rule."
                    ),
                },
            },
            "confidence_ensemble": {
                "config": asdict(ablation_config),
                "formula": {
                    "depth_direction": (
                        "sign(hand_z-object_z) * clipped linear ramp from "
                        "ablation depth_separation_margin_m to "
                        "confidence_depth_saturation_m"
                    ),
                    "stereo_direction": (
                        "clipped linear ramp from confidence_stereo_start "
                        "to confidence_stereo_saturation for "
                        "C1_observed*(1-C2_observed)"
                    ),
                    "direction": (
                        "normalized weighted mean of depth_direction and "
                        "stereo_direction"
                    ),
                    "contact_gain": (
                        "confidence_contact_floor + "
                        "(1-confidence_contact_floor)*max_fused_HaCo"
                    ),
                    "score": "max(direction,0)*contact_gain",
                    "selection": "score >= confidence_score_threshold",
                },
                "selected_finger_frames": int(
                    confidence_ensemble_evidence.selected.sum()
                ),
                "selected_frames": int(
                    np.any(
                        confidence_ensemble_evidence.selected,
                        axis=1,
                    ).sum()
                ),
                "score_min": float(confidence_score.min()),
                "score_mean": float(confidence_score.mean()),
                "score_max": float(confidence_score.max()),
            },
        }

    report = {
        "schema_version": 9,
        "camera2_is_final_view": True,
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "temporal_alignment": {
            "reference_view": "camera2/output",
            "camera1_frame_offset": int(args.camera1_frame_offset),
            "camera1_lookup": (
                "camera1_source_index = camera2_frame_index + "
                f"({args.camera1_frame_offset})"
            ),
            "camera1_source_frame_index": (
                camera1_source_frame_indices.tolist()
            ),
            "camera1_out_of_range_frames": int(
                np.count_nonzero(camera1_source_frame_indices < 0)
            ),
            "out_of_range_policy": "fail_open",
        },
        "finger_names": list(FINGER_NAMES),
        "output_modes": list(output_modes),
        "comparison_modes": list(comparison_modes),
        "config": asdict(config),
        "definitions": {
            "object_restore": (
                "camera2 raw RGB is restored only inside object_restore_mask; "
                "all geometry and occlusion decisions continue to use the "
                "modal object_mask"
            ),
            "visibility": (
                "camera1 observed AND camera2 not observed, limited to the "
                "camera2 modal object mask"
            ),
            "visibility_depth": (
                "visibility plus robot depth behind camera-2-primary robust "
                "registered object depth; camera 1 is admitted only when the "
                "two estimates agree"
            ),
            "visibility_depth_haco": (
                "strong stereo+depth evidence is always kept; max-fused "
                "per-finger HaCo from the available cameras only "
                "admits/stabilizes evidence between assisted_visibility_on "
                "and visibility_on"
            ),
            HACO_ONLY_MODE: (
                "max-fused camera1/camera2 HaCo active fingers intersected "
                "only with camera2 object mask and robot finger semantics"
            ),
            VISIBILITY_HACO_MODE: (
                "RGB-only direction from camera1-visible/camera2-hidden; "
                "strong stereo is retained and assisted stereo requires "
                "max-fused per-finger HaCo, limited to the camera2 object "
                "mask and robot finger semantics"
            ),
            METRIC_DEPTH_ORDER_MODE: (
                "C2-primary local metric hand/object ordering; temporally "
                "active fused HaCo selects fingers, stereo visibility only "
                "assists C2-ambiguous order"
            ),
            HACO_PRIORITY_MODE: (
                "max-fused temporally active camera1/camera2 HaCo is primary; "
                "resolved metric hand_front alone vetoes hiding, while "
                "object_front and ambiguous permit hiding inside the camera2 "
                "modal object mask and semantic robot fingers"
            ),
        },
        "decision_rule": {
            "visibility_evidence": "C1_observed * (1 - C2_observed)",
            "modal_observation": (
                "HaWoR detector confidence * fraction of projected MANO "
                "finger vertices supported by the SAM visible mask"
            ),
            "visibility_and_depth_baselines": (
                "temporal_hysteresis(evidence, assisted_visibility_on/off)"
            ),
            "visibility_depth_haco": (
                "strong_visibility OR (assisted_visibility AND haco_active)"
            ),
            HACO_ONLY_MODE: (
                "object_mask AND robot_finger_semantics AND "
                "temporal_hysteresis(max(camera1_haco,camera2_haco))"
            ),
            VISIBILITY_HACO_MODE: (
                "object_mask AND robot_finger_semantics AND "
                "(strong_stereo OR (assisted_stereo AND "
                "temporal_hysteresis(max(camera1_haco,camera2_haco))))"
            ),
            METRIC_DEPTH_ORDER_MODE: (
                "object_mask AND robot_finger_semantics AND haco_active AND "
                "(resolved_camera2_order == object_front)"
            ),
            HACO_PRIORITY_MODE: (
                "object_mask AND robot_finger_semantics AND haco_active AND "
                "(resolved_camera2_order != hand_front)"
            ),
            "metric_depth_order_resolver": (
                "C2 object_front always wins; C2 hand_front wins unless "
                "C1 hand_front plus strong C1-visible/C2-hidden evidence "
                "contradicts it; strong C1-visible/C2-hidden evidence promotes "
                "C2 ambiguous regardless of C1 metric order"
            ),
            "strong_visibility_thresholds": {
                "on": config.visibility_on,
                "off": config.visibility_off,
            },
            "assisted_visibility_thresholds": {
                "on": config.assisted_visibility_on,
                "off": config.assisted_visibility_off,
            },
            "haco_thresholds": {
                "on": config.haco_on,
                "off": config.haco_off,
            },
            "haco_fusion": (
                "per-finger maximum of finite camera1 and camera2 scores; "
                "missing camera/view contributes no evidence"
            ),
            "object_depth_fusion": (
                "both valid and within tolerance -> max(camera1,camera2); "
                "disagreement -> camera2; camera2 only -> camera2; "
                "camera1 only or neither -> NaN/fail-open"
            ),
            "object_depth_agreement_tolerance_m": (
                config.depth_agreement_tolerance_m
            ),
            "object_depth_coordinate_system": "camera2 output pixels and metric z",
        },
        "sources": {
            "camera1_rgb_dir": str(camera1_rgb),
            "camera2_rgb_dir": str(camera2_rgb),
            "camera1_hawor": str(camera1_hawor),
            "camera2_hawor": str(camera2_hawor),
            "camera1_tracks": str(camera1_tracks) if camera1_tracks else None,
            "camera2_tracks": str(camera2_tracks) if camera2_tracks else None,
            "camera1_observation_source": camera1_observation_source,
            "camera2_observation_source": camera2_observation_source,
            "camera1_visible_mask": (
                str(args.camera1_visible_mask.resolve())
                if args.camera1_visible_mask is not None
                else None
            ),
            "camera2_visible_mask": (
                str(args.camera2_visible_mask.resolve())
                if args.camera2_visible_mask is not None
                else None
            ),
            "camera1_contact_dir": (
                str(args.camera1_contact_dir.resolve())
                if args.camera1_contact_dir is not None
                else None
            ),
            "camera2_contact_dir": str(args.contact_dir.resolve()),
            "contact_dir": str(args.contact_dir.resolve()),
            "camera1_metric_depth_npz": (
                str(args.camera1_metric_depth_npz.resolve())
                if args.camera1_metric_depth_npz is not None
                else None
            ),
            "camera2_metric_depth_npz": (
                str(args.camera2_metric_depth_npz.resolve())
                if args.camera2_metric_depth_npz is not None
                else None
            ),
            "background": str(args.background.resolve()),
            "overlay_dir": str(overlay_dir),
            "object_mask": str(object_mask_path),
            "object_restore_mask": str(object_restore_mask_path),
            "object_depth_mask": (
                str(args.object_depth_mask.resolve())
                if args.object_depth_mask is not None
                else str(args.object_mask.resolve())
            ),
            "scene_depth_camera1": (
                str(args.scene_depth_camera1.resolve())
                if args.scene_depth_camera1 is not None
                else None
            ),
            "scene_depth_camera2": (
                str(args.scene_depth.resolve())
                if args.scene_depth is not None
                else None
            ),
            "scene_depth": (
                str(args.scene_depth.resolve())
                if args.scene_depth is not None
                else None
            ),
        },
        "sides": {
            "rendered": rendered_side,
            "camera1": camera1_side,
            "camera2": camera2_side,
        },
        "missing_contact_frames": camera2_missing_contact_frames,
        "missing_contact_frames_by_camera": {
            "camera1": camera1_missing_contact_frames,
            "camera2": camera2_missing_contact_frames,
        },
        "haco_coverage": {
            "camera1_enabled": args.camera1_contact_dir is not None,
            "camera2_enabled": True,
            "camera1_loaded_frames": (
                frame_count - camera1_missing_contact_frames
                if args.camera1_contact_dir is not None
                else 0
            ),
            "camera2_loaded_frames": frame_count - camera2_missing_contact_frames,
            "fusion_rule": "per-finger maximum of available finite view scores",
        },
        "metric_depth_order": metric_depth_summary,
        "haco_priority": haco_priority_summary,
        "projected_visibility_coverage": {
            "camera1_known_finger_frames": {
                finger: int(np.isfinite(camera1_visible_fraction[:, index]).sum())
                for index, finger in enumerate(FINGER_NAMES)
            },
            "camera2_known_finger_frames": {
                finger: int(np.isfinite(camera2_visible_fraction[:, index]).sum())
                for index, finger in enumerate(FINGER_NAMES)
            },
        },
        "valid_object_depth_frames": int(np.isfinite(object_depth_track).sum()),
        "valid_object_depth_frames_by_camera": {
            "camera1": int(np.isfinite(camera1_object_depth_track).sum()),
            "camera2": int(np.isfinite(camera2_object_depth_track).sum()),
            "fused": int(np.isfinite(object_depth_track).sum()),
        },
        "object_depth_fusion_coverage": {
            label: int(np.count_nonzero(object_depth_source == code))
            for code, label in enumerate(DEPTH_SOURCE_LABELS.tolist())
        },
        "object_depth_fusion_rule": (
            "C2 primary: agreeing pair -> farther/max; disagreement or C2-only "
            "-> C2; C1-only or neither -> fail-open"
        ),
        "object_depth_agreement": {
            "tolerance_m": config.depth_agreement_tolerance_m,
            "both_valid_frames": int(object_depth_both_valid.sum()),
            "agree_frames": int(object_depth_cameras_agree.sum()),
            "camera1_rejected_disagreement_frames": int(
                np.count_nonzero(
                    object_depth_source
                    == DEPTH_SOURCE_CAMERA2_REJECTED_CAMERA1
                )
            ),
            "camera1_only_unsupported_frames": int(
                np.count_nonzero(
                    object_depth_source == DEPTH_SOURCE_CAMERA1_UNSUPPORTED
                )
            ),
        },
        "object_depth_m_camera1": [
            float(value) if np.isfinite(value) else None
            for value in camera1_object_depth_track
        ],
        "object_depth_m_camera2": [
            float(value) if np.isfinite(value) else None
            for value in camera2_object_depth_track
        ],
        "object_depth_m": [
            float(value) if np.isfinite(value) else None
            for value in object_depth_track
        ],
        "visibility_active_runs": {
            finger: _true_runs(active_tracks["visibility"][:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "strong_visibility_active_runs": {
            finger: _true_runs(strong_visibility_active[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "haco_active_runs": {
            finger: _true_runs(haco_active[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "stable_pixel_candidate_runs": {
            mode: {
                finger: _true_runs(stable_presence[mode][:, index])
                for index, finger in enumerate(FINGER_NAMES)
            }
            for mode in output_modes
        },
        "mode_statistics": mode_statistics,
        "raw_object_pixels_total": int(raw_object_pixel_count.sum()),
        "object_pixel_counts": {
            "modal_geometry_pixels_total": int(modal_object_pixel_count.sum()),
            "raw_rgb_restore_pixels_total": int(raw_object_pixel_count.sum()),
            "modal_pixels_not_restored_total": int(
                modal_object_pixel_count.sum() - raw_object_pixel_count.sum()
            ),
            "restore_pixels_outside_modal_total": 0,
        },
        "object_restore_policy": {
            "explicit_mask_supplied": args.object_restore_mask is not None,
            "defaulted_to_modal_object_mask": args.object_restore_mask is None,
            "geometry_source": "sources.object_mask",
            "raw_rgb_restore_source": "sources.object_restore_mask",
        },
        "subset_violations": subset_violations,
        "invariants": {
            "occlusion_is_finger_only": True,
            "explicit_object_mask_must_be_modal": True,
            "geometry_uses_modal_object_mask": True,
            "raw_rgb_restore_uses_object_restore_mask_only": True,
            "object_restore_mask_matches_modal_shape_and_dtype": True,
            "object_restore_mask_is_subset_of_modal_object_mask": True,
            "object_restore_default_preserves_legacy_behavior": True,
            "missing_depth_fails_open_for_depth_modes": True,
            "missing_contact_never_removes_strong_stereo_evidence": True,
            "missing_contact_rejects_only_ambiguous_stereo_evidence": True,
            "dual_haco_uses_max_available_score": True,
            "haco_only_enabled": args.include_haco_only,
            "haco_only_ignores_visibility_and_depth": True,
            "metric_depth_order_enabled": metric_depth_enabled,
            "metric_depth_order_is_camera2_primary": True,
            "metric_depth_order_c2_object_front_is_authoritative": True,
            "metric_depth_order_c2_hand_front_can_be_overridden_only_by_"
            "stereo_contradiction": True,
            "metric_depth_order_haco_is_selector_only": True,
            "metric_depth_order_visibility_never_overrides_c2_object_front": True,
            "metric_depth_order_visibility_applies_only_to_ambiguity_or_"
            "strict_hand_front_contradiction": True,
            "metric_depth_order_ambiguous_fails_open": True,
            "haco_priority_enabled": args.include_haco_priority,
            "haco_priority_requires_dual_metric_depth_inputs": True,
            "haco_priority_uses_dual_camera_max_fused_temporal_haco": True,
            "haco_priority_ambiguous_depth_allows_active_haco": True,
            "haco_priority_only_hand_front_vetoes_active_haco": True,
            "haco_priority_missing_haco_fails_open": True,
            "haco_priority_is_limited_to_camera2_modal_object_mask": True,
            "haco_priority_is_limited_to_semantic_robot_fingers": True,
            "dual_depth_is_camera2_primary": True,
            "camera1_depth_requires_cross_view_agreement": True,
            "camera1_only_depth_fails_open": True,
            "modes_are_nested": all(value == 0 for value in subset_violations.values()),
        },
        "compositing_order": (
            "camera2_inpainted_background_then_camera2_raw_pixels_within_"
            "object_restore_mask_then_robot"
        ),
    }
    if args.include_ablation_modes:
        report["schema_version"] = 10
        report["ablation_comparison_modes"] = list(ABLATION_MODE_NAMES)
        report["ablation_modes"] = ablation_summary
        report["definitions"].update(
            {
                NO_OCCLUSION_MODE: (
                    "explicit baseline whose occlusion mask is always false"
                ),
                CAMERA2_DEPTH_ONLY_MODE: (
                    "native camera2 metric object_front fingers only; no "
                    "HaCo, camera1 metric order, or stereo assistance"
                ),
                VOTE_2OF3_MODE: (
                    "at least two positive votes from fixed HaCo, camera2 "
                    "metric-order, and strong stereo panel"
                ),
                CONFIDENCE_ENSEMBLE_MODE: (
                    "continuous direction from signed camera2 metric depth "
                    "and stereo visibility, multiplied by a HaCo-derived "
                    "contact-confidence gain"
                ),
            }
        )
        report["decision_rule"].update(
            {
                NO_OCCLUSION_MODE: "occluded_finger_mask = false everywhere",
                CAMERA2_DEPTH_ONLY_MODE: (
                    "object_mask AND robot_finger_semantics AND "
                    "(camera2_metric_order == object_front)"
                ),
                VOTE_2OF3_MODE: (
                    "object_mask AND robot_finger_semantics AND "
                    "(positive(HaCo_active) + positive(C2_object_front) + "
                    "positive(strong_stereo_active) >= 2); denominator remains "
                    "3 when depth is ambiguous/missing"
                ),
                CONFIDENCE_ENSEMBLE_MODE: (
                    "object_mask AND robot_finger_semantics AND "
                    "max(weighted_depth_plus_stereo_direction,0) * "
                    "HaCo_contact_gain >= confidence_score_threshold"
                ),
            }
        )
        report["invariants"].update(
            {
                "ablation_modes_enabled": True,
                "legacy_three_panel_comparison_unchanged": True,
                "ablation_comparison_is_separate_2x2": True,
                "no_occlusion_mask_is_always_false": True,
                "camera2_depth_only_uses_no_haco_or_camera1_cue": True,
                "vote_2of3_uses_fixed_denominator_three": True,
                "vote_2of3_requires_two_positive_votes": True,
                "confidence_haco_cannot_create_front_back_direction": True,
                "all_ablation_removal_is_limited_to_camera2_object_mask": True,
                "all_ablation_removal_is_limited_to_semantic_fingers": True,
            }
        )
    (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    for buffer in mask_buffers.values():
        buffer.flush()
    del mask_buffers
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] stereo occlusion comparison: {output_dir}", flush=True)
    for mode in output_modes:
        statistics = mode_statistics[mode]
        print(
            f"[info] {mode}: pixels={statistics['pixels']}, "
            f"frames={statistics['frames']}/{frame_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
