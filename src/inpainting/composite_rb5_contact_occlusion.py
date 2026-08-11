"""Contact-conditioned, finger-only object occlusion for RB5 + XHand.

The baseline compositor draws every robot pixel over the inpainted scene.  It
therefore makes a grasp look wrong whenever an object should cover the far
side of a robot finger.  This compositor changes only pixels that satisfy all
of the following:

1. Isaac semantic rendering says the pixel belongs to an XHand finger.
2. HaCo assigns high contact probability to the corresponding MANO finger.
   An optional synchronized auxiliary camera can raise this confidence through
   per-finger maximum fusion.
3. Contact-local image evidence says that part of the human finger was hidden.
4. In HaCo mode the robot pixel is behind the projected HaCo contact surface.
   An opt-in per-finger XHand transverse-thickness bias can conservatively
   test the back of the robot finger instead of only its rendered front face.
   In ensemble mode it must additionally pass the sensor object-depth gate.
   In object3d mode it must pass a dense visible-object surface gate.  That
   surface can be locally translated (within a bounded tolerance) so the
   estimated object model meets the MH HaCo contact points.

An explicit modal object mask and aligned scene depth are preferred.
Auxiliary HaCo never supplies image coordinates or depth: the final/primary
camera remains authoritative for projected contact points and surface depth,
so the dual-view contact pilot does not pretend to have stereo calibration.
For datasets that do not have those products yet, a conservative proxy is
built from the projected HaWoR hand, visible-hand mask, raw frame, and
hand-inpainted background. Ambiguous pixels fail open: the robot remains
visible. Temporal hysteresis prevents one-frame layer flicker. The HaCo-free
sensor-depth baseline lives in ``composite_rb5_depth_occlusion.py``.

Outputs are published atomically under ``<processed_demo>/contact_occlusion``:

    video_overlay_contact.mp4
    video_robot_only_contact.mp4
    debug_contact_occlusion.mp4
    occluded_finger_mask.npy
    report.json
"""

from __future__ import annotations

import argparse
import atexit
import functools
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from atomic_directory_publish import publish_directory

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_PARTS = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
}
XHAND_THUMB_THICKNESS_M = 0.03916
XHAND_FINGER_THICKNESS_M = 0.02930


@dataclass(frozen=True)
class OcclusionConfig:
    contact_score_threshold: float = 0.72
    contact_point_threshold: float = 0.78
    contact_top_fraction: float = 0.25
    min_contact_points: int = 6
    contact_radius_px: int = 22
    contact_interior_expand_px: int = 0
    contact_interior_expand_cap_fraction: float = 0.25
    point_probe_radius_px: int = 4
    hidden_fraction_on: float = 0.42
    hidden_fraction_off: float = 0.22
    min_on_frames: int = 2
    hold_frames: int = 3
    min_occlusion_run_frames: int = 2
    raw_bg_lab_threshold: float = 15.0
    object_mask_dilate_px: int = 5
    object_depth_erode_px: int = 7
    object_depth_margin_m: float = 0.010
    object_surface_min_samples: int = 12
    object_surface_contact_max_shift_m: float = 0.060
    object_surface_contact_consistency_m: float = 0.060
    object3d_force_surface: bool = False
    object3d_force_margin_m: float = 0.0
    object3d_temporal_max_gap_frames: int = 0
    object3d_temporal_motion_px: int = 6
    object3d_temporal_front_slack_m: float = 0.015
    contact_depth_tolerance_m: float = 0.012
    contact_depth_thickness_scale: float = 0.0
    xhand_thumb_thickness_m: float = XHAND_THUMB_THICKNESS_M
    xhand_finger_thickness_m: float = XHAND_FINGER_THICKNESS_M
    robot_edge_sigma_px: float = 0.6
    occlusion_edge_sigma_px: float = 1.2

    def contact_depth_bias_m(self, finger: str) -> float:
        """Return the configured contact-proxy depth bias for one finger."""
        if finger not in FINGER_NAMES:
            raise ValueError(f"unknown finger {finger!r}")
        thickness = (
            self.xhand_thumb_thickness_m
            if finger == "thumb"
            else self.xhand_finger_thickness_m
        )
        return float(self.contact_depth_thickness_scale * thickness)

    def validate(self) -> None:
        probability_fields = (
            self.contact_score_threshold,
            self.contact_point_threshold,
            self.contact_top_fraction,
            self.contact_interior_expand_cap_fraction,
            self.hidden_fraction_on,
            self.hidden_fraction_off,
        )
        if any(not 0.0 <= value <= 1.0 for value in probability_fields):
            raise ValueError("probability/fraction settings must be in [0,1]")
        if self.hidden_fraction_off > self.hidden_fraction_on:
            raise ValueError("hidden_fraction_off must not exceed on threshold")
        if self.min_contact_points <= 0:
            raise ValueError("min_contact_points must be positive")
        if (
            self.min_on_frames <= 0
            or self.hold_frames < 0
            or self.min_occlusion_run_frames <= 0
        ):
            raise ValueError("invalid temporal hysteresis settings")
        if (
            self.contact_radius_px <= 0
            or self.contact_interior_expand_px < 0
            or self.point_probe_radius_px < 0
            or self.object_mask_dilate_px < 0
            or self.object_depth_erode_px < 0
        ):
            raise ValueError("pixel radii must be non-negative")
        if (
            self.object_depth_margin_m < 0.0
            or self.contact_depth_tolerance_m < 0.0
        ):
            raise ValueError("depth margins must be non-negative")
        if self.object_surface_min_samples <= 0:
            raise ValueError("object surface minimum samples must be positive")
        object_surface_scales = (
            self.object_surface_contact_max_shift_m,
            self.object_surface_contact_consistency_m,
            self.object3d_force_margin_m,
            self.object3d_temporal_front_slack_m,
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in object_surface_scales
        ):
            raise ValueError(
                "object surface contact scales must be finite and non-negative"
            )
        if (
            self.object_surface_contact_max_shift_m
            > self.object_surface_contact_consistency_m
        ):
            raise ValueError(
                "object surface max shift must not exceed consistency tolerance"
            )
        if (
            self.object3d_temporal_max_gap_frames < 0
            or self.object3d_temporal_motion_px < 0
        ):
            raise ValueError("invalid Object3D temporal filter settings")
        thickness_values = (
            self.contact_depth_thickness_scale,
            self.xhand_thumb_thickness_m,
            self.xhand_finger_thickness_m,
        )
        if any(not math.isfinite(value) for value in thickness_values):
            raise ValueError("XHand thickness settings must be finite")
        if self.contact_depth_thickness_scale < 0.0:
            raise ValueError("contact depth thickness scale must be non-negative")
        if (
            self.xhand_thumb_thickness_m <= 0.0
            or self.xhand_finger_thickness_m <= 0.0
        ):
            raise ValueError("XHand full thicknesses must be positive")
        if any(
            not math.isfinite(self.contact_depth_bias_m(finger))
            for finger in FINGER_NAMES
        ):
            raise ValueError("scaled XHand contact depth bias must be finite")


def temporal_hysteresis(
    evidence: np.ndarray,
    *,
    on_threshold: float,
    off_threshold: float,
    min_on_frames: int,
    hold_frames: int,
) -> np.ndarray:
    """Turn a noisy confidence track into a stable boolean state track."""
    values = np.asarray(evidence, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"evidence must be one-dimensional, got {values.shape}")
    if not 0.0 <= off_threshold <= on_threshold <= 1.0:
        raise ValueError("expected 0 <= off <= on <= 1")
    if min_on_frames <= 0 or hold_frames < 0:
        raise ValueError("invalid hysteresis frame counts")

    active = np.zeros(len(values), dtype=bool)
    state = False
    above = 0
    below = 0
    for index, value in enumerate(values):
        value = float(value) if np.isfinite(value) else 0.0
        if not state:
            above = above + 1 if value >= on_threshold else 0
            if above >= min_on_frames:
                state = True
                start = index - min_on_frames + 1
                active[start:index + 1] = True
                below = 0
        else:
            if value < off_threshold:
                below += 1
                if below > hold_frames:
                    state = False
                    above = 0
                    below = 0
            else:
                below = 0
            if state:
                active[index] = True
    return active


def suppress_short_runs(
    values: np.ndarray,
    *,
    min_frames: int,
) -> np.ndarray:
    """Remove short true runs independently from each boolean track."""
    tracks = np.asarray(values, dtype=bool)
    was_vector = tracks.ndim == 1
    if was_vector:
        tracks = tracks[:, None]
    if tracks.ndim != 2:
        raise ValueError(f"tracks must have shape (T,) or (T,N), got {tracks.shape}")
    if min_frames <= 0:
        raise ValueError("min_frames must be positive")
    output = tracks.copy()
    for column in range(tracks.shape[1]):
        changes = np.diff(
            np.r_[False, tracks[:, column], False].astype(np.int8)
        )
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends):
            if end - start < min_frames:
                output[start:end, column] = False
    return output[:, 0] if was_vector else output


def project_camera_points(
    points_cam: np.ndarray,
    *,
    focal_px: float,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project CV-camera XYZ points and return (uv, valid_depth)."""
    points = np.asarray(points_cam, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-5)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    if valid.any():
        p = points[valid]
        uv[valid, 0] = focal_px * p[:, 0] / p[:, 2] + image_width / 2.0
        uv[valid, 1] = focal_px * p[:, 1] / p[:, 2] + image_height / 2.0
    return uv, valid


def disk_support(
    points_uv: np.ndarray,
    shape: tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    """Rasterize a union of contact-centred disks."""
    height, width = shape
    out = np.zeros((height, width), dtype=np.uint8)
    for point in np.asarray(points_uv, dtype=np.float32):
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if x < -radius_px or x >= width + radius_px:
            continue
        if y < -radius_px or y >= height + radius_px:
            continue
        cv2.circle(out, (x, y), radius_px, 1, thickness=-1)
    return out.astype(bool)


def sample_local_fraction(
    mask: np.ndarray,
    points_uv: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    """Measure binary-mask support around each projected point."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {binary.shape}")
    height, width = binary.shape
    fractions = np.zeros(len(points_uv), dtype=np.float32)
    for index, point in enumerate(np.asarray(points_uv, dtype=np.float32)):
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        x0, x1 = max(0, x - radius_px), min(width, x + radius_px + 1)
        y0, y1 = max(0, y - radius_px), min(height, y + radius_px + 1)
        if x0 < x1 and y0 < y1:
            fractions[index] = float(binary[y0:y1, x0:x1].mean())
    return fractions


def skin_mask_bgr(frame: np.ndarray) -> np.ndarray:
    """Conservative skin-colour cue used only by the no-object-mask fallback."""
    image = np.asarray(frame, dtype=np.uint8)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, value = cv2.split(hsv)
    ycc_skin = (cr >= 132) & (cr <= 178) & (cb >= 72) & (cb <= 135)
    hsv_skin = (
        ((hue <= 25) | (hue >= 170))
        & (sat >= 15)
        & (sat <= 210)
        & (value >= 45)
    )
    mask = (ycc_skin & hsv_skin).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)


def lab_frame_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape:
        raise ValueError(f"frame shape mismatch: {first.shape} vs {second.shape}")
    first_lab = cv2.cvtColor(first, cv2.COLOR_BGR2LAB).astype(np.float32)
    second_lab = cv2.cvtColor(second, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.linalg.norm(first_lab - second_lab, axis=2)


def proxy_occluder_mask(
    raw_frame: np.ndarray,
    inpainted_frame: np.ndarray,
    hawor_amodal_mask: np.ndarray,
    visible_hand_mask: np.ndarray,
    *,
    lab_threshold: float,
) -> np.ndarray:
    """Estimate only the hand-local pixels likely showing an occluding object."""
    amodal = np.asarray(hawor_amodal_mask, dtype=bool)
    visible = np.asarray(visible_hand_mask, dtype=bool)
    if amodal.shape != raw_frame.shape[:2] or visible.shape != amodal.shape:
        raise ValueError("proxy visibility inputs are not frame-aligned")
    delta = lab_frame_delta(raw_frame, inpainted_frame)
    non_skin = ~skin_mask_bgr(raw_frame)
    # Either the background was preserved photometrically or SAM did not mark
    # the pixel as visible human. Requiring non-skin and the HaWoR full-hand
    # silhouette prevents the table/background from becoming a global layer.
    candidate = (
        amodal
        & non_skin
        & ((delta <= lab_threshold) | ~visible)
    ).astype(np.uint8)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, open_kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_kernel)
    return candidate.astype(bool)


def estimate_object_depth_track(
    scene_depth: np.ndarray,
    object_mask: np.ndarray,
    *,
    output_shape: tuple[int, int],
    erode_px: int,
    min_samples: int = 30,
) -> np.ndarray:
    """Robust per-frame scalar object depth; missing frames remain NaN."""
    depth = scene_depth
    masks = object_mask
    if depth.ndim != 3 or masks.ndim != 3 or len(depth) != len(masks):
        raise ValueError("scene depth/object mask must be aligned (T,H,W) arrays")
    height, width = output_shape
    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erode_px + 1, 2 * erode_px + 1),
        )
        if erode_px > 0
        else None
    )
    result = np.full(len(depth), np.nan, dtype=np.float32)
    for frame_index in range(len(depth)):
        frame_depth = np.asarray(depth[frame_index], dtype=np.float32)
        frame_mask = np.asarray(masks[frame_index], dtype=np.uint8)
        if frame_depth.shape != (height, width):
            frame_depth = cv2.resize(
                frame_depth,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        if frame_mask.shape != (height, width):
            frame_mask = cv2.resize(
                frame_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        if kernel is not None:
            frame_mask = cv2.erode(frame_mask, kernel)
        valid = (
            (frame_mask > 0)
            & np.isfinite(frame_depth)
            & (frame_depth > 0.02)
            & (frame_depth < 5.0)
        )
        samples = frame_depth[valid]
        if len(samples) >= min_samples:
            lo, hi = np.quantile(samples, (0.1, 0.9))
            trimmed = samples[(samples >= lo) & (samples <= hi)]
            if len(trimmed) >= min_samples:
                result[frame_index] = float(np.median(trimmed))
    return _median_fill_short_gaps(result, max_gap=5, window=5)


def _median_fill_short_gaps(
    values: np.ndarray,
    *,
    max_gap: int,
    window: int,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(out)
    index = 0
    while index < len(out):
        if valid[index]:
            index += 1
            continue
        end = index
        while end < len(out) and not valid[end]:
            end += 1
        if (
            end - index <= max_gap
            and index > 0
            and end < len(out)
            and valid[index - 1]
            and valid[end]
        ):
            out[index:end] = np.linspace(
                out[index - 1],
                out[end],
                end - index + 2,
                dtype=np.float32,
            )[1:-1]
            valid[index:end] = True
        index = end
    if window > 1:
        radius = window // 2
        smoothed = out.copy()
        for index in range(len(out)):
            if not np.isfinite(out[index]):
                continue
            chunk = out[max(0, index - radius):min(len(out), index + radius + 1)]
            finite = chunk[np.isfinite(chunk)]
            if len(finite):
                smoothed[index] = float(np.median(finite))
        out = smoothed
    return out


def compute_occluded_fingers(
    *,
    robot_mask: np.ndarray,
    finger_mask: np.ndarray,
    robot_depth: np.ndarray,
    occluder_mask: np.ndarray,
    contact_support_mask: np.ndarray,
    object_depth_m: float = math.nan,
    contact_depth_m: float = math.nan,
    object_depth_margin_m: float = 0.010,
    contact_depth_tolerance_m: float = 0.012,
    contact_depth_bias_m: float = 0.0,
) -> np.ndarray:
    """Compute a fail-open finger-only mask for one frame/finger ROI.

    ``contact_depth_bias_m`` is applied only when the HaCo contact surface is
    the depth proxy.  The independently measured metric object-depth branch is
    intentionally unchanged.
    """
    robot = np.asarray(robot_mask, dtype=bool)
    fingers = np.asarray(finger_mask, dtype=bool)
    depth = np.asarray(robot_depth, dtype=np.float32)
    occluder = np.asarray(occluder_mask, dtype=bool)
    support = np.asarray(contact_support_mask, dtype=bool)
    if not (
        robot.shape == fingers.shape == depth.shape
        == occluder.shape == support.shape
    ):
        raise ValueError("occlusion inputs must share one (H,W) shape")
    if (
        not math.isfinite(contact_depth_bias_m)
        or contact_depth_bias_m < 0.0
    ):
        raise ValueError("contact depth bias must be finite and non-negative")
    candidate = robot & fingers & occluder & support & np.isfinite(depth)
    if np.isfinite(object_depth_m):
        depth_gate = depth > float(object_depth_m) + object_depth_margin_m
    elif np.isfinite(contact_depth_m):
        # A HaCo contact vertex lies on the human finger surface rather than
        # independently measured object geometry. Treat it as a local depth
        # proxy with a small tolerance, never as a global object plane.
        effective_robot_depth = depth
        if contact_depth_bias_m > 0.0:
            effective_robot_depth = depth + np.float32(contact_depth_bias_m)
        depth_gate = effective_robot_depth >= (
            float(contact_depth_m) - contact_depth_tolerance_m
        )
    else:
        return np.zeros_like(robot)
    return candidate & depth_gate


def resize_positive_depth(
    depth: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Resize a positive depth map without blending unknown zeros into it."""
    frame = np.asarray(depth, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError(f"depth frame must be two-dimensional, got {frame.shape}")
    if frame.shape == (height, width):
        return frame.copy()
    valid = np.isfinite(frame) & (frame > 0.0)
    weighted = cv2.resize(
        np.where(valid, frame, 0.0),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    weights = cv2.resize(
        valid.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = np.zeros((height, width), dtype=np.float32)
    np.divide(weighted, weights, out=resized, where=weights > 0.5)
    return resized


def object_surface_contact_alignment(
    object_surface_depth: np.ndarray,
    object_mask: np.ndarray,
    contact_support_mask: np.ndarray,
    *,
    contact_depth_m: float,
    alignment: str,
    min_samples: int,
    max_shift_m: float,
    consistency_m: float,
) -> dict[str, float | int | bool]:
    """Estimate a bounded local Z translation from object surface to contact.

    The dense surface shape is retained.  HaCo contributes only a local
    registration translation: it does not replace the model's per-pixel depth
    variation.  Missing samples or a grossly inconsistent contact fail open.
    """
    if alignment not in {"none", "contact"}:
        raise ValueError("object surface alignment must be 'none' or 'contact'")
    surface = np.asarray(object_surface_depth, dtype=np.float32)
    mask = np.asarray(object_mask, dtype=bool)
    support = np.asarray(contact_support_mask, dtype=bool)
    if not (surface.shape == mask.shape == support.shape):
        raise ValueError("object surface alignment arrays must share one shape")
    if min_samples <= 0:
        raise ValueError("object surface min_samples must be positive")
    if (
        not math.isfinite(max_shift_m)
        or not math.isfinite(consistency_m)
        or max_shift_m < 0.0
        or consistency_m < max_shift_m
    ):
        raise ValueError("invalid object surface contact tolerances")
    valid = (
        mask
        & support
        & np.isfinite(surface)
        & (surface > 0.02)
        & (surface < 5.0)
    )
    samples = surface[valid]
    raw_count = int(len(samples))
    if raw_count < min_samples:
        return {
            "valid": False,
            "consistent": False,
            "sample_count": raw_count,
            "inlier_count": 0,
            "local_surface_depth_m": math.nan,
            "contact_residual_m": math.nan,
            "applied_shift_m": math.nan,
        }
    low, high = np.quantile(samples, (0.10, 0.90))
    inliers = samples[(samples >= low) & (samples <= high)]
    if len(inliers) >= min_samples:
        center = float(np.median(inliers))
        mad = float(np.median(np.abs(inliers - center)))
        if mad > np.finfo(np.float32).eps:
            sigma = 1.4826 * mad
            refined = inliers[np.abs(inliers - center) <= 3.5 * sigma]
            if len(refined) >= min_samples:
                inliers = refined
    if len(inliers) < min_samples:
        return {
            "valid": False,
            "consistent": False,
            "sample_count": raw_count,
            "inlier_count": int(len(inliers)),
            "local_surface_depth_m": math.nan,
            "contact_residual_m": math.nan,
            "applied_shift_m": math.nan,
        }
    local_depth = float(np.median(inliers))
    if alignment == "none":
        return {
            "valid": True,
            "consistent": True,
            "sample_count": raw_count,
            "inlier_count": int(len(inliers)),
            "local_surface_depth_m": local_depth,
            "contact_residual_m": (
                float(contact_depth_m - local_depth)
                if np.isfinite(contact_depth_m)
                else math.nan
            ),
            "applied_shift_m": 0.0,
        }
    if not np.isfinite(contact_depth_m):
        return {
            "valid": False,
            "consistent": False,
            "sample_count": raw_count,
            "inlier_count": int(len(inliers)),
            "local_surface_depth_m": local_depth,
            "contact_residual_m": math.nan,
            "applied_shift_m": math.nan,
        }
    residual = float(contact_depth_m - local_depth)
    consistent = abs(residual) <= consistency_m
    return {
        "valid": bool(consistent),
        "consistent": bool(consistent),
        "sample_count": raw_count,
        "inlier_count": int(len(inliers)),
        "local_surface_depth_m": local_depth,
        "contact_residual_m": residual,
        "applied_shift_m": (
            float(np.clip(residual, -max_shift_m, max_shift_m))
            if consistent
            else math.nan
        ),
    }


def compute_occluded_fingers_surface(
    *,
    robot_mask: np.ndarray,
    finger_mask: np.ndarray,
    robot_depth: np.ndarray,
    occluder_mask: np.ndarray,
    contact_support_mask: np.ndarray,
    object_surface_depth: np.ndarray,
    surface_shift_m: float = 0.0,
    object_depth_margin_m: float = 0.010,
) -> np.ndarray:
    """Apply a per-pixel visible-object camera-Z gate to one robot finger."""
    robot = np.asarray(robot_mask, dtype=bool)
    finger = np.asarray(finger_mask, dtype=bool)
    depth = np.asarray(robot_depth, dtype=np.float32)
    occluder = np.asarray(occluder_mask, dtype=bool)
    support = np.asarray(contact_support_mask, dtype=bool)
    surface = np.asarray(object_surface_depth, dtype=np.float32)
    if not (
        robot.shape == finger.shape == depth.shape == occluder.shape
        == support.shape == surface.shape
    ):
        raise ValueError("object surface occlusion inputs must share one shape")
    if not math.isfinite(surface_shift_m):
        return np.zeros_like(robot)
    if not math.isfinite(object_depth_margin_m) or object_depth_margin_m < 0.0:
        raise ValueError("object surface depth margin must be non-negative")
    valid_surface = (
        np.isfinite(surface)
        & (surface > 0.02)
        & (surface < 5.0)
    )
    candidate = (
        robot
        & finger
        & occluder
        & support
        & np.isfinite(depth)
        & valid_surface
    )
    threshold = surface + np.float32(surface_shift_m + object_depth_margin_m)
    return candidate & (depth > threshold)


def object_surface_temporal_eligibility(
    *,
    finger_mask: np.ndarray,
    robot_depth: np.ndarray,
    occluder_mask: np.ndarray,
    object_surface_depth: np.ndarray,
    front_slack_m: float,
) -> np.ndarray:
    """Return current-frame pixels eligible for short-gap interpolation.

    A short temporal gap may be caused by one missing/noisy surface frame.  A
    valid depth that places the robot clearly in front vetoes interpolation;
    missing surface depth does not, because the filter still requires matching
    occlusion on both sides of the gap and the current semantic finger/object
    overlap.  Robot depth itself must always be finite.
    """
    finger = np.asarray(finger_mask, dtype=bool)
    depth = np.asarray(robot_depth, dtype=np.float32)
    occluder = np.asarray(occluder_mask, dtype=bool)
    surface = np.asarray(object_surface_depth, dtype=np.float32)
    if not (finger.shape == depth.shape == occluder.shape == surface.shape):
        raise ValueError("temporal Object3D inputs must share one shape")
    if not math.isfinite(front_slack_m) or front_slack_m < 0.0:
        raise ValueError("temporal front slack must be finite and non-negative")
    finite_robot = np.isfinite(depth)
    valid_surface = (
        np.isfinite(surface)
        & (surface > 0.02)
        & (surface < 5.0)
    )
    clearly_in_front = (
        finite_robot
        & valid_surface
        & (depth <= surface - np.float32(front_slack_m))
    )
    return finger & occluder & finite_robot & ~clearly_in_front


def bridge_short_occlusion_gaps(
    raw_masks: np.ndarray,
    eligible_masks: np.ndarray,
    finger_labels: np.ndarray,
    *,
    max_gap_frames: int,
    motion_radius_px: int,
    label_count: int = len(FINGER_NAMES),
    source_presence: np.ndarray | None = None,
    output: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Close short bidirectional mask gaps without copying stale hand pixels.

    The operation is offline by design: an added pixel needs spatially nearby
    support from the same semantic finger both before and after the current
    frame.  The total open interval may not exceed ``max_gap_frames``.  Both
    supports are dilated according to their temporal distance, intersected,
    and finally clipped to current-frame eligibility and finger semantics.
    Raw occlusion is only added to, never removed.
    """
    raw = np.asanyarray(raw_masks)
    eligible = np.asanyarray(eligible_masks)
    labels = np.asanyarray(finger_labels)
    if not (raw.shape == eligible.shape == labels.shape) or raw.ndim != 3:
        raise ValueError("temporal masks/labels must share shape (T,H,W)")
    if max_gap_frames < 0 or motion_radius_px < 0:
        raise ValueError("invalid temporal gap filter settings")
    if label_count <= 0 or label_count > 255:
        raise ValueError("temporal label count must be in 1..255")
    frame_count, height, width = raw.shape
    if output is None:
        filtered = np.empty(raw.shape, dtype=bool)
    else:
        filtered = np.asanyarray(output)
        if filtered.shape != raw.shape or filtered.dtype != np.bool_:
            raise ValueError("temporal output must be bool with shape (T,H,W)")

    if source_presence is None:
        presence = np.zeros((frame_count, label_count), dtype=bool)
        for frame_index in range(frame_count):
            frame_raw = np.asarray(raw[frame_index], dtype=bool)
            frame_labels = np.asarray(labels[frame_index], dtype=np.uint8)
            for finger_index in range(label_count):
                presence[frame_index, finger_index] = bool(
                    np.any(frame_raw & (frame_labels == finger_index + 1))
                )
    else:
        presence = np.asarray(source_presence, dtype=bool)
        expected = (frame_count, label_count)
        if presence.shape != expected:
            raise ValueError(
                f"source presence must have shape {expected}, got {presence.shape}"
            )

    kernel_cache: dict[int, np.ndarray] = {}

    @functools.lru_cache(maxsize=512)
    def expanded_source(
        frame_index: int,
        finger_id: int,
        distance: int,
    ) -> tuple[int, int, int, int, np.ndarray] | None:
        source = (
            np.asarray(raw[frame_index], dtype=bool)
            & (np.asarray(labels[frame_index], dtype=np.uint8) == finger_id)
        )
        rows, columns = np.nonzero(source)
        if not len(rows):
            return None
        radius = int(motion_radius_px * distance)
        y0 = max(0, int(rows.min()) - radius)
        y1 = min(height, int(rows.max()) + radius + 1)
        x0 = max(0, int(columns.min()) - radius)
        x1 = min(width, int(columns.max()) + radius + 1)
        region = source[y0:y1, x0:x1].astype(np.uint8)
        if radius > 0:
            kernel = kernel_cache.get(radius)
            if kernel is None:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * radius + 1, 2 * radius + 1),
                )
                kernel_cache[radius] = kernel
            region = cv2.dilate(region, kernel)
        return y0, y1, x0, x1, region.astype(bool)

    added_per_frame_finger = np.zeros(
        (frame_count, label_count),
        dtype=np.int64,
    )
    for frame_index in range(frame_count):
        frame_raw = np.asarray(raw[frame_index], dtype=bool)
        frame_filtered = frame_raw.copy()
        frame_eligible = np.asarray(eligible[frame_index], dtype=bool)
        frame_labels = np.asarray(labels[frame_index], dtype=np.uint8)
        if max_gap_frames > 0:
            for finger_index in range(label_count):
                finger_id = finger_index + 1
                current_eligible = frame_eligible & (frame_labels == finger_id)
                if not np.any(current_eligible):
                    continue
                for left_distance in range(1, max_gap_frames + 1):
                    left_index = frame_index - left_distance
                    if left_index < 0:
                        break
                    if not presence[left_index, finger_index]:
                        continue
                    max_right_distance = (
                        max_gap_frames + 1 - left_distance
                    )
                    for right_distance in range(1, max_right_distance + 1):
                        right_index = frame_index + right_distance
                        if right_index >= frame_count:
                            break
                        if not presence[right_index, finger_index]:
                            continue
                        left = expanded_source(
                            left_index,
                            finger_id,
                            left_distance,
                        )
                        right = expanded_source(
                            right_index,
                            finger_id,
                            right_distance,
                        )
                        if left is None or right is None:
                            continue
                        y0 = max(left[0], right[0])
                        y1 = min(left[1], right[1])
                        x0 = max(left[2], right[2])
                        x1 = min(left[3], right[3])
                        if y0 >= y1 or x0 >= x1:
                            continue
                        left_region = left[4][
                            y0 - left[0] : y1 - left[0],
                            x0 - left[2] : x1 - left[2],
                        ]
                        right_region = right[4][
                            y0 - right[0] : y1 - right[0],
                            x0 - right[2] : x1 - right[2],
                        ]
                        bridge = (
                            left_region
                            & right_region
                            & current_eligible[y0:y1, x0:x1]
                        )
                        frame_filtered[y0:y1, x0:x1] |= bridge
        added = frame_filtered & ~frame_raw
        for finger_index in range(label_count):
            added_per_frame_finger[frame_index, finger_index] = int(
                np.sum(added & (frame_labels == finger_index + 1))
            )
        filtered[frame_index] = frame_filtered

    return filtered, {
        "added_per_frame_finger": added_per_frame_finger,
        "added_pixels": int(added_per_frame_finger.sum()),
        "added_frames": int(
            np.any(added_per_frame_finger > 0, axis=1).sum()
        ),
        "added_frame_fingers": int((added_per_frame_finger > 0).sum()),
    }


def expand_verified_contact_interior(
    candidate_mask: np.ndarray,
    *,
    eligible_mask: np.ndarray,
    finger_mask: np.ndarray,
    expand_px: int,
    added_cap_fraction: float,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Complete a border contact inside the same verified MH finger region.

    ``candidate_mask`` is the existing HaCo/object/depth intersection.  It is
    allowed to grow only when it touches the inner one-pixel boundary of the
    rendered semantic finger, and only through eight-connected pixels in
    ``eligible_mask & finger_mask``.  Repeated 3x3 dilations impose a bounded
    geodesic distance; a deterministic distance/row-major ordering enforces
    the added-pixel cap when the last growth layer is only partly accepted.

    The cap is relative to the original verified candidate rather than to the
    complete semantic finger.  Consequently a one-pixel alignment accident
    cannot turn into an unbounded whole-finger removal.
    """
    candidate = np.asarray(candidate_mask, dtype=bool)
    eligible = np.asarray(eligible_mask, dtype=bool)
    finger = np.asarray(finger_mask, dtype=bool)
    if not (candidate.shape == eligible.shape == finger.shape):
        raise ValueError("contact interior masks must share one shape")
    if candidate.ndim != 2:
        raise ValueError(
            f"contact interior masks must be two-dimensional, got "
            f"{candidate.shape}"
        )
    if expand_px < 0:
        raise ValueError("contact interior expansion must be non-negative")
    if not np.isfinite(added_cap_fraction) or not (
        0.0 <= added_cap_fraction <= 1.0
    ):
        raise ValueError("contact interior cap fraction must be in [0,1]")

    allowed = eligible & finger
    if np.any(candidate & ~allowed):
        raise ValueError(
            "verified contact candidate must be a subset of the eligible "
            "semantic finger region"
        )

    output = candidate.copy()
    seed_pixels = int(candidate.sum())
    cap_pixels = int(math.floor(seed_pixels * added_cap_fraction))
    diagnostics: dict[str, int | bool] = {
        "seed_pixels": seed_pixels,
        "boundary_seed_pixels": 0,
        "eligible_pixels": int(allowed.sum()),
        "added_cap_pixels": cap_pixels,
        "added_pixels": 0,
        "cap_limited": False,
        "expanded": False,
    }
    if seed_pixels == 0 or expand_px == 0:
        return output, diagnostics

    kernel = np.ones((3, 3), dtype=np.uint8)
    inner_boundary = finger & ~cv2.erode(
        finger.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    boundary_seed = candidate & inner_boundary
    boundary_seed_pixels = int(boundary_seed.sum())
    diagnostics["boundary_seed_pixels"] = boundary_seed_pixels
    if boundary_seed_pixels == 0:
        return output, diagnostics

    # Restrict growth to eligible connected components that contain a
    # boundary-qualified seed. This is stronger than merely masking the final
    # dilation and prevents a disconnected finger fragment from being filled.
    _, component_labels = cv2.connectedComponents(
        allowed.astype(np.uint8),
        connectivity=8,
    )
    touched_components = np.unique(component_labels[boundary_seed])
    touched_components = touched_components[touched_components > 0]
    if not len(touched_components):
        return output, diagnostics
    growth_region = np.isin(component_labels, touched_components)
    growing = candidate & growth_region

    # Distance is used only to break a partially accepted final geodesic
    # layer. np.flatnonzero supplies a stable row-major tie-break.
    distance_from_seed = cv2.distanceTransform(
        (~growing).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    remaining = cap_pixels
    for _ in range(expand_px):
        frontier = (
            cv2.dilate(growing.astype(np.uint8), kernel).astype(bool)
            & growth_region
            & ~growing
        )
        frontier_count = int(frontier.sum())
        if frontier_count == 0:
            break
        if remaining <= 0:
            diagnostics["cap_limited"] = True
            break
        if frontier_count > remaining:
            flat = np.flatnonzero(frontier)
            distances = distance_from_seed.ravel()[flat]
            order = np.lexsort((flat, distances))
            accepted = flat[order[:remaining]]
            growing.ravel()[accepted] = True
            remaining = 0
            diagnostics["cap_limited"] = True
            break
        growing |= frontier
        remaining -= frontier_count

    output |= growing
    added_pixels = int((output & ~candidate).sum())
    diagnostics["added_pixels"] = added_pixels
    diagnostics["expanded"] = added_pixels > 0
    return output, diagnostics


def composite_frame(
    background_bgr: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    finger_mask: np.ndarray,
    occluded_mask: np.ndarray,
    *,
    robot_edge_sigma_px: float,
    occlusion_edge_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final BGR, robot-only BGR, and effective robot alpha."""
    background = np.asarray(background_bgr, dtype=np.uint8)
    robot = np.asarray(robot_rgb, dtype=np.uint8)[..., ::-1]
    base = np.asarray(robot_mask, dtype=np.float32)
    fingers = np.asarray(finger_mask, dtype=bool)
    occluded = np.asarray(occluded_mask, dtype=np.float32)
    if (
        background.shape != robot.shape
        or base.shape != background.shape[:2]
        or fingers.shape != base.shape
        or occluded.shape != base.shape
    ):
        raise ValueError("composite inputs are not aligned")
    if np.any((occluded > 0) & ~fingers):
        raise ValueError("occlusion mask contains non-finger pixels")
    if robot_edge_sigma_px > 0:
        base = cv2.GaussianBlur(base, (0, 0), robot_edge_sigma_px)
    if occlusion_edge_sigma_px > 0:
        occluded = cv2.GaussianBlur(
            occluded,
            (0, 0),
            occlusion_edge_sigma_px,
        )
    # Never let feathering propagate into palm/arm pixels.
    occluded[~fingers] = 0.0
    alpha = np.clip(base * (1.0 - np.clip(occluded, 0.0, 1.0)), 0.0, 1.0)
    alpha3 = alpha[..., None]
    final = np.clip(
        background.astype(np.float32) * (1.0 - alpha3)
        + robot.astype(np.float32) * alpha3,
        0,
        255,
    ).astype(np.uint8)
    robot_only = np.clip(
        robot.astype(np.float32) * alpha3,
        0,
        255,
    ).astype(np.uint8)
    return final, robot_only, alpha


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    if width <= 0 or height <= 0 or frames <= 0 or fps <= 0:
        raise ValueError(f"invalid video metadata for {path}")
    return width, height, frames, fps


def _open_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer {path}")
    return writer


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(mask, dtype=np.uint8)
    if array.shape != (height, width):
        array = cv2.resize(
            array,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return array.astype(bool)


def _resize_overlay_frame(
    robot_rgb: np.ndarray,
    robot_depth: np.ndarray,
    robot_mask: np.ndarray,
    finger_labels: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resize one Isaac RGB-D/semantic frame without corrupting labels."""
    rgb = np.asarray(robot_rgb)
    depth = np.asarray(robot_depth, dtype=np.float32)
    # Isaac arrays are normally opened read-only with mmap_mode="r".  Keep a
    # writable frame copy because the semantic-subset repair below is in-place.
    robot = np.array(robot_mask, dtype=bool, copy=True)
    labels = np.asarray(finger_labels, dtype=np.uint8)
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(
            depth,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        robot = cv2.resize(
            robot.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        labels = cv2.resize(
            labels,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
    fingers = labels > 0
    # Semantic fingers are rendered robot geometry. Preserve that invariant
    # even when nearest-neighbour sampling chooses opposite sides of one edge.
    robot |= fingers
    return rgb, depth, robot, fingers, labels


def _frame_occluder(
    *,
    raw: np.ndarray,
    background: np.ndarray,
    amodal: np.ndarray,
    visible: np.ndarray,
    explicit_object_mask: np.ndarray | None,
    config: OcclusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if explicit_object_mask is not None:
        core = np.asarray(explicit_object_mask, dtype=bool)
        # A verified modal mask is already the exact foreground boundary.
        # Dilating it would remove robot pixels outside the real object and
        # expose the inpainted background as a translucent silhouette.
        return core, core
    else:
        core = proxy_occluder_mask(
            raw,
            background,
            amodal,
            visible,
            lab_threshold=config.raw_bg_lab_threshold,
        )
    dilated = core
    if config.object_mask_dilate_px > 0:
        radius = config.object_mask_dilate_px
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        dilated = cv2.dilate(
            core.astype(np.uint8),
            kernel,
        ).astype(bool)
    return core, dilated


def _camera_vertices(
    vertices: np.ndarray,
    retarget: np.lib.npyio.NpzFile,
    frame_index: int,
) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float32)
    if bool(retarget["frame_is_cam_space"]):
        return points
    if "R_c2w" not in retarget.files or "t_c2w" not in retarget.files:
        raise ValueError(
            "world-space HaWoR vertices require R_c2w/t_c2w for projection"
        )
    rotation = np.asarray(retarget["R_c2w"][frame_index], dtype=np.float32)
    translation = np.asarray(
        retarget["t_c2w"][frame_index],
        dtype=np.float32,
    )
    return (points - translation) @ rotation


def fuse_contact_scores(
    primary_scores: np.ndarray,
    auxiliary_scores: np.ndarray | None,
) -> np.ndarray:
    """Max-fuse per-finger HaCo scores without mixing camera geometry.

    The primary view remains authoritative for projected contact locations and
    contact-surface depth.  The auxiliary view contributes only independent
    per-finger contact confidence, so no cross-camera extrinsics are implied.
    A missing/NaN auxiliary sample leaves the primary score unchanged.
    """
    primary = np.asarray(primary_scores, dtype=np.float32)
    if primary.ndim != 2 or primary.shape[1] != len(FINGER_NAMES):
        raise ValueError(
            "primary contact scores must have shape (T,5), got "
            f"{primary.shape}"
        )
    if auxiliary_scores is None:
        return primary.copy()
    auxiliary = np.asarray(auxiliary_scores, dtype=np.float32)
    if auxiliary.shape != primary.shape:
        raise ValueError(
            "auxiliary contact scores must match primary shape, got "
            f"{auxiliary.shape} != {primary.shape}"
        )
    return np.fmax(primary, auxiliary).astype(np.float32, copy=False)


def contact_activation_tracks(
    primary_scores: np.ndarray,
    auxiliary_scores: np.ndarray | None,
    hidden_fraction: np.ndarray,
    config: OcclusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build five-finger contact states with MH-local SH rescue.

    ``primary_scores`` and ``hidden_fraction`` belong to the final MH view.
    SH may propose contact for the same finger, but an SH-only proposal is
    eligible only when the MH contact neighbourhood has at least the existing
    hysteresis off-threshold of object support.  Starting a run still requires
    the unchanged MH on-threshold.  This makes the confidence-only policy
    explicit while preserving the previous max-score + MH-evidence behaviour.
    """
    fused_scores = fuse_contact_scores(primary_scores, auxiliary_scores)
    primary = np.asarray(primary_scores, dtype=np.float32)
    local_support = np.asarray(hidden_fraction, dtype=np.float32)
    if local_support.shape != primary.shape:
        raise ValueError(
            "hidden fractions must match primary scores, got "
            f"{local_support.shape} != {primary.shape}"
        )

    primary_gate = primary >= config.contact_score_threshold
    auxiliary_available = np.zeros_like(primary_gate)
    auxiliary_gate = np.zeros_like(primary_gate)
    if auxiliary_scores is not None:
        auxiliary = np.asarray(auxiliary_scores, dtype=np.float32)
        auxiliary_available = np.isfinite(auxiliary)
        auxiliary_gate = (
            auxiliary_available
            & (auxiliary >= config.contact_score_threshold)
        )

    auxiliary_proposal = auxiliary_gate & ~primary_gate
    auxiliary_qualified = (
        auxiliary_proposal
        & np.isfinite(local_support)
        & (local_support >= config.hidden_fraction_off)
    )
    fused_gate = primary_gate | auxiliary_qualified
    evidence = np.where(fused_gate, local_support, 0.0).astype(np.float32)
    evidence[~np.isfinite(evidence)] = 0.0

    active = np.zeros_like(fused_gate)
    for finger_index in range(len(FINGER_NAMES)):
        active[:, finger_index] = temporal_hysteresis(
            evidence[:, finger_index],
            on_threshold=config.hidden_fraction_on,
            off_threshold=config.hidden_fraction_off,
            min_on_frames=config.min_on_frames,
            hold_frames=config.hold_frames,
        )

    gates = {
        "primary": primary_gate,
        "auxiliary_available": auxiliary_available,
        "auxiliary": auxiliary_gate,
        "auxiliary_proposal": auxiliary_proposal,
        "auxiliary_qualified": auxiliary_qualified,
        "fused": fused_gate,
    }
    return fused_scores, evidence, active, gates


def _contact_frame_selection(
    *,
    contact_path: Path,
    frame_index: int,
    side: str,
    parts: np.ndarray,
    palmar: np.ndarray,
    config: OcclusionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load one HaCo frame and select stable palmar contact vertices."""
    scores = np.zeros(len(FINGER_NAMES), dtype=np.float32)
    selected_indices: dict[str, np.ndarray] = {}
    if not contact_path.is_file():
        raise FileNotFoundError(contact_path)
    with np.load(contact_path) as contact:
        metadata_index = (
            int(contact["hawor_frame_index"])
            if "hawor_frame_index" in contact.files
            else frame_index
        )
        if metadata_index != frame_index:
            raise ValueError(
                f"contact frame mismatch in {contact_path}: "
                f"{metadata_index} != {frame_index}"
            )
        probability_key = f"{side}_contact_probability"
        mask_key = f"{side}_contact_mask"
        if probability_key in contact.files:
            probability = np.asarray(
                contact[probability_key],
                dtype=np.float32,
            )
        elif mask_key in contact.files:
            probability = np.asarray(contact[mask_key], dtype=np.float32)
        else:
            probability = np.zeros(778, dtype=np.float32)
        if mask_key in contact.files:
            filtered_contact = np.asarray(
                contact[mask_key],
                dtype=bool,
            )
        else:
            filtered_contact = (
                probability >= config.contact_point_threshold
            )
    if probability.shape != (778,):
        raise ValueError(
            f"invalid contact probability shape in {contact_path}: "
            f"{probability.shape}"
        )
    if filtered_contact.shape != (778,):
        raise ValueError(
            f"invalid contact mask shape in {contact_path}: "
            f"{filtered_contact.shape}"
        )
    for finger_index, finger in enumerate(FINGER_NAMES):
        eligible = (
            palmar
            & np.isin(parts, FINGER_PARTS[finger])
            & filtered_contact
        )
        vertex_indices = np.flatnonzero(eligible)
        if len(vertex_indices) < config.min_contact_points:
            selected_indices[finger] = np.empty(0, dtype=np.int64)
            continue
        values = probability[vertex_indices]
        top_count = max(
            config.min_contact_points,
            int(math.ceil(len(values) * config.contact_top_fraction)),
        )
        top_count = min(top_count, len(values))
        order = np.argsort(values)[::-1]
        top = values[order[:top_count]]
        scores[finger_index] = float(top.mean()) if len(top) else 0.0

        selected_local = order[
            values[order] >= config.contact_point_threshold
        ]
        if len(selected_local) < config.min_contact_points:
            selected_local = order[:config.min_contact_points]
        selected_indices[finger] = vertex_indices[selected_local]
    return scores, selected_indices


def _contact_frame_features(
    *,
    contact_path: Path,
    retarget: np.lib.npyio.NpzFile,
    frame_index: int,
    side: str,
    parts: np.ndarray,
    palmar: np.ndarray,
    config: OcclusionConfig,
    focal_output_px: float,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    scores, selected_indices = _contact_frame_selection(
        contact_path=contact_path,
        frame_index=frame_index,
        side=side,
        parts=parts,
        palmar=palmar,
        config=config,
    )
    points_uv: dict[str, np.ndarray] = {}
    points_z: dict[str, np.ndarray] = {}
    vertices = _camera_vertices(
        retarget[f"verts_{side}"][frame_index],
        retarget,
        frame_index,
    )
    for finger in FINGER_NAMES:
        selected = selected_indices[finger]
        if not len(selected):
            points_uv[finger] = np.empty((0, 2), dtype=np.float32)
            points_z[finger] = np.empty(0, dtype=np.float32)
            continue
        selected_points = vertices[selected]
        uv, valid = project_camera_points(
            selected_points,
            focal_px=focal_output_px,
            image_width=output_width,
            image_height=output_height,
        )
        in_frame = (
            valid
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < output_width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < output_height)
        )
        points_uv[finger] = uv[in_frame]
        points_z[finger] = selected_points[in_frame, 2].astype(np.float32)
    return scores, points_uv, points_z


def _debug_grid(
    raw: np.ndarray,
    finger_mask: np.ndarray,
    contact_support: np.ndarray,
    occluder: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    occluded: np.ndarray,
    final: np.ndarray,
) -> np.ndarray:
    height, width = raw.shape[:2]
    first = raw.copy()
    first[finger_mask] = (
        0.35 * first[finger_mask]
        + 0.65 * np.array([255, 255, 0])
    ).astype(np.uint8)
    second = raw.copy()
    second[occluder] = (
        0.35 * second[occluder]
        + 0.65 * np.array([0, 0, 255])
    ).astype(np.uint8)
    second[contact_support] = (
        0.35 * second[contact_support]
        + 0.65 * np.array([0, 255, 255])
    ).astype(np.uint8)
    third = np.zeros_like(raw)
    robot_bgr = np.asarray(robot_rgb)[..., ::-1]
    visible = np.asarray(robot_mask, dtype=bool) & ~occluded
    third[visible] = robot_bgr[visible]
    third[occluded] = (0, 0, 255)
    labels = (
        (first, "Isaac finger"),
        (second, "contact / occluder"),
        (third, "visible / hidden"),
        (final, "final"),
    )
    panels = []
    for panel, label in labels:
        resized = cv2.resize(
            panel,
            (width // 2, height // 2),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            resized,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(resized)
    return np.concatenate(
        [
            np.concatenate(panels[:2], axis=1),
            np.concatenate(panels[2:], axis=1),
        ],
        axis=0,
    )


def _true_runs(values: np.ndarray) -> list[list[int]]:
    binary = np.asarray(values, dtype=bool)
    changes = np.diff(np.r_[False, binary, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--background", type=Path, default=None)
    parser.add_argument("--raw_video", type=Path, default=None)
    parser.add_argument("--hawor_npz", type=Path, default=None)
    parser.add_argument("--contact_dir", type=Path, default=None)
    parser.add_argument(
        "--aux_contact_dir",
        type=Path,
        default=None,
        help=(
            "Optional synchronized second-view HaCo directory. Its per-finger "
            "scores are max-fused with the primary view; projected contact "
            "locations and depth always remain primary-view geometry."
        ),
    )
    parser.add_argument(
        "--aux_frame_offset",
        type=int,
        default=0,
        help=(
            "Auxiliary lookup offset: aux index = primary/output index + "
            "offset; out-of-range samples fail open."
        ),
    )
    parser.add_argument(
        "--aux_side",
        choices=("left", "right"),
        default=None,
        help="Auxiliary HaCo hand side; defaults to the rendered primary side",
    )
    parser.add_argument(
        "--overlay_dir",
        type=Path,
        default=None,
        help="Isaac overlay arrays; default: <processed_demo>/overlay_processor",
    )
    parser.add_argument(
        "--object_mask",
        type=Path,
        default=None,
        help="Preferred modal object foreground mask (T,H,W); amodal is unsafe",
    )
    parser.add_argument(
        "--object_restore_mask",
        type=Path,
        default=None,
        help=(
            "Optional clean observed-object mask used only to restore RGB from "
            "--raw_video. Occlusion geometry still uses --object_mask. The "
            "restore mask must be a subset of the explicit modal object mask."
        ),
    )
    parser.add_argument(
        "--object_depth_mask",
        type=Path,
        default=None,
        help="Modal object mask used only to sample scene depth",
    )
    parser.add_argument(
        "--scene_depth",
        type=Path,
        default=None,
        help="Aligned metric scene depth (T,H,W)",
    )
    parser.add_argument(
        "--object_surface_depth",
        type=Path,
        default=None,
        help=(
            "Dense visible-object camera-Z model (T,H,W), normally produced "
            "by build_object_surface_model.py; zero means unknown"
        ),
    )
    parser.add_argument(
        "--object_surface_alignment",
        choices=("none", "contact"),
        default="contact",
        help=(
            "Optionally translate the local object surface in Z so it meets "
            "the primary/MH HaCo contact depth"
        ),
    )
    parser.add_argument(
        "--occlusion_mode",
        choices=("auto", "haco", "ensemble", "object3d"),
        default="auto",
        help=(
            "haco uses the HaCo contact-surface depth only; ensemble requires "
            "both HaCo evidence and sensor scene depth; object3d intersects "
            "HaCo with a dense object surface. auto prefers object3d, then "
            "ensemble, then haco according to supplied inputs."
        ),
    )
    parser.add_argument(
        "--contact_score_threshold",
        type=float,
        default=OcclusionConfig.contact_score_threshold,
    )
    parser.add_argument(
        "--contact_point_threshold",
        type=float,
        default=OcclusionConfig.contact_point_threshold,
    )
    parser.add_argument(
        "--hidden_fraction_on",
        type=float,
        default=OcclusionConfig.hidden_fraction_on,
    )
    parser.add_argument(
        "--hidden_fraction_off",
        type=float,
        default=OcclusionConfig.hidden_fraction_off,
    )
    parser.add_argument(
        "--contact_radius_px",
        type=int,
        default=OcclusionConfig.contact_radius_px,
    )
    parser.add_argument(
        "--contact_interior_expand_px",
        type=int,
        default=OcclusionConfig.contact_interior_expand_px,
        help=(
            "Opt-in MH semantic-finger interior completion radius. Zero "
            "preserves the original contact-disk mask exactly."
        ),
    )
    parser.add_argument(
        "--contact_interior_expand_cap_fraction",
        type=float,
        default=OcclusionConfig.contact_interior_expand_cap_fraction,
        help=(
            "Maximum added interior pixels as a fraction of the original "
            "verified per-frame/finger candidate."
        ),
    )
    parser.add_argument(
        "--depth_margin_m",
        type=float,
        default=OcclusionConfig.object_depth_margin_m,
    )
    parser.add_argument(
        "--object_depth_erode_px",
        type=int,
        default=OcclusionConfig.object_depth_erode_px,
        help=(
            "Erode the modal object mask before sampling scene depth. "
            "Increase this for approximately aligned hardware depth."
        ),
    )
    parser.add_argument(
        "--object_surface_min_samples",
        type=int,
        default=OcclusionConfig.object_surface_min_samples,
        help="Minimum local object-surface samples around one MH HaCo contact",
    )
    parser.add_argument(
        "--object_surface_contact_max_shift_m",
        type=float,
        default=OcclusionConfig.object_surface_contact_max_shift_m,
        help="Maximum absolute local object-surface Z registration shift",
    )
    parser.add_argument(
        "--object_surface_contact_consistency_m",
        type=float,
        default=OcclusionConfig.object_surface_contact_consistency_m,
        help="Reject object/contact depth residuals larger than this value",
    )
    parser.add_argument(
        "--object3d_force_surface",
        action="store_true",
        help=(
            "Also hide every semantic-finger pixel strictly behind the raw "
            "dense object surface, without HaCo activation/contact locality"
        ),
    )
    parser.add_argument(
        "--object3d_force_margin_m",
        type=float,
        default=OcclusionConfig.object3d_force_margin_m,
        help="Extra Z margin for the HaCo-free full-finger surface-force gate",
    )
    parser.add_argument(
        "--object3d_temporal_max_gap_frames",
        type=int,
        default=OcclusionConfig.object3d_temporal_max_gap_frames,
        help=(
            "Offline bidirectional temporal closing length; zero disables "
            "penetration-gap suppression"
        ),
    )
    parser.add_argument(
        "--object3d_temporal_motion_px",
        type=int,
        default=OcclusionConfig.object3d_temporal_motion_px,
        help="Allowed per-frame image motion when bridging a short mask gap",
    )
    parser.add_argument(
        "--object3d_temporal_front_slack_m",
        type=float,
        default=OcclusionConfig.object3d_temporal_front_slack_m,
        help=(
            "Do not bridge where valid depth puts the robot this far or more "
            "in front of the object surface"
        ),
    )
    parser.add_argument(
        "--contact_depth_tolerance_m",
        type=float,
        default=OcclusionConfig.contact_depth_tolerance_m,
    )
    parser.add_argument(
        "--contact_depth_thickness_scale",
        type=float,
        default=OcclusionConfig.contact_depth_thickness_scale,
        help=(
            "Opt-in XHand transverse-thickness multiplier for the HaCo "
            "contact-depth proxy: 0 keeps the rendered front-face gate, "
            "0.5 tests a half-thickness, and 1.0 tests a full thickness."
        ),
    )
    parser.add_argument(
        "--xhand_thumb_thickness_m",
        type=float,
        default=OcclusionConfig.xhand_thumb_thickness_m,
        help="Full XHand thumb transverse thickness in metres",
    )
    parser.add_argument(
        "--xhand_finger_thickness_m",
        type=float,
        default=OcclusionConfig.xhand_finger_thickness_m,
        help="Full XHand non-thumb transverse thickness in metres",
    )
    parser.add_argument(
        "--min_occlusion_run_frames",
        type=int,
        default=OcclusionConfig.min_occlusion_run_frames,
    )
    args = parser.parse_args()
    occlusion_mode = args.occlusion_mode
    if occlusion_mode == "auto":
        if args.object_surface_depth is not None:
            occlusion_mode = "object3d"
        elif args.scene_depth is not None:
            occlusion_mode = "ensemble"
        else:
            occlusion_mode = "haco"
    if occlusion_mode == "ensemble" and args.scene_depth is None:
        parser.error("--occlusion_mode ensemble requires --scene_depth")
    if occlusion_mode == "object3d" and args.object_surface_depth is None:
        parser.error(
            "--occlusion_mode object3d requires --object_surface_depth"
        )
    if occlusion_mode != "object3d" and (
        args.object3d_force_surface
        or args.object3d_temporal_max_gap_frames > 0
    ):
        parser.error("Object3D penetration controls require object3d mode")
    if args.object3d_force_surface and args.contact_interior_expand_px > 0:
        parser.error(
            "surface-force already uses full-finger support and cannot be "
            "combined with contact interior expansion"
        )

    config = OcclusionConfig(
        contact_score_threshold=args.contact_score_threshold,
        contact_point_threshold=args.contact_point_threshold,
        hidden_fraction_on=args.hidden_fraction_on,
        hidden_fraction_off=args.hidden_fraction_off,
        contact_radius_px=args.contact_radius_px,
        contact_interior_expand_px=args.contact_interior_expand_px,
        contact_interior_expand_cap_fraction=(
            args.contact_interior_expand_cap_fraction
        ),
        object_depth_erode_px=args.object_depth_erode_px,
        object_depth_margin_m=args.depth_margin_m,
        object_surface_min_samples=args.object_surface_min_samples,
        object_surface_contact_max_shift_m=(
            args.object_surface_contact_max_shift_m
        ),
        object_surface_contact_consistency_m=(
            args.object_surface_contact_consistency_m
        ),
        object3d_force_surface=args.object3d_force_surface,
        object3d_force_margin_m=args.object3d_force_margin_m,
        object3d_temporal_max_gap_frames=(
            args.object3d_temporal_max_gap_frames
        ),
        object3d_temporal_motion_px=args.object3d_temporal_motion_px,
        object3d_temporal_front_slack_m=(
            args.object3d_temporal_front_slack_m
        ),
        contact_depth_tolerance_m=args.contact_depth_tolerance_m,
        contact_depth_thickness_scale=(
            args.contact_depth_thickness_scale
        ),
        xhand_thumb_thickness_m=args.xhand_thumb_thickness_m,
        xhand_finger_thickness_m=args.xhand_finger_thickness_m,
        min_occlusion_run_frames=args.min_occlusion_run_frames,
    )
    config.validate()

    processed = args.processed_demo.resolve()
    episode = args.episode_dir.resolve()
    output_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else processed / "contact_occlusion"
    )
    background_path = (
        args.background.resolve()
        if args.background is not None
        else processed / "inpaint_processor" / "video_human_inpaint.mkv"
    )
    raw_video_path = (
        args.raw_video.resolve()
        if args.raw_video is not None
        else processed / "video_L.mp4"
    )
    hawor_path = (
        args.hawor_npz.resolve()
        if args.hawor_npz is not None
        else episode / "rgb_hawor" / "retarget_input.npz"
    )
    contact_dir = (
        args.contact_dir.resolve()
        if args.contact_dir is not None
        else episode / "contact"
    )
    aux_contact_dir = (
        args.aux_contact_dir.resolve()
        if args.aux_contact_dir is not None
        else None
    )
    overlay_dir = (
        args.overlay_dir.resolve()
        if args.overlay_dir is not None
        else processed / "overlay_processor"
    )
    overlay_manifest = json.loads(
        (overlay_dir / "manifest.json").read_text()
    )
    side = overlay_manifest.get("side")
    if side is None:
        with np.load(
            processed / "rb5_calibration" / "rb5_overlay_input.npz"
        ) as overlay_input:
            side = str(overlay_input["side"])
    side = str(side)
    if side not in {"left", "right"}:
        raise ValueError(f"invalid rendered hand side {side!r}")
    aux_side = args.aux_side or side
    if aux_contact_dir is None and (
        args.aux_frame_offset != 0 or args.aux_side is not None
    ):
        raise ValueError(
            "--aux_frame_offset/--aux_side require --aux_contact_dir"
        )

    robot_rgb = np.load(overlay_dir / "robot_rgb.npy", mmap_mode="r")
    robot_depth = np.load(overlay_dir / "robot_depth.npy", mmap_mode="r")
    robot_mask = np.load(overlay_dir / "robot_mask.npy", mmap_mode="r")
    finger_mask = np.load(
        overlay_dir / "robot_finger_mask.npy",
        mmap_mode="r",
    )
    finger_labels = np.load(
        overlay_dir / "robot_finger_labels.npy",
        mmap_mode="r",
    )
    width, height, frame_count, fps = _video_metadata(background_path)
    raw_width, raw_height, raw_frames, raw_fps = _video_metadata(raw_video_path)
    if robot_mask.ndim != 3:
        raise ValueError(f"robot_mask must be (T,H,W), got {robot_mask.shape}")
    overlay_height, overlay_width = robot_mask.shape[1:3]
    expected_shapes = {
        "robot_rgb": (frame_count, overlay_height, overlay_width, 3),
        "robot_depth": (frame_count, overlay_height, overlay_width),
        "robot_mask": (frame_count, overlay_height, overlay_width),
        "robot_finger_mask": (frame_count, overlay_height, overlay_width),
        "robot_finger_labels": (frame_count, overlay_height, overlay_width),
    }
    arrays = {
        "robot_rgb": robot_rgb,
        "robot_depth": robot_depth,
        "robot_mask": robot_mask,
        "robot_finger_mask": finger_mask,
        "robot_finger_labels": finger_labels,
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(
                f"{name} shape mismatch: {arrays[name].shape} != {expected}"
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
    if raw_frames != frame_count or not np.isclose(raw_fps, fps, atol=0.1):
        raise ValueError(
            "raw/background video mismatch: "
            f"frames {raw_frames}/{frame_count}, fps {raw_fps}/{fps}"
        )
    for frame_index in range(frame_count):
        if np.any(finger_mask[frame_index] & ~robot_mask[frame_index]):
            raise ValueError(
                "Isaac finger mask is not a robot-mask subset at frame "
                f"{frame_index}"
            )
        labels = np.asarray(finger_labels[frame_index])
        if labels.dtype != np.uint8 or np.any(labels > len(FINGER_NAMES)):
            raise ValueError(
                f"invalid robot finger labels at frame {frame_index}"
            )
        if not np.array_equal(labels > 0, finger_mask[frame_index]):
            raise ValueError(
                "robot finger label/mask mismatch at frame "
                f"{frame_index}"
            )

    source_images = sorted((episode / "rgb").glob("*.jpg"))
    if len(source_images) != frame_count:
        raise ValueError(
            f"expected {frame_count} JPG source frames, got {len(source_images)}"
        )
    contact_paths = [contact_dir / f"{path.stem}.npz" for path in source_images]
    if any(not path.is_file() for path in contact_paths):
        missing = [str(path) for path in contact_paths if not path.is_file()]
        raise FileNotFoundError(
            f"missing {len(missing)} HaCo frames; first={missing[0]}"
        )
    if aux_contact_dir is not None:
        if abs(args.aux_frame_offset) >= frame_count:
            raise ValueError(
                f"auxiliary frame offset {args.aux_frame_offset} is outside "
                f"a {frame_count}-frame sequence"
            )
        missing_auxiliary = []
        for frame_index in range(frame_count):
            auxiliary_index = frame_index + args.aux_frame_offset
            if not 0 <= auxiliary_index < frame_count:
                continue
            path = (
                aux_contact_dir
                / f"{source_images[auxiliary_index].stem}.npz"
            )
            if not path.is_file():
                missing_auxiliary.append(str(path))
        if missing_auxiliary:
            raise FileNotFoundError(
                "missing auxiliary HaCo frames; "
                f"first={missing_auxiliary[0]}, count={len(missing_auxiliary)}"
            )

    object_mask_array = (
        np.load(args.object_mask.resolve(), mmap_mode="r")
        if args.object_mask is not None
        else None
    )
    if object_mask_array is not None and len(object_mask_array) != frame_count:
        raise ValueError("object mask frame count mismatch")
    if args.object_restore_mask is not None and object_mask_array is None:
        raise ValueError(
            "--object_restore_mask requires an explicit --object_mask"
        )
    object_restore_mask_array = (
        np.load(args.object_restore_mask.resolve(), mmap_mode="r")
        if args.object_restore_mask is not None
        else object_mask_array
    )
    if object_restore_mask_array is not None:
        if (
            object_restore_mask_array.ndim != 3
            or len(object_restore_mask_array) != frame_count
        ):
            raise ValueError("object restore mask frame count/shape mismatch")
        assert object_mask_array is not None
        if object_restore_mask_array.shape != object_mask_array.shape:
            raise ValueError(
                "object restore mask must exactly align with object mask: "
                f"{object_restore_mask_array.shape} != {object_mask_array.shape}"
            )
        for frame_index in range(frame_count):
            restore = np.asarray(
                object_restore_mask_array[frame_index], dtype=bool
            )
            modal = np.asarray(object_mask_array[frame_index], dtype=bool)
            if np.any(restore & ~modal):
                raise ValueError(
                    "object restore mask is not a modal-object subset at "
                    f"frame {frame_index}"
                )
    visible_masks: np.ndarray | None = None
    hawor_amodal: np.ndarray | None = None
    if object_mask_array is None:
        visible_masks = np.load(
            processed / "segmentation_processor" / "masks_arm.npy",
            mmap_mode="r",
        )
        tracks = sorted(
            (episode / "rgb_hawor").glob("tracks_*/model_masks.npy")
        )
        if len(tracks) != 1:
            raise ValueError(f"expected one HaWoR model_masks.npy, got {tracks}")
        hawor_amodal = np.load(tracks[0], mmap_mode="r")
        if len(visible_masks) != frame_count or len(hawor_amodal) != frame_count:
            raise ValueError("human visibility masks are not frame-aligned")
    object_depth_track = np.full(frame_count, np.nan, dtype=np.float32)
    object_depth_mask_array = None
    object_surface_array = None
    if occlusion_mode == "ensemble":
        if object_mask_array is None and args.object_depth_mask is None:
            raise ValueError("--scene_depth requires an object mask")
        depth_array = np.load(args.scene_depth.resolve(), mmap_mode="r")
        object_depth_mask_array = (
            np.load(args.object_depth_mask.resolve(), mmap_mode="r")
            if args.object_depth_mask is not None
            else object_mask_array
        )
        if (
            len(depth_array) != frame_count
            or len(object_depth_mask_array) != frame_count
        ):
            raise ValueError("scene depth/object depth mask frame mismatch")
        object_depth_track = estimate_object_depth_track(
            depth_array,
            object_depth_mask_array,
            output_shape=(height, width),
            erode_px=config.object_depth_erode_px,
        )
    elif occlusion_mode == "object3d":
        if object_mask_array is None:
            raise ValueError(
                "object3d mode requires an explicit modal --object_mask"
            )
        object_surface_array = np.load(
            args.object_surface_depth.resolve(),
            mmap_mode="r",
        )
        if object_surface_array.ndim != 3:
            raise ValueError(
                "object surface depth must have shape (T,H,W), got "
                f"{object_surface_array.shape}"
            )
        if len(object_surface_array) != frame_count:
            raise ValueError("object surface depth frame count mismatch")

    assets = Path(__file__).resolve().parents[1] / "retargeting" / "assets"
    with np.load(hawor_path) as retarget:
        if retarget[f"verts_{side}"].shape[0] != frame_count:
            raise ValueError("HaWoR/overlay frame count mismatch")
        source_focal = float(retarget["img_focal"])
        if not np.isclose(
            width / raw_width,
            height / raw_height,
            atol=1e-3,
        ):
            raise ValueError("raw/background resize must preserve aspect scale")
        output_focal = source_focal * width / raw_width
        parts = np.load(assets / f"finger_part_{side}.npy").astype(np.int32)
        palmar = np.load(assets / f"palmar_mask_{side}.npy").astype(bool)

        primary_scores = np.zeros(
            (frame_count, len(FINGER_NAMES)),
            dtype=np.float32,
        )
        hidden_fraction = np.zeros_like(primary_scores)
        all_points_uv: list[dict[str, np.ndarray]] = []
        all_points_z: list[dict[str, np.ndarray]] = []
        raw_capture = cv2.VideoCapture(str(raw_video_path))
        bg_capture = cv2.VideoCapture(str(background_path))
        try:
            for frame_index in range(frame_count):
                ok_raw, raw = raw_capture.read()
                ok_bg, background = bg_capture.read()
                if not ok_raw or not ok_bg:
                    raise RuntimeError(
                        f"video read failed during evidence pass at {frame_index}"
                    )
                if raw.shape[:2] != (height, width):
                    raw = cv2.resize(
                        raw,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )
                frame_scores, points_uv, points_z = _contact_frame_features(
                    contact_path=contact_paths[frame_index],
                    retarget=retarget,
                    frame_index=frame_index,
                    side=side,
                    parts=parts,
                    palmar=palmar,
                    config=config,
                    focal_output_px=output_focal,
                    output_width=width,
                    output_height=height,
                )
                primary_scores[frame_index] = frame_scores
                all_points_uv.append(points_uv)
                all_points_z.append(points_z)

                if object_mask_array is not None:
                    # The dedicated SAM2 object track already uses both hands
                    # as negative prompts. Trust its modal boundary directly:
                    # masks_arm can include a held object and would otherwise
                    # erase exactly the evidence needed at a grasp.
                    evidence_mask = _resize_mask(
                        object_mask_array[frame_index],
                        width,
                        height,
                    )
                else:
                    assert visible_masks is not None
                    assert hawor_amodal is not None
                    visible = _resize_mask(
                        visible_masks[frame_index],
                        width,
                        height,
                    )
                    amodal = _resize_mask(
                        hawor_amodal[frame_index],
                        width,
                        height,
                    )
                    evidence_mask = proxy_occluder_mask(
                        raw,
                        background,
                        amodal,
                        visible,
                        lab_threshold=config.raw_bg_lab_threshold,
                    )
                for finger_index, finger in enumerate(FINGER_NAMES):
                    fractions = sample_local_fraction(
                        evidence_mask,
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
                        f"[evidence] {frame_index + 1}/{frame_count}",
                        flush=True,
                    )
        finally:
            raw_capture.release()
            bg_capture.release()

    auxiliary_scores: np.ndarray | None = None
    auxiliary_frame_lookup: list[int | None] = [None] * frame_count
    if aux_contact_dir is not None:
        auxiliary_parts = np.load(
            assets / f"finger_part_{aux_side}.npy"
        ).astype(np.int32)
        auxiliary_palmar = np.load(
            assets / f"palmar_mask_{aux_side}.npy"
        ).astype(bool)
        auxiliary_scores = np.full_like(primary_scores, np.nan)
        for frame_index in range(frame_count):
            auxiliary_index = frame_index + args.aux_frame_offset
            if not 0 <= auxiliary_index < frame_count:
                continue
            auxiliary_frame_lookup[frame_index] = auxiliary_index
            auxiliary_path = (
                aux_contact_dir
                / f"{source_images[auxiliary_index].stem}.npz"
            )
            frame_scores, _ = _contact_frame_selection(
                contact_path=auxiliary_path,
                frame_index=auxiliary_index,
                side=aux_side,
                parts=auxiliary_parts,
                palmar=auxiliary_palmar,
                config=config,
            )
            auxiliary_scores[frame_index] = frame_scores
    _, _, primary_active, _ = contact_activation_tracks(
        primary_scores,
        None,
        hidden_fraction,
        config,
    )
    scores, evidence, active, contact_gates = contact_activation_tracks(
        primary_scores,
        auxiliary_scores,
        hidden_fraction,
        config,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=".contact_occlusion.",
        dir=output_dir.parent,
    ))
    atexit.register(shutil.rmtree, staging, ignore_errors=True)
    occluded_buffer = np.lib.format.open_memmap(
        staging / "occluded_finger_mask.npy",
        mode="w+",
        dtype=bool,
        shape=(frame_count, height, width),
    )
    temporal_eligible_buffer = None
    temporal_labels_buffer = None
    temporal_eligible_path = staging / ".object3d_temporal_eligible.npy"
    temporal_labels_path = staging / ".object3d_temporal_labels.npy"
    if config.object3d_temporal_max_gap_frames > 0:
        temporal_eligible_buffer = np.lib.format.open_memmap(
            temporal_eligible_path,
            mode="w+",
            dtype=bool,
            shape=(frame_count, height, width),
        )
        temporal_labels_buffer = np.lib.format.open_memmap(
            temporal_labels_path,
            mode="w+",
            dtype=np.uint8,
            shape=(frame_count, height, width),
        )
    raw_capture = cv2.VideoCapture(str(raw_video_path))
    bg_capture = cv2.VideoCapture(str(background_path))
    candidate_presence = np.zeros(
        (frame_count, len(FINGER_NAMES)),
        dtype=bool,
    )
    object_surface_valid = np.zeros_like(candidate_presence)
    object_surface_consistent = np.zeros_like(candidate_presence)
    object_surface_sample_count = np.zeros(
        candidate_presence.shape,
        dtype=np.int32,
    )
    object_surface_inlier_count = np.zeros_like(
        object_surface_sample_count
    )
    object_surface_local_depth = np.full(
        candidate_presence.shape,
        np.nan,
        dtype=np.float32,
    )
    object_surface_contact_residual = np.full_like(
        object_surface_local_depth,
        np.nan,
    )
    object_surface_applied_shift = np.full_like(
        object_surface_local_depth,
        np.nan,
    )
    object3d_force_candidate_pixels = np.zeros(
        candidate_presence.shape,
        dtype=np.int64,
    )
    object3d_temporal_added_pixels = np.zeros_like(
        object3d_force_candidate_pixels
    )
    interior_seed_pixels = np.zeros(
        (frame_count, len(FINGER_NAMES)),
        dtype=np.int64,
    )
    interior_boundary_seed_pixels = np.zeros_like(interior_seed_pixels)
    interior_added_cap_pixels = np.zeros_like(interior_seed_pixels)
    interior_added_pixels = np.zeros_like(interior_seed_pixels)
    interior_cap_limited = np.zeros_like(candidate_presence)
    full_frame_support = np.ones((height, width), dtype=bool)
    try:
        for frame_index in range(frame_count):
            ok_raw, raw = raw_capture.read()
            ok_bg, background = bg_capture.read()
            if not ok_raw or not ok_bg:
                raise RuntimeError(
                    f"video read failed during mask pass at {frame_index}"
                )
            if raw.shape[:2] != (height, width):
                raw = cv2.resize(
                    raw,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            explicit_mask = (
                _resize_mask(
                    object_mask_array[frame_index],
                    width,
                    height,
                )
                if object_mask_array is not None
                else None
            )
            if explicit_mask is not None:
                visible = np.zeros((height, width), dtype=bool)
                amodal = visible
            else:
                assert visible_masks is not None
                assert hawor_amodal is not None
                visible = _resize_mask(
                    visible_masks[frame_index],
                    width,
                    height,
                )
                amodal = _resize_mask(
                    hawor_amodal[frame_index],
                    width,
                    height,
                )
            _, occluder = _frame_occluder(
                raw=raw,
                background=background,
                amodal=amodal,
                visible=visible,
                explicit_object_mask=explicit_mask,
                config=config,
            )

            frame_occluded = np.zeros((height, width), dtype=bool)
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
            frame_object_surface = None
            if object_surface_array is not None:
                frame_object_surface = resize_positive_depth(
                    object_surface_array[frame_index],
                    width,
                    height,
                )
            if temporal_eligible_buffer is not None:
                assert temporal_labels_buffer is not None
                assert frame_object_surface is not None
                temporal_eligible_buffer[frame_index] = (
                    object_surface_temporal_eligibility(
                        finger_mask=frame_finger_mask,
                        robot_depth=frame_robot_depth,
                        occluder_mask=occluder,
                        object_surface_depth=frame_object_surface,
                        front_slack_m=(
                            config.object3d_temporal_front_slack_m
                        ),
                    )
                )
                temporal_labels_buffer[frame_index] = frame_finger_labels
            if config.object3d_force_surface:
                assert frame_object_surface is not None
                for finger_index in range(len(FINGER_NAMES)):
                    semantic_finger = (
                        frame_finger_labels == finger_index + 1
                    )
                    force_candidate = compute_occluded_fingers_surface(
                        robot_mask=frame_robot_mask,
                        finger_mask=semantic_finger,
                        robot_depth=frame_robot_depth,
                        occluder_mask=occluder,
                        contact_support_mask=full_frame_support,
                        object_surface_depth=frame_object_surface,
                        surface_shift_m=0.0,
                        object_depth_margin_m=(
                            config.object3d_force_margin_m
                        ),
                    )
                    object3d_force_candidate_pixels[
                        frame_index, finger_index
                    ] = int(force_candidate.sum())
                    candidate_presence[
                        frame_index, finger_index
                    ] |= bool(force_candidate.any())
                    frame_occluded |= force_candidate
            for finger_index, finger in enumerate(FINGER_NAMES):
                if not active[frame_index, finger_index]:
                    continue
                support = disk_support(
                    all_points_uv[frame_index][finger],
                    (height, width),
                    config.contact_radius_px,
                )
                point_depths = all_points_z[frame_index][finger]
                contact_depth = (
                    float(np.median(point_depths))
                    if len(point_depths)
                    else math.nan
                )
                contact_depth_bias = config.contact_depth_bias_m(finger)
                semantic_finger = frame_finger_labels == finger_index + 1
                haco_occluded = compute_occluded_fingers(
                    robot_mask=frame_robot_mask,
                    finger_mask=semantic_finger,
                    robot_depth=frame_robot_depth,
                    occluder_mask=occluder,
                    contact_support_mask=support,
                    object_depth_m=math.nan,
                    contact_depth_m=contact_depth,
                    object_depth_margin_m=config.object_depth_margin_m,
                    contact_depth_tolerance_m=(
                        config.contact_depth_tolerance_m
                    ),
                    contact_depth_bias_m=contact_depth_bias,
                )
                if occlusion_mode == "ensemble":
                    depth_occluder = _resize_mask(
                        object_depth_mask_array[frame_index],
                        width,
                        height,
                    )
                    depth_occluded = compute_occluded_fingers(
                        robot_mask=frame_robot_mask,
                        finger_mask=(
                            frame_finger_labels == finger_index + 1
                        ),
                        robot_depth=frame_robot_depth,
                        occluder_mask=depth_occluder,
                        contact_support_mask=full_frame_support,
                        object_depth_m=float(
                            object_depth_track[frame_index]
                        ),
                        contact_depth_m=math.nan,
                        object_depth_margin_m=(
                            config.object_depth_margin_m
                        ),
                        contact_depth_tolerance_m=(
                            config.contact_depth_tolerance_m
                        ),
                    )
                    finger_occluded = haco_occluded & depth_occluded
                elif occlusion_mode == "object3d":
                    assert frame_object_surface is not None
                    alignment = object_surface_contact_alignment(
                        frame_object_surface,
                        occluder,
                        support,
                        contact_depth_m=contact_depth,
                        alignment=args.object_surface_alignment,
                        min_samples=config.object_surface_min_samples,
                        max_shift_m=(
                            config.object_surface_contact_max_shift_m
                        ),
                        consistency_m=(
                            config.object_surface_contact_consistency_m
                        ),
                    )
                    object_surface_valid[
                        frame_index, finger_index
                    ] = bool(alignment["valid"])
                    object_surface_consistent[
                        frame_index, finger_index
                    ] = bool(alignment["consistent"])
                    object_surface_sample_count[
                        frame_index, finger_index
                    ] = int(alignment["sample_count"])
                    object_surface_inlier_count[
                        frame_index, finger_index
                    ] = int(alignment["inlier_count"])
                    object_surface_local_depth[
                        frame_index, finger_index
                    ] = float(alignment["local_surface_depth_m"])
                    object_surface_contact_residual[
                        frame_index, finger_index
                    ] = float(alignment["contact_residual_m"])
                    object_surface_applied_shift[
                        frame_index, finger_index
                    ] = float(alignment["applied_shift_m"])
                    # In object3d mode HaCo selects the active semantic finger
                    # and its local MH support.  The estimated object surface,
                    # rather than the old MANO contact-Z proxy, owns the actual
                    # front/behind decision.  This is intentionally not an
                    # intersection with ``haco_occluded``: a real object-depth
                    # gate must be able to recover penetrations that the proxy
                    # missed.
                    finger_occluded = compute_occluded_fingers_surface(
                        robot_mask=frame_robot_mask,
                        finger_mask=semantic_finger,
                        robot_depth=frame_robot_depth,
                        occluder_mask=occluder,
                        contact_support_mask=support,
                        object_surface_depth=frame_object_surface,
                        surface_shift_m=float(
                            alignment["applied_shift_m"]
                        ),
                        object_depth_margin_m=(
                            config.object_depth_margin_m
                        ),
                    )
                    # A full-finger version is retained only as the verified
                    # eligibility region for the opt-in interior completion.
                    depth_occluded = compute_occluded_fingers_surface(
                        robot_mask=frame_robot_mask,
                        finger_mask=semantic_finger,
                        robot_depth=frame_robot_depth,
                        occluder_mask=occluder,
                        contact_support_mask=full_frame_support,
                        object_surface_depth=frame_object_surface,
                        surface_shift_m=float(
                            alignment["applied_shift_m"]
                        ),
                        object_depth_margin_m=(
                            config.object_depth_margin_m
                        ),
                    )
                else:
                    finger_occluded = haco_occluded
                    depth_occluded = full_frame_support

                if config.contact_interior_expand_px > 0:
                    # Remove only the contact-disk locality constraint. Every
                    # other MH constraint stays identical: semantic finger,
                    # modal object, finite render depth, and the same local
                    # HaCo contact-surface depth gate. Ensemble mode retains
                    # its independent sensor-depth intersection as well.
                    haco_eligible = compute_occluded_fingers(
                        robot_mask=frame_robot_mask,
                        finger_mask=semantic_finger,
                        robot_depth=frame_robot_depth,
                        occluder_mask=occluder,
                        contact_support_mask=full_frame_support,
                        object_depth_m=math.nan,
                        contact_depth_m=contact_depth,
                        object_depth_margin_m=(
                            config.object_depth_margin_m
                        ),
                        contact_depth_tolerance_m=(
                            config.contact_depth_tolerance_m
                        ),
                        contact_depth_bias_m=contact_depth_bias,
                    )
                    interior_eligible = haco_eligible & depth_occluded
                    finger_occluded, expansion = (
                        expand_verified_contact_interior(
                            finger_occluded,
                            eligible_mask=interior_eligible,
                            finger_mask=semantic_finger,
                            expand_px=(
                                config.contact_interior_expand_px
                            ),
                            added_cap_fraction=(
                                config.contact_interior_expand_cap_fraction
                            ),
                        )
                    )
                    interior_seed_pixels[frame_index, finger_index] = int(
                        expansion["seed_pixels"]
                    )
                    interior_boundary_seed_pixels[
                        frame_index, finger_index
                    ] = int(expansion["boundary_seed_pixels"])
                    interior_added_cap_pixels[
                        frame_index, finger_index
                    ] = int(expansion["added_cap_pixels"])
                    interior_added_pixels[frame_index, finger_index] = int(
                        expansion["added_pixels"]
                    )
                    interior_cap_limited[frame_index, finger_index] = bool(
                        expansion["cap_limited"]
                    )
                    added = finger_occluded & ~(
                        haco_occluded & depth_occluded
                    )
                    if np.any(added & ~interior_eligible):
                        raise RuntimeError(
                            "contact interior expansion escaped the verified "
                            "MH region"
                        )
                    if int(added.sum()) > int(
                        expansion["added_cap_pixels"]
                    ):
                        raise RuntimeError(
                            "contact interior expansion exceeded its pixel cap"
                        )
                candidate_presence[frame_index, finger_index] |= bool(
                    finger_occluded.any()
                )
                frame_occluded |= finger_occluded
            if np.any(frame_occluded & ~frame_finger_mask):
                raise RuntimeError(
                    f"non-finger occlusion invariant failed at {frame_index}"
                )
            occluded_buffer[frame_index] = frame_occluded
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[occlusion-mask] {frame_index + 1}/{frame_count}",
                    flush=True,
                )
    finally:
        raw_capture.release()
        bg_capture.release()
        occluded_buffer.flush()
        if temporal_eligible_buffer is not None:
            temporal_eligible_buffer.flush()
        if temporal_labels_buffer is not None:
            temporal_labels_buffer.flush()

    if config.object3d_force_surface:
        # Literal surface-force must not lose a valid one-frame behind-surface
        # result to the legacy short-ON-run cleanup.
        stable_presence = candidate_presence.copy()
    else:
        stable_presence = suppress_short_runs(
            candidate_presence,
            min_frames=config.min_occlusion_run_frames,
        )
    retained_interior_added_pixels = np.where(
        stable_presence,
        interior_added_pixels,
        0,
    )
    occluded_counts = np.zeros(frame_count, dtype=np.int64)
    for frame_index in range(frame_count):
        frame_occluded = np.asarray(
            occluded_buffer[frame_index],
            dtype=bool,
        ).copy()
        _, _, _, _, labels = _resize_overlay_frame(
            robot_rgb[frame_index],
            robot_depth[frame_index],
            robot_mask[frame_index],
            finger_labels[frame_index],
            width=width,
            height=height,
        )
        for finger_index in range(len(FINGER_NAMES)):
            if not stable_presence[frame_index, finger_index]:
                frame_occluded[labels == finger_index + 1] = False
        occluded_buffer[frame_index] = frame_occluded
        occluded_counts[frame_index] = int(frame_occluded.sum())
    occluded_buffer.flush()

    temporal_diagnostics: dict[str, np.ndarray | int] = {
        "added_per_frame_finger": object3d_temporal_added_pixels,
        "added_pixels": 0,
        "added_frames": 0,
        "added_frame_fingers": 0,
    }
    temporal_final_presence = stable_presence.copy()
    if temporal_eligible_buffer is not None:
        assert temporal_labels_buffer is not None
        temporal_filtered_path = staging / ".occluded_temporal_filtered.npy"
        temporal_filtered_buffer = np.lib.format.open_memmap(
            temporal_filtered_path,
            mode="w+",
            dtype=bool,
            shape=(frame_count, height, width),
        )
        _, temporal_diagnostics = bridge_short_occlusion_gaps(
            occluded_buffer,
            temporal_eligible_buffer,
            temporal_labels_buffer,
            max_gap_frames=config.object3d_temporal_max_gap_frames,
            motion_radius_px=config.object3d_temporal_motion_px,
            source_presence=stable_presence,
            output=temporal_filtered_buffer,
        )
        object3d_temporal_added_pixels[:] = np.asarray(
            temporal_diagnostics["added_per_frame_finger"],
            dtype=np.int64,
        )
        temporal_final_presence |= object3d_temporal_added_pixels > 0
        temporal_filtered_buffer.flush()
        for frame_index in range(frame_count):
            filtered_frame = np.asarray(
                temporal_filtered_buffer[frame_index],
                dtype=bool,
            )
            occluded_buffer[frame_index] = filtered_frame
            occluded_counts[frame_index] = int(filtered_frame.sum())
        occluded_buffer.flush()
        del temporal_filtered_buffer
        temporal_filtered_path.unlink(missing_ok=True)
        del temporal_eligible_buffer
        del temporal_labels_buffer
        temporal_eligible_buffer = None
        temporal_labels_buffer = None
        temporal_eligible_path.unlink(missing_ok=True)
        temporal_labels_path.unlink(missing_ok=True)

    final_writer = _open_writer(
        staging / "video_overlay_contact.mp4",
        fps,
        (width, height),
    )
    robot_writer = _open_writer(
        staging / "video_robot_only_contact.mp4",
        fps,
        (width, height),
    )
    debug_writer = _open_writer(
        staging / "debug_contact_occlusion.mp4",
        fps,
        (width, height),
    )
    raw_capture = cv2.VideoCapture(str(raw_video_path))
    bg_capture = cv2.VideoCapture(str(background_path))
    raw_object_counts = np.zeros(frame_count, dtype=np.int64)
    try:
        for frame_index in range(frame_count):
            ok_raw, raw = raw_capture.read()
            ok_bg, background = bg_capture.read()
            if not ok_raw or not ok_bg:
                raise RuntimeError(
                    f"video read failed during composite pass at {frame_index}"
                )
            if raw.shape[:2] != (height, width):
                raw = cv2.resize(
                    raw,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            explicit_mask = (
                _resize_mask(
                    object_mask_array[frame_index],
                    width,
                    height,
                )
                if object_mask_array is not None
                else None
            )
            if explicit_mask is not None:
                visible = np.zeros((height, width), dtype=bool)
                amodal = visible
            else:
                assert visible_masks is not None
                assert hawor_amodal is not None
                visible = _resize_mask(
                    visible_masks[frame_index],
                    width,
                    height,
                )
                amodal = _resize_mask(
                    hawor_amodal[frame_index],
                    width,
                    height,
                )
            core_occluder, occluder = _frame_occluder(
                raw=raw,
                background=background,
                amodal=amodal,
                visible=visible,
                explicit_object_mask=explicit_mask,
                config=config,
            )
            restore_object = (
                _resize_mask(
                    object_restore_mask_array[frame_index],
                    width,
                    height,
                )
                if object_restore_mask_array is not None
                else core_occluder
            )
            frame_support = np.zeros((height, width), dtype=bool)
            for finger_index, finger in enumerate(FINGER_NAMES):
                if active[frame_index, finger_index]:
                    frame_support |= disk_support(
                        all_points_uv[frame_index][finger],
                        (height, width),
                        config.contact_radius_px,
                    )
            frame_occluded = np.asarray(
                occluded_buffer[frame_index],
                dtype=bool,
            )
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
            # Restore the verified clean observed object, not merely its
            # overlap with an occluded finger.  A separately supplied restore
            # mask excludes hand-contested pixels while the full modal mask
            # remains authoritative for occlusion geometry.
            # Robot pixels that are physically in front remain visible because
            # they are composited after this background layer; only
            # frame_occluded robot pixels are removed.
            raw_object_pixels = restore_object
            composite_background = background.copy()
            composite_background[raw_object_pixels] = raw[raw_object_pixels]
            raw_object_counts[frame_index] = int(raw_object_pixels.sum())
            final, robot_only, _ = composite_frame(
                composite_background,
                frame_robot_rgb,
                frame_robot_mask,
                frame_finger_mask,
                frame_occluded,
                robot_edge_sigma_px=config.robot_edge_sigma_px,
                occlusion_edge_sigma_px=(
                    0.0
                    if object_mask_array is not None
                    else config.occlusion_edge_sigma_px
                ),
            )
            final_writer.write(final)
            robot_writer.write(robot_only)
            debug_writer.write(_debug_grid(
                raw,
                frame_finger_mask,
                frame_support,
                occluder,
                frame_robot_rgb,
                frame_robot_mask,
                frame_occluded,
                final,
            ))
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[composite] {frame_index + 1}/{frame_count} "
                    f"hidden_px={occluded_counts[frame_index]}",
                    flush=True,
                )
    finally:
        raw_capture.release()
        bg_capture.release()
        final_writer.release()
        robot_writer.release()
        debug_writer.release()
        occluded_buffer.flush()
        del occluded_buffer

    auxiliary_dominant = np.zeros_like(active)
    if auxiliary_scores is not None:
        auxiliary_dominant = (
            np.isfinite(auxiliary_scores)
            & (auxiliary_scores > primary_scores)
        )
    auxiliary_rescued_active = active & ~primary_active
    auxiliary_rescued_candidate = (
        candidate_presence & auxiliary_rescued_active
    )
    auxiliary_rescued_stable = stable_presence & auxiliary_rescued_active
    primary_projected_contact = np.asarray(
        [
            [len(points[finger]) > 0 for finger in FINGER_NAMES]
            for points in all_points_uv
        ],
        dtype=bool,
    )
    primary_contact_depth = np.asarray(
        [
            [
                bool(np.isfinite(depths[finger]).any())
                for finger in FINGER_NAMES
            ]
            for depths in all_points_z
        ],
        dtype=bool,
    )

    def _finger_counts(values: np.ndarray) -> dict[str, int]:
        return {
            finger: int(values[:, index].sum())
            for index, finger in enumerate(FINGER_NAMES)
        }

    def _finite_percentiles(values: np.ndarray) -> list[float] | None:
        finite = np.asarray(values, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return None
        return [
            float(value)
            for value in np.percentile(finite, (5, 50, 95))
        ]

    if occlusion_mode == "object3d":
        np.savez(
            staging / "object_surface_contact_evidence.npz",
            finger_names=np.asarray(FINGER_NAMES),
            active=active,
            valid=object_surface_valid,
            consistent=object_surface_consistent,
            sample_count=object_surface_sample_count,
            inlier_count=object_surface_inlier_count,
            local_surface_depth_m=object_surface_local_depth,
            contact_residual_m=object_surface_contact_residual,
            applied_shift_m=object_surface_applied_shift,
        )
        if (
            config.object3d_force_surface
            or config.object3d_temporal_max_gap_frames > 0
        ):
            np.savez(
                staging / "object3d_penetration_evidence.npz",
                finger_names=np.asarray(FINGER_NAMES),
                force_candidate_pixels=(
                    object3d_force_candidate_pixels
                ),
                temporal_added_pixels=(
                    object3d_temporal_added_pixels
                ),
                stable_presence_before_temporal=stable_presence,
                final_presence=temporal_final_presence,
            )

    report = {
        "schema_version": 1,
        "occlusion_mode": occlusion_mode,
        "mode": (
            "modal_object_mask"
            if object_mask_array is not None
            else "contact_local_proxy"
        ),
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "side": side,
        "config": asdict(config),
        "sources": {
            "processed_demo": str(processed),
            "episode_dir": str(episode),
            "background": str(background_path),
            "raw_video": str(raw_video_path),
            "hawor_npz": str(hawor_path),
            "contact_dir": str(contact_dir),
            "aux_contact_dir": (
                str(aux_contact_dir)
                if aux_contact_dir is not None
                else None
            ),
            "overlay_dir": str(overlay_dir),
            "object_mask": (
                str(args.object_mask.resolve())
                if args.object_mask is not None
                else None
            ),
            "object_restore_mask": (
                str(args.object_restore_mask.resolve())
                if args.object_restore_mask is not None
                else (
                    str(args.object_mask.resolve())
                    if args.object_mask is not None
                    else None
                )
            ),
            "scene_depth": (
                str(args.scene_depth.resolve())
                if args.scene_depth is not None
                else None
            ),
            "object_surface_depth": (
                str(args.object_surface_depth.resolve())
                if args.object_surface_depth is not None
                else None
            ),
        },
        "finger_names": list(FINGER_NAMES),
        "contact_fusion": (
            "per-finger maximum of primary/auxiliary HaCo scores"
            if auxiliary_scores is not None
            else "primary HaCo scores only"
        ),
        "contact_activation_policy": {
            "name": (
                "mh_geometry_with_sh_confidence_rescue"
                if auxiliary_scores is not None
                else "mh_branch_parity"
            ),
            "unit": "five MANO fingers",
            "primary_view_role": (
                "contact confidence, projected contact location, modal "
                "object overlap, and contact-surface depth"
            ),
            "auxiliary_view_role": (
                "same-finger contact-confidence proposal only"
                if auxiliary_scores is not None
                else None
            ),
            "auxiliary_geometry_used": False,
            "auxiliary_rescue_support_threshold": (
                float(config.hidden_fraction_off)
                if auxiliary_scores is not None
                else None
            ),
            "activation_on_support_threshold": float(
                config.hidden_fraction_on
            ),
            "auxiliary_frame_lookup": (
                auxiliary_frame_lookup
                if auxiliary_scores is not None
                else None
            ),
            "counts": {
                "auxiliary_available_frames": int(
                    np.any(
                        contact_gates["auxiliary_available"], axis=1
                    ).sum()
                ),
                "auxiliary_score_dominant_frame_fingers": int(
                    auxiliary_dominant.sum()
                ),
                "primary_threshold_frame_fingers": int(
                    contact_gates["primary"].sum()
                ),
                "auxiliary_threshold_frame_fingers": int(
                    contact_gates["auxiliary"].sum()
                ),
                "auxiliary_only_threshold_proposals": int(
                    contact_gates["auxiliary_proposal"].sum()
                ),
                "auxiliary_proposals_with_primary_local_support": int(
                    contact_gates["auxiliary_qualified"].sum()
                ),
                "auxiliary_proposals_with_primary_on_support": int(
                    (
                        contact_gates["auxiliary_proposal"]
                        & (hidden_fraction >= config.hidden_fraction_on)
                    ).sum()
                ),
                "active_frame_fingers_added_vs_primary": int(
                    auxiliary_rescued_active.sum()
                ),
                "candidate_frame_fingers_added_vs_primary": int(
                    auxiliary_rescued_candidate.sum()
                ),
                "stable_candidate_frame_fingers_added_vs_primary": int(
                    auxiliary_rescued_stable.sum()
                ),
                "primary_projected_contact_frame_fingers": int(
                    primary_projected_contact.sum()
                ),
                "primary_contact_depth_frame_fingers": int(
                    primary_contact_depth.sum()
                ),
            },
            "active_frame_fingers_added_by_finger": _finger_counts(
                auxiliary_rescued_active
            ),
        },
        "contact_interior_expansion": {
            "enabled": config.contact_interior_expand_px > 0,
            "policy": (
                "MH semantic-finger inner-boundary trigger with bounded "
                "3x3 geodesic growth"
            ),
            "geometry_view": "primary/MH only",
            "auxiliary_geometry_used": False,
            "expand_px": int(config.contact_interior_expand_px),
            "added_pixel_cap_fraction_of_verified_seed": float(
                config.contact_interior_expand_cap_fraction
            ),
            "verified_seed_pixels_total": int(
                interior_seed_pixels.sum()
            ),
            "boundary_seed_pixels_total": int(
                interior_boundary_seed_pixels.sum()
            ),
            "boundary_trigger_frame_fingers": int(
                (interior_boundary_seed_pixels > 0).sum()
            ),
            "expanded_frame_fingers": int(
                (interior_added_pixels > 0).sum()
            ),
            "cap_limited_frame_fingers": int(
                interior_cap_limited.sum()
            ),
            "added_pixels_before_temporal_suppression": int(
                interior_added_pixels.sum()
            ),
            "added_pixels_final": int(
                retained_interior_added_pixels.sum()
            ),
            "added_pixels_final_by_finger": _finger_counts(
                retained_interior_added_pixels
            ),
        },
        "xhand_contact_depth_bias": {
            "enabled": config.contact_depth_thickness_scale > 0.0,
            "policy": (
                "effective robot depth = rendered front depth + scale * "
                "per-finger full transverse thickness"
            ),
            "applies_to": "HaCo contact-surface proxy gate only",
            "metric_object_depth_gate_modified": False,
            "scale": float(config.contact_depth_thickness_scale),
            "full_thickness_m": {
                "thumb": float(config.xhand_thumb_thickness_m),
                "index": float(config.xhand_finger_thickness_m),
                "middle": float(config.xhand_finger_thickness_m),
                "ring": float(config.xhand_finger_thickness_m),
                "pinky": float(config.xhand_finger_thickness_m),
            },
            "applied_bias_m": {
                finger: config.contact_depth_bias_m(finger)
                for finger in FINGER_NAMES
            },
        },
        "object_surface_3d": {
            "enabled": occlusion_mode == "object3d",
            "representation": (
                "dense visible modal-object camera-Z surface"
                if occlusion_mode == "object3d"
                else None
            ),
            "depth_gate": (
                "rendered XHand front Z > "
                + (
                    "bounded contact-registered object surface Z + margin"
                    if args.object_surface_alignment == "contact"
                    else "object surface Z + margin"
                )
                if occlusion_mode == "object3d"
                else None
            ),
            "haco_role": (
                (
                    "baseline branch selects active finger/local MH support; "
                    "surface-force branch bypasses both"
                    if config.object3d_force_surface
                    else "active finger and local MH contact-support selector; "
                    "not the depth-order gate"
                )
                if occlusion_mode == "object3d"
                else None
            ),
            "alignment": (
                args.object_surface_alignment
                if occlusion_mode == "object3d"
                else None
            ),
            "geometry_view": "primary/MH only",
            "auxiliary_view_role": (
                "same-finger HaCo confidence rescue only"
                if auxiliary_scores is not None
                else None
            ),
            "active_frame_fingers": int(active.sum()),
            "valid_local_surface_frame_fingers": int(
                object_surface_valid.sum()
            ),
            "consistent_contact_frame_fingers": int(
                object_surface_consistent.sum()
            ),
            "contacts_within_max_shift_frame_fingers": int(
                (
                    object_surface_valid
                    & np.isfinite(object_surface_contact_residual)
                    & (
                        np.abs(object_surface_contact_residual)
                        <= config.object_surface_contact_max_shift_m + 1.0e-7
                    )
                ).sum()
            ),
            "fully_registered_frame_fingers": (
                int(object_surface_valid.sum())
                if (
                    occlusion_mode == "object3d"
                    and args.object_surface_alignment == "contact"
                )
                else 0
            ),
            "valid_by_finger": _finger_counts(object_surface_valid),
            "contact_residual_p05_median_p95_m": _finite_percentiles(
                object_surface_contact_residual[object_surface_valid]
            ),
            "applied_shift_p05_median_p95_m": _finite_percentiles(
                object_surface_applied_shift[object_surface_valid]
            ),
            "evidence_file": (
                "object_surface_contact_evidence.npz"
                if occlusion_mode == "object3d"
                else None
            ),
            "calibrated_sensor_depth": False,
            "provenance_warning": (
                "Depth Anything V2 surface is HaWoR-Z anchored and is an "
                "overlay-coordinate depth proxy, not independent ground truth."
                if occlusion_mode == "object3d"
                else None
            ),
        },
        "object3d_penetration_control": {
            "enabled": bool(
                occlusion_mode == "object3d"
                and (
                    config.object3d_force_surface
                    or config.object3d_temporal_max_gap_frames > 0
                )
            ),
            "surface_force": {
                "enabled": bool(config.object3d_force_surface),
                "policy": (
                    "baseline OR full semantic-finger pixels with rendered "
                    "Z > raw dense object-surface Z + force margin"
                    if config.object3d_force_surface
                    else None
                ),
                "haco_activation_used_for_added_branch": False,
                "contact_disk_used_for_added_branch": False,
                "contact_registration_used_for_added_branch": False,
                "includes_palm": False,
                "force_margin_m": float(config.object3d_force_margin_m),
                "candidate_pixels": int(
                    object3d_force_candidate_pixels.sum()
                ),
                "candidate_frame_fingers": int(
                    (object3d_force_candidate_pixels > 0).sum()
                ),
                "candidate_pixels_by_finger": _finger_counts(
                    object3d_force_candidate_pixels
                ),
                "short_on_run_suppression_bypassed": bool(
                    config.object3d_force_surface
                ),
            },
            "temporal_filter": {
                "enabled": bool(
                    config.object3d_temporal_max_gap_frames > 0
                ),
                "policy": (
                    "offline bidirectional same-finger spatiotemporal closing, "
                    "clipped to current modal-object/finger overlap with a "
                    "clear-front depth veto"
                    if config.object3d_temporal_max_gap_frames > 0
                    else None
                ),
                "uses_future_frames": bool(
                    config.object3d_temporal_max_gap_frames > 0
                ),
                "max_gap_frames": int(
                    config.object3d_temporal_max_gap_frames
                ),
                "motion_radius_px_per_frame": int(
                    config.object3d_temporal_motion_px
                ),
                "front_slack_m": float(
                    config.object3d_temporal_front_slack_m
                ),
                "added_pixels": int(temporal_diagnostics["added_pixels"]),
                "added_frames": int(temporal_diagnostics["added_frames"]),
                "added_frame_fingers": int(
                    temporal_diagnostics["added_frame_fingers"]
                ),
                "added_pixels_by_finger": _finger_counts(
                    object3d_temporal_added_pixels
                ),
            },
            "evidence_file": (
                "object3d_penetration_evidence.npz"
                if (
                    config.object3d_force_surface
                    or config.object3d_temporal_max_gap_frames > 0
                )
                else None
            ),
        },
        "aux_frame_offset": int(args.aux_frame_offset),
        "aux_side": aux_side if auxiliary_scores is not None else None,
        "contact_score": scores.round(6).tolist(),
        "contact_score_primary": primary_scores.round(6).tolist(),
        "contact_score_auxiliary": (
            auxiliary_scores.round(6).tolist()
            if auxiliary_scores is not None
            else None
        ),
        "contact_score_fused": scores.round(6).tolist(),
        "hidden_fraction": hidden_fraction.round(6).tolist(),
        "active_runs": {
            finger: _true_runs(active[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "active_runs_primary": {
            finger: _true_runs(primary_active[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "active_frame_count": {
            finger: int(active[:, index].sum())
            for index, finger in enumerate(FINGER_NAMES)
        },
        "active_frame_count_primary": _finger_counts(primary_active),
        "candidate_occlusion_runs": {
            finger: _true_runs(candidate_presence[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "stable_occlusion_runs": {
            finger: _true_runs(stable_presence[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "final_occlusion_runs": {
            finger: _true_runs(temporal_final_presence[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "suppressed_short_finger_frames": int(
            candidate_presence.sum() - stable_presence.sum()
        ),
        "object_depth_m": [
            float(value) if np.isfinite(value) else None
            for value in object_depth_track
        ],
        "occluded_pixel_count": occluded_counts.tolist(),
        "occluded_pixels_total": int(occluded_counts.sum()),
        "frames_with_occlusion": int((occluded_counts > 0).sum()),
        "raw_object_pixel_count": raw_object_counts.tolist(),
        "raw_object_pixels_total": int(raw_object_counts.sum()),
        "invariants": {
            "occluded_subset_of_robot_fingers": True,
            "occlusion_is_corresponding_finger_only": True,
            "explicit_object_mask_must_be_modal": True,
            "raw_rgb_restore_uses_object_restore_mask_only": True,
            "object_restore_defaults_to_object_mask_when_omitted": True,
            "object_restore_mask_subset_of_object_mask": True,
            "ambiguous_depth_fails_open": bool(
                config.object3d_temporal_max_gap_frames == 0
            ),
            "bounded_temporal_surface_holes_require_bidirectional_support": bool(
                config.object3d_temporal_max_gap_frames > 0
            ),
            "auxiliary_haco_is_confidence_only": True,
            "auxiliary_geometry_used": False,
            "primary_view_owns_contact_projection_and_depth": True,
            "xhand_thickness_bias_is_contact_proxy_only": True,
            "sensor_object_depth_gate_is_unbiased": True,
            "zero_xhand_thickness_scale_preserves_legacy_gate": True,
            "contact_interior_expansion_same_finger_only": True,
            "contact_interior_expansion_within_verified_mh_region": True,
            "contact_interior_expansion_respects_added_pixel_cap": bool(
                np.all(
                    interior_added_pixels <= interior_added_cap_pixels
                )
            ),
            "ensemble_requires_haco_and_sensor_depth": (
                occlusion_mode == "ensemble"
            ),
            "object3d_requires_haco_and_dense_object_surface": (
                occlusion_mode == "object3d"
                and not config.object3d_force_surface
            ),
            "object3d_surface_is_primary_mh_geometry": True,
            "object3d_haco_is_selector_only": (
                occlusion_mode == "object3d"
                and not config.object3d_force_surface
            ),
            "object3d_force_bypasses_haco_selector": bool(
                occlusion_mode == "object3d"
                and config.object3d_force_surface
            ),
            "object3d_temporal_filter_only_adds_occlusion": bool(
                np.all(object3d_temporal_added_pixels >= 0)
            ),
            "object3d_contact_alignment_is_bounded": bool(
                np.all(
                    np.abs(
                        object_surface_applied_shift[
                            np.isfinite(object_surface_applied_shift)
                        ]
                    )
                    <= config.object_surface_contact_max_shift_m + 1.0e-7
                )
            ),
        },
        "visibility_evidence": (
            "verified_modal_object_mask"
            if object_mask_array is not None
            else "partial_occlusion_proxy_no_joint_visibility"
        ),
        "compositing_order": (
            "inpainted_background_with_clean_raw_object_restore_then_robot"
            if object_mask_array is not None
            else "inpainted_background_then_contact_occluded_robot"
        ),
    }
    (staging / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    publish_directory(str(staging), str(output_dir))
    print(f"[ok] contact-aware overlay: {output_dir}", flush=True)
    print(
        f"[info] mode={occlusion_mode}, "
        f"occluded pixels={int(occluded_counts.sum())}, "
        f"frames={int((occluded_counts > 0).sum())}/{frame_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
