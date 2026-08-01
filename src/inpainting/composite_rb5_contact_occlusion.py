"""Contact-conditioned, finger-only object occlusion for RB5 + XHand.

The baseline compositor draws every robot pixel over the inpainted scene.  It
therefore makes a grasp look wrong whenever an object should cover the far
side of a robot finger.  This compositor changes only pixels that satisfy all
of the following:

1. Isaac semantic rendering says the pixel belongs to an XHand finger.
2. HaCo assigns high contact probability to the corresponding MANO finger.
3. Contact-local image evidence says that part of the human finger was hidden.
4. In HaCo mode the robot pixel is behind the projected HaCo contact surface.
   In ensemble mode it must additionally pass the sensor object-depth gate.

An explicit modal object mask and aligned scene depth are preferred.
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


@dataclass(frozen=True)
class OcclusionConfig:
    contact_score_threshold: float = 0.72
    contact_point_threshold: float = 0.78
    contact_top_fraction: float = 0.25
    min_contact_points: int = 6
    contact_radius_px: int = 22
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
    contact_depth_tolerance_m: float = 0.012
    robot_edge_sigma_px: float = 0.6
    occlusion_edge_sigma_px: float = 1.2

    def validate(self) -> None:
        probability_fields = (
            self.contact_score_threshold,
            self.contact_point_threshold,
            self.contact_top_fraction,
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
) -> np.ndarray:
    """Compute a fail-open finger-only mask for one frame/finger ROI."""
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
    candidate = robot & fingers & occluder & support & np.isfinite(depth)
    if np.isfinite(object_depth_m):
        depth_gate = depth > float(object_depth_m) + object_depth_margin_m
    elif np.isfinite(contact_depth_m):
        # A HaCo contact vertex lies on the human finger surface rather than
        # independently measured object geometry. Treat it as a local depth
        # proxy with a small tolerance, never as a global object plane.
        depth_gate = depth >= float(contact_depth_m) - contact_depth_tolerance_m
    else:
        return np.zeros_like(robot)
    return candidate & depth_gate


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
    scores = np.zeros(len(FINGER_NAMES), dtype=np.float32)
    points_uv: dict[str, np.ndarray] = {}
    points_z: dict[str, np.ndarray] = {}
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
    vertices = _camera_vertices(
        retarget[f"verts_{side}"][frame_index],
        retarget,
        frame_index,
    )
    for finger_index, finger in enumerate(FINGER_NAMES):
        eligible = (
            palmar
            & np.isin(parts, FINGER_PARTS[finger])
            & filtered_contact
        )
        vertex_indices = np.flatnonzero(eligible)
        if len(vertex_indices) < config.min_contact_points:
            points_uv[finger] = np.empty((0, 2), dtype=np.float32)
            points_z[finger] = np.empty(0, dtype=np.float32)
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
        selected = vertex_indices[selected_local]
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
        "--occlusion_mode",
        choices=("auto", "haco", "ensemble"),
        default="auto",
        help=(
            "haco uses the HaCo contact-surface depth only; ensemble requires "
            "both HaCo evidence and sensor scene depth. auto selects ensemble "
            "when --scene_depth is supplied, otherwise haco."
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
        "--contact_depth_tolerance_m",
        type=float,
        default=OcclusionConfig.contact_depth_tolerance_m,
    )
    parser.add_argument(
        "--min_occlusion_run_frames",
        type=int,
        default=OcclusionConfig.min_occlusion_run_frames,
    )
    args = parser.parse_args()
    occlusion_mode = args.occlusion_mode
    if occlusion_mode == "auto":
        occlusion_mode = (
            "ensemble" if args.scene_depth is not None else "haco"
        )
    if occlusion_mode == "ensemble" and args.scene_depth is None:
        parser.error("--occlusion_mode ensemble requires --scene_depth")

    config = OcclusionConfig(
        contact_score_threshold=args.contact_score_threshold,
        contact_point_threshold=args.contact_point_threshold,
        hidden_fraction_on=args.hidden_fraction_on,
        hidden_fraction_off=args.hidden_fraction_off,
        contact_radius_px=args.contact_radius_px,
        object_depth_erode_px=args.object_depth_erode_px,
        object_depth_margin_m=args.depth_margin_m,
        contact_depth_tolerance_m=args.contact_depth_tolerance_m,
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

    visible_masks = np.load(
        processed / "segmentation_processor" / "masks_arm.npy",
        mmap_mode="r",
    )
    tracks = sorted((episode / "rgb_hawor").glob("tracks_*/model_masks.npy"))
    if len(tracks) != 1:
        raise ValueError(f"expected one HaWoR model_masks.npy, got {tracks}")
    hawor_amodal = np.load(tracks[0], mmap_mode="r")
    if len(visible_masks) != frame_count or len(hawor_amodal) != frame_count:
        raise ValueError("human visibility masks are not frame-aligned")

    object_mask_array = (
        np.load(args.object_mask.resolve(), mmap_mode="r")
        if args.object_mask is not None
        else None
    )
    if object_mask_array is not None and len(object_mask_array) != frame_count:
        raise ValueError("object mask frame count mismatch")
    object_depth_track = np.full(frame_count, np.nan, dtype=np.float32)
    object_depth_mask_array = None
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

        scores = np.zeros((frame_count, len(FINGER_NAMES)), dtype=np.float32)
        hidden_fraction = np.zeros_like(scores)
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
                scores[frame_index] = frame_scores
                all_points_uv.append(points_uv)
                all_points_z.append(points_z)

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

    evidence = np.where(
        scores >= config.contact_score_threshold,
        hidden_fraction,
        0.0,
    )
    active = np.zeros_like(evidence, dtype=bool)
    for finger_index in range(len(FINGER_NAMES)):
        active[:, finger_index] = temporal_hysteresis(
            evidence[:, finger_index],
            on_threshold=config.hidden_fraction_on,
            off_threshold=config.hidden_fraction_off,
            min_on_frames=config.min_on_frames,
            hold_frames=config.hold_frames,
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
    raw_capture = cv2.VideoCapture(str(raw_video_path))
    bg_capture = cv2.VideoCapture(str(background_path))
    candidate_presence = np.zeros(
        (frame_count, len(FINGER_NAMES)),
        dtype=bool,
    )
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
            explicit_mask = (
                _resize_mask(
                    object_mask_array[frame_index],
                    width,
                    height,
                )
                if object_mask_array is not None
                else None
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
                haco_occluded = compute_occluded_fingers(
                    robot_mask=frame_robot_mask,
                    finger_mask=frame_finger_labels == finger_index + 1,
                    robot_depth=frame_robot_depth,
                    occluder_mask=occluder,
                    contact_support_mask=support,
                    object_depth_m=math.nan,
                    contact_depth_m=contact_depth,
                    object_depth_margin_m=config.object_depth_margin_m,
                    contact_depth_tolerance_m=(
                        config.contact_depth_tolerance_m
                    ),
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
                else:
                    finger_occluded = haco_occluded
                candidate_presence[frame_index, finger_index] = bool(
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

    stable_presence = suppress_short_runs(
        candidate_presence,
        min_frames=config.min_occlusion_run_frames,
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
            explicit_mask = (
                _resize_mask(
                    object_mask_array[frame_index],
                    width,
                    height,
                )
                if object_mask_array is not None
                else None
            )
            core_occluder, occluder = _frame_occluder(
                raw=raw,
                background=background,
                amodal=amodal,
                visible=visible,
                explicit_object_mask=explicit_mask,
                config=config,
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
            # Restore the complete visible object, not merely its overlap with
            # an occluded finger. Otherwise the inpainted background erases
            # most of the manipulated object and leaves a floating fragment.
            # The object segmenter uses human-hand negative prompts, so the
            # broader arm mask is intentionally not subtracted here.
            # Robot pixels that are physically in front remain visible because
            # they are composited after this background layer; only
            # frame_occluded robot pixels are removed.
            raw_object_pixels = core_occluder
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
            "overlay_dir": str(overlay_dir),
            "object_mask": (
                str(args.object_mask.resolve())
                if args.object_mask is not None
                else None
            ),
            "scene_depth": (
                str(args.scene_depth.resolve())
                if args.scene_depth is not None
                else None
            ),
        },
        "finger_names": list(FINGER_NAMES),
        "contact_score": scores.round(6).tolist(),
        "hidden_fraction": hidden_fraction.round(6).tolist(),
        "active_runs": {
            finger: _true_runs(active[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "active_frame_count": {
            finger: int(active[:, index].sum())
            for index, finger in enumerate(FINGER_NAMES)
        },
        "candidate_occlusion_runs": {
            finger: _true_runs(candidate_presence[:, index])
            for index, finger in enumerate(FINGER_NAMES)
        },
        "stable_occlusion_runs": {
            finger: _true_runs(stable_presence[:, index])
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
            "ambiguous_depth_fails_open": True,
            "ensemble_requires_haco_and_sensor_depth": (
                occlusion_mode == "ensemble"
            ),
        },
        "visibility_evidence": (
            "verified_modal_object_mask"
            if object_mask_array is not None
            else "partial_occlusion_proxy_no_joint_visibility"
        ),
        "compositing_order": (
            "inpainted_background_then_robot_then_opaque_raw_object_core"
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
