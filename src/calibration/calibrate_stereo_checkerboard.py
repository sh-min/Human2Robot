"""Calibrate a fixed RGB stereo pair from indexed checkerboard images.

The input may be either a ZIP archive or an extracted directory containing
``camera_1/<index>_Color.png`` and ``camera_2/<index>_Color.png``.  The two
images with the same index are treated as a candidate stereo pair.  Mono
intrinsics are fitted independently, then the candidate pairs are checked for
180-degree checkerboard-order ambiguity and robustly filtered before fitting
``T_camera2_from_camera1``.

When ``--square-size-mm`` is omitted, translation is reported in checker-square
units.  Intrinsics and relative rotation remain valid, but metric translation
must not be consumed until the physical square edge has been measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


_IMAGE_RE = re.compile(
    r"(?:^|/)camera_(?P<camera>[12])/(?P<frame>[0-9]+)_Color[.]png$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_images(source: Path) -> tuple[dict[int, dict[int, np.ndarray]], dict]:
    images: dict[int, dict[int, np.ndarray]] = {1: {}, 2: {}}
    entries: dict[int, dict[int, str]] = {1: {}, 2: {}}

    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                match = _IMAGE_RE.search(name)
                if match is None:
                    continue
                camera = int(match.group("camera"))
                frame = int(match.group("frame"))
                if frame in images[camera]:
                    raise ValueError(f"duplicate camera_{camera} frame {frame}")
                payload = np.frombuffer(archive.read(name), dtype=np.uint8)
                image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"failed to decode {name}")
                images[camera][frame] = image
                entries[camera][frame] = name
        source_meta = {
            "kind": "zip",
            "path": str(source.resolve()),
            "sha256": _sha256(source),
        }
    elif source.is_dir():
        for path in source.rglob("*_Color.png"):
            match = _IMAGE_RE.search(path.as_posix())
            if match is None:
                continue
            camera = int(match.group("camera"))
            frame = int(match.group("frame"))
            if frame in images[camera]:
                raise ValueError(f"duplicate camera_{camera} frame {frame}")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode {path}")
            images[camera][frame] = image
            entries[camera][frame] = str(path.resolve())
        source_meta = {"kind": "directory", "path": str(source.resolve())}
    else:
        raise ValueError(f"input is neither a ZIP nor directory: {source}")

    if not images[1] or not images[2]:
        raise ValueError("expected camera_1 and camera_2 PNG images")

    shapes = {
        tuple(image.shape)
        for camera_images in images.values()
        for image in camera_images.values()
    }
    if len(shapes) != 1:
        raise ValueError(f"all images must share one shape, got {sorted(shapes)}")
    height, width = next(iter(shapes))[:2]
    source_meta.update(
        {
            "entries": entries,
            "frame_ids": {
                str(camera): sorted(camera_images)
                for camera, camera_images in images.items()
            },
            "image_size_wh": [width, height],
        }
    )
    return images, source_meta


def _detect_corners(
    image: np.ndarray, pattern: tuple[int, int]
) -> tuple[np.ndarray | None, str | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sb_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, sb_flags)
    if ok:
        return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2), "sb"

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, classic_flags)
    if not ok:
        return None, None
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (7, 7),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4),
    )
    return np.asarray(refined, dtype=np.float32).reshape(-1, 1, 2), "classic"


def _detect_all(
    images: dict[int, dict[int, np.ndarray]], pattern: tuple[int, int]
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, dict[int, str]], dict]:
    detections: dict[int, dict[int, np.ndarray]] = {1: {}, 2: {}}
    methods: dict[int, dict[int, str]] = {1: {}, 2: {}}
    coverage: dict[int, dict[int, float]] = {1: {}, 2: {}}
    for camera, camera_images in images.items():
        for frame, image in sorted(camera_images.items()):
            corners, method = _detect_corners(image, pattern)
            if corners is None or method is None:
                continue
            detections[camera][frame] = corners
            methods[camera][frame] = method
            hull = cv2.convexHull(corners.astype(np.float32))
            coverage[camera][frame] = float(
                cv2.contourArea(hull) / (image.shape[0] * image.shape[1])
            )
    report = {
        str(camera): {
            "detected_ids": sorted(detections[camera]),
            "failed_ids": sorted(set(images[camera]) - set(detections[camera])),
            "method_by_id": {
                str(frame): methods[camera][frame]
                for frame in sorted(methods[camera])
            },
            "coverage_fraction_by_id": {
                str(frame): coverage[camera][frame]
                for frame in sorted(coverage[camera])
            },
        }
        for camera in (1, 2)
    }
    return detections, methods, report


def _object_points(pattern: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= float(square_size)
    return points


def _mad_threshold(values: np.ndarray, floor: float, scale: float = 3.5) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    return max(float(floor), median + scale * robust_sigma)


def _calibrate_once(
    ids: list[int],
    detections: dict[int, np.ndarray],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    fix_aspect_ratio: bool,
) -> dict:
    object_points = [object_template.copy() for _ in ids]
    image_points = [detections[frame] for frame in ids]
    flags = 0
    camera_matrix = None
    if fix_aspect_ratio:
        initial = cv2.initCameraMatrix2D(object_points, image_points, image_size)
        focal = float((initial[0, 0] + initial[1, 1]) * 0.5)
        camera_matrix = np.array(
            [
                [focal, 0.0, image_size[0] * 0.5],
                [0.0, focal, image_size[1] * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    (
        rms,
        camera_matrix,
        distortion,
        rvecs,
        tvecs,
        std_intrinsics,
        std_extrinsics,
        per_view_errors,
    ) = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        None,
        flags=flags,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            200,
            1e-10,
        ),
    )
    return {
        "ids": ids,
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "distortion": distortion,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "std_intrinsics": np.asarray(std_intrinsics).reshape(-1),
        "std_extrinsics": np.asarray(std_extrinsics).reshape(-1),
        "per_view_errors": np.asarray(per_view_errors).reshape(-1),
    }


def _calibrate_camera_robust(
    detections: dict[int, np.ndarray],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    *,
    fix_aspect_ratio: bool,
    error_floor_px: float,
    min_views: int,
) -> tuple[dict, list[dict]]:
    active = sorted(detections)
    if len(active) < min_views:
        raise ValueError(
            f"only {len(active)} mono detections, at least {min_views} required"
        )
    rejected: list[dict] = []
    while True:
        fit = _calibrate_once(
            active,
            detections,
            object_template,
            image_size,
            fix_aspect_ratio,
        )
        errors = fit["per_view_errors"]
        threshold = _mad_threshold(errors, error_floor_px)
        worst_index = int(np.argmax(errors))
        worst_error = float(errors[worst_index])
        if worst_error <= threshold or len(active) <= min_views:
            fit["outlier_threshold_px"] = threshold
            return fit, rejected
        rejected.append(
            {
                "frame_id": active[worst_index],
                "error_px": worst_error,
                "threshold_px": threshold,
            }
        )
        del active[worst_index]


def _sampson_rms(
    fundamental: np.ndarray, points1: np.ndarray, points2: np.ndarray
) -> float:
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    x1 = np.column_stack([p1, np.ones(len(p1))])
    x2 = np.column_stack([p2, np.ones(len(p2))])
    lines2 = (fundamental @ x1.T).T
    lines1 = (fundamental.T @ x2.T).T
    numerators = np.sum(x2 * lines2, axis=1) ** 2
    denominators = (
        lines1[:, 0] ** 2
        + lines1[:, 1] ** 2
        + lines2[:, 0] ** 2
        + lines2[:, 1] ** 2
    )
    distances = numerators / np.maximum(denominators, 1e-12)
    return float(np.sqrt(np.mean(distances)))


def _undistort_pixels(
    points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    """Return ideal pinhole pixel coordinates for distorted observations."""
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        camera_matrix,
        distortion,
        P=camera_matrix,
    ).reshape(-1, 1, 2)


def _calibrated_epipolar_rms(
    fundamental: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    camera_matrix1: np.ndarray,
    distortion1: np.ndarray,
    camera_matrix2: np.ndarray,
    distortion2: np.ndarray,
) -> float:
    return _sampson_rms(
        fundamental,
        _undistort_pixels(points1, camera_matrix1, distortion1),
        _undistort_pixels(points2, camera_matrix2, distortion2),
    )


def _initial_fundamental(
    pair_ids: Iterable[int],
    points1: dict[int, np.ndarray],
    points2: dict[int, np.ndarray],
) -> np.ndarray | None:
    first = np.concatenate(
        [points1[frame].reshape(-1, 2) for frame in pair_ids], axis=0
    )
    second = np.concatenate(
        [points2[frame].reshape(-1, 2) for frame in pair_ids], axis=0
    )
    fundamental, _ = cv2.findFundamentalMat(
        first,
        second,
        cv2.FM_RANSAC,
        1.0,
        0.999,
    )
    if fundamental is None or np.asarray(fundamental).shape != (3, 3):
        return None
    return np.asarray(fundamental, dtype=np.float64)


def _stereo_once(
    ids: list[int],
    points1: dict[int, np.ndarray],
    points2: dict[int, np.ndarray],
    reversed_ids: set[int],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    mono1: dict,
    mono2: dict,
) -> dict:
    selected2 = {
        frame: (points2[frame][::-1].copy() if frame in reversed_ids else points2[frame])
        for frame in ids
    }
    result = cv2.stereoCalibrate(
        [object_template.copy() for _ in ids],
        [points1[frame] for frame in ids],
        [selected2[frame] for frame in ids],
        mono1["camera_matrix"].copy(),
        mono1["distortion"].copy(),
        mono2["camera_matrix"].copy(),
        mono2["distortion"].copy(),
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            300,
            1e-10,
        ),
    )
    rms, k1, d1, k2, d2, rotation, translation, essential, fundamental = result
    pair_errors = {
        frame: _calibrated_epipolar_rms(
            fundamental,
            points1[frame],
            selected2[frame],
            k1,
            d1,
            k2,
            d2,
        )
        for frame in ids
    }
    return {
        "ids": ids,
        "selected_points2": selected2,
        "reversed_ids": set(reversed_ids),
        "rms": float(rms),
        "camera_matrix1": k1,
        "distortion1": d1,
        "camera_matrix2": k2,
        "distortion2": d2,
        "rotation": rotation,
        "translation": translation.reshape(3),
        "essential": essential,
        "fundamental": fundamental,
        "pair_epipolar_rms_px": pair_errors,
    }


def _calibrate_stereo_robust(
    points1: dict[int, np.ndarray],
    points2: dict[int, np.ndarray],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    mono1: dict,
    mono2: dict,
    *,
    error_floor_px: float,
    min_pairs: int,
) -> tuple[dict, list[dict]]:
    active = sorted(set(points1) & set(points2))
    if len(active) < min_pairs:
        raise ValueError(
            f"only {len(active)} stereo pairs, at least {min_pairs} required"
        )

    reversed_ids: set[int] = set()
    initial_f = _initial_fundamental(active, points1, points2)
    if initial_f is not None:
        for frame in active:
            original = _sampson_rms(initial_f, points1[frame], points2[frame])
            reversed_error = _sampson_rms(
                initial_f,
                points1[frame],
                points2[frame][::-1],
            )
            if reversed_error < original:
                reversed_ids.add(frame)

    rejected: list[dict] = []
    while True:
        # Refine the per-pair 180-degree decision against the physical stereo F.
        for _ in range(3):
            fit = _stereo_once(
                active,
                points1,
                points2,
                reversed_ids,
                object_template,
                image_size,
                mono1,
                mono2,
            )
            updated: set[int] = set()
            for frame in active:
                original = _calibrated_epipolar_rms(
                    fit["fundamental"],
                    points1[frame],
                    points2[frame],
                    fit["camera_matrix1"],
                    fit["distortion1"],
                    fit["camera_matrix2"],
                    fit["distortion2"],
                )
                reversed_error = _calibrated_epipolar_rms(
                    fit["fundamental"],
                    points1[frame],
                    points2[frame][::-1],
                    fit["camera_matrix1"],
                    fit["distortion1"],
                    fit["camera_matrix2"],
                    fit["distortion2"],
                )
                if reversed_error < original:
                    updated.add(frame)
            if updated == reversed_ids:
                break
            reversed_ids = updated

        fit = _stereo_once(
            active,
            points1,
            points2,
            reversed_ids,
            object_template,
            image_size,
            mono1,
            mono2,
        )
        errors = np.array(
            [fit["pair_epipolar_rms_px"][frame] for frame in active],
            dtype=np.float64,
        )
        threshold = _mad_threshold(errors, error_floor_px)
        worst_index = int(np.argmax(errors))
        worst_error = float(errors[worst_index])
        if worst_error <= threshold or len(active) <= min_pairs:
            fit["outlier_threshold_px"] = threshold
            return fit, rejected
        frame = active[worst_index]
        rejected.append(
            {
                "frame_id": frame,
                "epipolar_rms_px": worst_error,
                "threshold_px": threshold,
            }
        )
        active.remove(frame)
        reversed_ids.discard(frame)


def _matrix4(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _pnp_pair_consistency(
    fit: dict,
    points1: dict[int, np.ndarray],
    object_template: np.ndarray,
) -> dict[int, dict]:
    expected_rotation = fit["rotation"]
    expected_translation = fit["translation"]
    results: dict[int, dict] = {}
    for frame in fit["ids"]:
        ok1, rvec1, tvec1 = cv2.solvePnP(
            object_template,
            points1[frame],
            fit["camera_matrix1"],
            fit["distortion1"],
        )
        ok2, rvec2, tvec2 = cv2.solvePnP(
            object_template,
            fit["selected_points2"][frame],
            fit["camera_matrix2"],
            fit["distortion2"],
        )
        if not ok1 or not ok2:
            continue
        rotation1 = cv2.Rodrigues(rvec1)[0]
        rotation2 = cv2.Rodrigues(rvec2)[0]
        relative_rotation = rotation2 @ rotation1.T
        relative_translation = tvec2.reshape(3) - relative_rotation @ tvec1.reshape(3)
        results[frame] = {
            "rotation_deviation_deg": _rotation_angle_degrees(
                relative_rotation @ expected_rotation.T
            ),
            "translation_deviation": float(
                np.linalg.norm(relative_translation - expected_translation)
            ),
        }
    return results


def _stereo_transfer_errors(
    fit: dict,
    points1: dict[int, np.ndarray],
    object_template: np.ndarray,
) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    """Project each camera-1 board pose into camera 2 and score the transfer."""
    errors: dict[int, float] = {}
    projected_by_frame: dict[int, np.ndarray] = {}
    for frame in fit["ids"]:
        ok, rvec1, tvec1 = cv2.solvePnP(
            object_template,
            points1[frame],
            fit["camera_matrix1"],
            fit["distortion1"],
        )
        if not ok:
            continue
        rotation1 = cv2.Rodrigues(rvec1)[0]
        rotation2 = fit["rotation"] @ rotation1
        translation2 = (
            fit["rotation"] @ tvec1.reshape(3) + fit["translation"]
        )
        projected, _ = cv2.projectPoints(
            object_template,
            cv2.Rodrigues(rotation2)[0],
            translation2,
            fit["camera_matrix2"],
            fit["distortion2"],
        )
        projected = projected.reshape(-1, 2)
        observed = fit["selected_points2"][frame].reshape(-1, 2)
        errors[frame] = float(
            np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1)))
        )
        projected_by_frame[frame] = projected
    return errors, projected_by_frame


def _stats(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _camera_report(fit: dict, rejected: list[dict], free_fit: dict) -> dict:
    return {
        "model": "opencv_brown_5_coefficients",
        "square_pixel_aspect_ratio_constrained": True,
        "camera_matrix": fit["camera_matrix"].tolist(),
        "distortion_k1_k2_p1_p2_k3": fit["distortion"].reshape(-1).tolist(),
        "rms_reprojection_px": fit["rms"],
        "per_view_error_px": {
            str(frame): float(error)
            for frame, error in zip(fit["ids"], fit["per_view_errors"])
        },
        "per_view_error_stats_px": _stats(fit["per_view_errors"]),
        "used_frame_ids": fit["ids"],
        "rejected_outliers": rejected,
        "outlier_threshold_px": fit["outlier_threshold_px"],
        "intrinsic_parameter_stddev": fit["std_intrinsics"].tolist(),
        "free_aspect_diagnostic": {
            "rms_reprojection_px": free_fit["rms"],
            "camera_matrix": free_fit["camera_matrix"].tolist(),
            "fx_over_fy": float(
                free_fit["camera_matrix"][0, 0]
                / free_fit["camera_matrix"][1, 1]
            ),
        },
    }


def _make_qa(
    qa_dir: Path,
    images: dict[int, dict[int, np.ndarray]],
    detections: dict[int, dict[int, np.ndarray]],
    pattern: tuple[int, int],
    fit: dict,
    object_template: np.ndarray,
    image_size: tuple[int, int],
) -> dict:
    qa_dir.mkdir(parents=True, exist_ok=True)
    transfer_errors, projected_by_frame = _stereo_transfer_errors(
        fit, detections[1], object_template
    )
    best_frame = min(
        fit["ids"],
        key=lambda frame: (
            fit["pair_epipolar_rms_px"][frame]
            + transfer_errors.get(frame, float("inf"))
        ),
    )
    raw1 = images[1][best_frame].copy()
    raw2 = images[2][best_frame].copy()
    cv2.drawChessboardCorners(raw1, pattern, detections[1][best_frame], True)
    cv2.drawChessboardCorners(
        raw2, pattern, fit["selected_points2"][best_frame], True
    )
    corner_path = qa_dir / f"frame_{best_frame:02d}_corners.png"
    cv2.imwrite(str(corner_path), np.hstack([raw1, raw2]))

    transfer = images[2][best_frame].copy()
    observed = fit["selected_points2"][best_frame].reshape(-1, 2)
    projected = projected_by_frame[best_frame]
    for observation, prediction in zip(observed, projected):
        ox, oy = np.rint(observation).astype(int)
        px, py = np.rint(prediction).astype(int)
        cv2.circle(transfer, (ox, oy), 3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(
            transfer,
            (px - 3, py - 3),
            (px + 3, py + 3),
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            transfer,
            (px - 3, py + 3),
            (px + 3, py - 3),
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        transfer,
        (
            f"green=observed magenta=cam1-to-cam2 prediction  "
            f"RMS={transfer_errors[best_frame]:.3f}px"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    transfer_path = qa_dir / f"frame_{best_frame:02d}_stereo_transfer.png"
    cv2.imwrite(str(transfer_path), transfer)

    rotation1, rotation2, projection1, projection2, q, roi1, roi2 = (
        cv2.stereoRectify(
            fit["camera_matrix1"],
            fit["distortion1"],
            fit["camera_matrix2"],
            fit["distortion2"],
            image_size,
            fit["rotation"],
            fit["translation"],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=1,
        )
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        fit["camera_matrix1"],
        fit["distortion1"],
        rotation1,
        projection1,
        image_size,
        cv2.CV_32FC1,
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        fit["camera_matrix2"],
        fit["distortion2"],
        rotation2,
        projection2,
        image_size,
        cv2.CV_32FC1,
    )
    rectified1 = cv2.remap(images[1][best_frame], map1x, map1y, cv2.INTER_LINEAR)
    rectified2 = cv2.remap(images[2][best_frame], map2x, map2y, cv2.INTER_LINEAR)
    rectified = np.hstack([rectified1, rectified2])
    for y in range(40, image_size[1], 60):
        cv2.line(rectified, (0, y), (image_size[0] * 2 - 1, y), (0, 255, 0), 1)
    rectified_path = qa_dir / f"frame_{best_frame:02d}_rectified.png"
    cv2.imwrite(str(rectified_path), rectified)
    return {
        "selected_frame_id": best_frame,
        "corner_preview": str(corner_path.resolve()),
        "stereo_transfer_preview": str(transfer_path.resolve()),
        "stereo_transfer_rms_px": {
            str(frame): error for frame, error in transfer_errors.items()
        },
        "stereo_transfer_error_stats_px": _stats(transfer_errors.values()),
        "rectified_preview": str(rectified_path.resolve()),
        "rectification": {
            "R1": rotation1.tolist(),
            "R2": rotation2.tolist(),
            "P1": projection1.tolist(),
            "P2": projection2.tolist(),
            "Q": q.tolist(),
            "valid_roi1_xywh": list(map(int, roi1)),
            "valid_roi2_xywh": list(map(int, roi2)),
            "full_frame_common_roi_available": bool(roi1[2] and roi1[3] and roi2[2] and roi2[3]),
        },
    }


def calibrate(args: argparse.Namespace) -> dict:
    source = Path(args.input).resolve()
    images, source_meta = _read_images(source)
    image_size = tuple(source_meta["image_size_wh"])
    pattern = (args.pattern_cols, args.pattern_rows)
    detections, _, detection_report = _detect_all(images, pattern)

    if args.square_size_mm is None:
        square_size = 1.0
        length_unit = "checker_square"
        metric_scale_verified = False
    else:
        if args.square_size_mm <= 0:
            raise ValueError("--square-size-mm must be positive")
        square_size = args.square_size_mm / 1000.0
        length_unit = "meter"
        metric_scale_verified = True
    object_template = _object_points(pattern, square_size)

    mono: dict[int, dict] = {}
    mono_rejected: dict[int, list[dict]] = {}
    free_diagnostic: dict[int, dict] = {}
    for camera in (1, 2):
        mono[camera], mono_rejected[camera] = _calibrate_camera_robust(
            detections[camera],
            object_template,
            image_size,
            fix_aspect_ratio=True,
            error_floor_px=args.mono_outlier_floor_px,
            min_views=args.min_mono_views,
        )
        free_diagnostic[camera] = _calibrate_once(
            mono[camera]["ids"],
            detections[camera],
            object_template,
            image_size,
            False,
        )

    stereo, stereo_rejected = _calibrate_stereo_robust(
        detections[1],
        detections[2],
        object_template,
        image_size,
        mono[1],
        mono[2],
        error_floor_px=args.stereo_outlier_floor_px,
        min_pairs=args.min_stereo_pairs,
    )

    pnp_consistency = _pnp_pair_consistency(stereo, detections[1], object_template)
    transform_camera2_from_camera1 = _matrix4(
        stereo["rotation"], stereo["translation"]
    )
    transform_camera1_from_camera2 = _invert_transform(
        transform_camera2_from_camera1
    )

    reference_id = args.reference_frame
    if reference_id is None or reference_id not in stereo["ids"]:
        reference_id = stereo["ids"][0]
    ok, rvec, tvec = cv2.solvePnP(
        object_template,
        detections[1][reference_id],
        stereo["camera_matrix1"],
        stereo["distortion1"],
    )
    if not ok:
        raise RuntimeError(f"solvePnP failed for reference frame {reference_id}")
    transform_camera1_from_board = _matrix4(cv2.Rodrigues(rvec)[0], tvec)
    transform_camera2_from_board = (
        transform_camera2_from_camera1 @ transform_camera1_from_board
    )

    qa = _make_qa(
        Path(args.qa_dir).resolve(),
        images,
        detections,
        pattern,
        stereo,
        object_template,
        image_size,
    )
    epipolar_values = list(stereo["pair_epipolar_rms_px"].values())
    rotation_deviations = [
        item["rotation_deviation_deg"] for item in pnp_consistency.values()
    ]
    translation_deviations = [
        item["translation_deviation"] for item in pnp_consistency.values()
    ]
    transfer_stats = qa["stereo_transfer_error_stats_px"]
    quality_checks = {
        "camera_1_mono_rms_le_0_8px": mono[1]["rms"] <= 0.8,
        "camera_2_mono_rms_le_0_8px": mono[2]["rms"] <= 0.8,
        "stereo_rms_le_1_0px": stereo["rms"] <= 1.0,
        "epipolar_p95_le_1_0px": float(np.percentile(epipolar_values, 95)) <= 1.0,
        # This deliberately uses a looser gate than the joint stereo RMS: it
        # estimates the board pose from camera 1 alone, where the target covers
        # as little as 0.5% of the wide image, then transfers it across a very
        # wide baseline.  It is an independent stability check, not the fitted
        # stereo residual.
        "single_view_transfer_p95_le_5_0px": transfer_stats["p95"] <= 5.0,
        "at_least_12_stereo_pairs": len(stereo["ids"]) >= 12,
        "metric_scale_verified": metric_scale_verified,
        "robot_or_table_extrinsic_verified": False,
    }
    geometry_checks = [
        value
        for name, value in quality_checks.items()
        if name not in {"metric_scale_verified", "robot_or_table_extrinsic_verified"}
    ]

    if all(geometry_checks):
        quality_status = (
            "relative_geometry_pass"
            if metric_scale_verified
            else "relative_geometry_pass_scale_pending"
        )
    else:
        quality_status = "review"

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_meta,
        "checkerboard": {
            "inner_corners_cols_rows": list(pattern),
            "outer_squares_cols_rows": [pattern[0] + 1, pattern[1] + 1],
            "square_size_mm": args.square_size_mm,
            "length_unit": length_unit,
            "metric_scale_verified": metric_scale_verified,
        },
        "coordinate_convention": {
            "camera": "OpenCV: +x right, +y down, +z forward",
            "relative_transform": (
                "X_camera2 = R_camera2_from_camera1 * X_camera1 "
                "+ t_camera2_from_camera1"
            ),
            "reference_board": (
                "origin is the first detected inner corner in camera_1; "
                "+x follows checkerboard columns, +y rows, +z by right-hand rule"
            ),
        },
        "detection": detection_report,
        "camera_1": _camera_report(
            mono[1], mono_rejected[1], free_diagnostic[1]
        ),
        "camera_2": _camera_report(
            mono[2], mono_rejected[2], free_diagnostic[2]
        ),
        "stereo": {
            "rms_reprojection_px": stereo["rms"],
            "used_pair_ids": stereo["ids"],
            "reversed_180_pair_ids": sorted(stereo["reversed_ids"]),
            "rejected_pairs": stereo_rejected,
            "outlier_threshold_px": stereo["outlier_threshold_px"],
            "pair_epipolar_rms_px": {
                str(frame): error
                for frame, error in stereo["pair_epipolar_rms_px"].items()
            },
            "epipolar_error_stats_px": _stats(epipolar_values),
            "R_camera2_from_camera1": stereo["rotation"].tolist(),
            "t_camera2_from_camera1": stereo["translation"].tolist(),
            "translation_unit": length_unit,
            "baseline": float(np.linalg.norm(stereo["translation"])),
            "relative_rotation_angle_deg": _rotation_angle_degrees(
                stereo["rotation"]
            ),
            "camera2_origin_in_camera1": (
                -stereo["rotation"].T @ stereo["translation"]
            ).tolist(),
            "camera1_origin_in_camera2": stereo["translation"].tolist(),
            "T_camera2_from_camera1": transform_camera2_from_camera1.tolist(),
            "T_camera1_from_camera2": transform_camera1_from_camera2.tolist(),
            "essential_matrix": stereo["essential"].tolist(),
            "fundamental_matrix": stereo["fundamental"].tolist(),
            "pair_pose_consistency": {
                str(frame): value for frame, value in pnp_consistency.items()
            },
            "rotation_deviation_stats_deg": _stats(rotation_deviations),
            "translation_deviation_stats": _stats(translation_deviations),
        },
        "reference_board_pose": {
            "frame_id": reference_id,
            "T_camera1_from_board": transform_camera1_from_board.tolist(),
            "T_board_from_camera1": _invert_transform(
                transform_camera1_from_board
            ).tolist(),
            "T_camera2_from_board": transform_camera2_from_board.tolist(),
            "T_board_from_camera2": _invert_transform(
                transform_camera2_from_board
            ).tolist(),
            "translation_unit": length_unit,
            "is_robot_or_table_frame": False,
        },
        "rectification": qa["rectification"],
        "qa": {
            key: value for key, value in qa.items() if key != "rectification"
        },
        "quality": {
            "status": quality_status,
            "checks": quality_checks,
            "limitations": [
                (
                    "The ZIP has no physical checker-square length; metric "
                    "translation is unavailable."
                    if not metric_scale_verified
                    else "Metric scale comes from the CLI square-size measurement."
                ),
                (
                    "Same-index images were saved sequentially rather than by "
                    "hardware sync; rejected pairs may indicate board motion."
                ),
                (
                    "No checkerboard-to-robot-base or checkerboard-to-table "
                    "transform is provided, so absolute Isaac placement is unverified."
                ),
                (
                    "Only RGB images are present; this does not calibrate RGB-depth "
                    "registration or depth intrinsics."
                ),
                (
                    "The cameras have a very large convergent angle, so OpenCV "
                    "full-frame horizontal rectification has no common valid ROI. "
                    "Use R/T-based 3-D reprojection rather than block stereo."
                ),
            ],
        },
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Calibration ZIP or directory")
    parser.add_argument("--out", required=True, help="Output calibration JSON")
    parser.add_argument("--qa-dir", required=True, help="Directory for QA previews")
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=None,
        help="Measured checker square edge. Omit to use square units.",
    )
    parser.add_argument("--reference-frame", type=int, default=1)
    parser.add_argument("--min-mono-views", type=int, default=12)
    parser.add_argument("--min-stereo-pairs", type=int, default=12)
    parser.add_argument("--mono-outlier-floor-px", type=float, default=0.75)
    parser.add_argument("--stereo-outlier-floor-px", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = calibrate(args)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"calibration: {output}")
    print(f"quality: {report['quality']['status']}")
    print(
        "mono RMS: "
        f"{report['camera_1']['rms_reprojection_px']:.4f}, "
        f"{report['camera_2']['rms_reprojection_px']:.4f} px"
    )
    print(f"stereo RMS: {report['stereo']['rms_reprojection_px']:.4f} px")
    print(
        "baseline: "
        f"{report['stereo']['baseline']:.6f} "
        f"{report['stereo']['translation_unit']}"
    )


if __name__ == "__main__":
    main()
