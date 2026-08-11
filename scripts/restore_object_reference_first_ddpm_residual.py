#!/usr/bin/env python3
"""Restore hidden object RGB from real reference pixels before DDPM fallback.

Reference frames are selected per SAM2 track.  Their trusted observed pixels
are aligned to each target with ORB/RANSAC affine registration (PCA silhouette
alignment is the fallback).  Per-pixel selection favours donors farther from a
reference-mask boundary.  DDPM RGB is accepted only where no real reference
pixel reaches an inferred hidden object pixel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer: {path}")
    return writer


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def mask_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    def stats(mask: np.ndarray):
        ys, xs = np.nonzero(mask)
        points = np.stack((xs, ys), axis=1).astype(np.float64)
        center = points.mean(axis=0)
        covariance = np.cov((points - center).T) + np.eye(2) * 1e-3
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        return center, values[order], vectors[:, order]

    source_center, source_values, source_vectors = stats(source)
    target_center, target_values, target_vectors = stats(target)
    # Resolve the PCA sign ambiguity using a proper rotation.
    if np.linalg.det(target_vectors @ source_vectors.T) < 0:
        target_vectors[:, 1] *= -1
    scales = np.sqrt(target_values / np.maximum(source_values, 1e-3))
    scales = np.clip(scales, 0.55, 1.8)
    linear = target_vectors @ np.diag(scales) @ source_vectors.T
    translation = target_center - linear @ source_center
    return np.concatenate((linear, translation[:, None]), axis=1).astype(np.float32)


def orb_affine(
    source_gray: np.ndarray,
    source_mask: np.ndarray,
    target_gray: np.ndarray,
    target_mask: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, int]:
    orb = cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8)
    source_kp, source_desc = orb.detectAndCompute(
        source_gray, source_mask.astype(np.uint8) * 255
    )
    target_kp, target_desc = orb.detectAndCompute(
        target_gray, target_mask.astype(np.uint8) * 255
    )
    if source_desc is None or target_desc is None:
        return fallback, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_desc, target_desc, k=2)
    good = [first for first, second in pairs if first.distance < 0.78 * second.distance]
    if len(good) < 10:
        return fallback, len(good)
    source_points = np.float32([source_kp[item.queryIdx].pt for item in good])
    target_points = np.float32([target_kp[item.trainIdx].pt for item in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=2500,
        confidence=0.995,
    )
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if affine is None or inlier_count < 8:
        return fallback, inlier_count
    return affine.astype(np.float32), inlier_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_video", type=Path, required=True)
    parser.add_argument("--base_completion", type=Path, required=True)
    parser.add_argument("--ddpm_candidate", type=Path, required=True)
    parser.add_argument("--observed_mask", type=Path, required=True)
    parser.add_argument("--amodal_mask", type=Path, required=True)
    parser.add_argument("--sam2_report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--minimum_alignment_iou", type=float, default=0.28)
    parser.add_argument("--mask_tolerance_px", type=int, default=9)
    parser.add_argument("--reference_boundary_px", type=float, default=5.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_cap = cv2.VideoCapture(str(args.base_completion))
    ddpm_cap = cv2.VideoCapture(str(args.ddpm_candidate))
    raw_cap = cv2.VideoCapture(str(args.raw_video))
    if not base_cap.isOpened() or not ddpm_cap.isOpened() or not raw_cap.isOpened():
        raise FileNotFoundError("could not open input videos")
    width = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(round(base_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(base_cap.get(cv2.CAP_PROP_FPS) or 30.0)
    observed_all = np.load(args.observed_mask, mmap_mode="r")
    amodal_all = np.load(args.amodal_mask, mmap_mode="r")
    if len(observed_all) != frames or len(amodal_all) != frames:
        raise ValueError("mask/video frame count mismatch")
    report = json.loads(args.sam2_report.read_text())
    episodes = report["episodes"]
    frame_episode = np.full(frames, -1, dtype=np.int16)
    references_by_episode: list[list[int]] = []
    for episode_index, episode in enumerate(episodes):
        start, end = int(episode["start"]), int(episode["end"])
        frame_episode[start:end + 1] = episode_index
        references = list(dict.fromkeys(
            [int(episode["clean_reference_frame"]), int(episode["seed_frame"])]
            + [start, end, (start + end) // 2]
        ))
        references_by_episode.append(references)

    reference_indices = sorted(set(sum(references_by_episode, [])))
    raw_references: dict[int, np.ndarray] = {}
    raw_index = 0
    while True:
        ok, frame = raw_cap.read()
        if not ok:
            break
        if raw_index in reference_indices:
            raw_references[raw_index] = frame.copy()
        raw_index += 1
    raw_cap.release()
    if raw_index != frames or len(raw_references) != len(reference_indices):
        raise RuntimeError("failed to load reference frames")

    reference_data = {}
    for index, image in raw_references.items():
        mask = resize_mask(observed_all[index], width, height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        reference_data[index] = (image, gray, mask, distance)

    output_writer = open_writer(
        args.output_dir / "video_object_reference_first_ddpm_residual.mp4v.mp4",
        fps,
        (width, height),
    )
    debug_writer = open_writer(
        args.output_dir / "video_reference_ddpm_evidence.mp4v.mp4",
        fps,
        (width, height + 64),
    )
    reference_counts = np.zeros(frames, dtype=np.int64)
    ddpm_counts = np.zeros(frames, dtype=np.int64)
    unresolved_counts = np.zeros(frames, dtype=np.int64)
    orb_inlier_total = 0
    accepted_alignment_total = 0
    tolerance_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.mask_tolerance_px + 1,) * 2
    )

    try:
        for frame_index in range(frames):
            ok_base, base = base_cap.read()
            ok_ddpm, ddpm = ddpm_cap.read()
            if not ok_base or not ok_ddpm:
                raise RuntimeError(f"input video ended at frame {frame_index}")
            observed = resize_mask(observed_all[frame_index], width, height)
            amodal = resize_mask(amodal_all[frame_index], width, height)
            hidden = amodal & ~observed
            target_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            best_score = np.zeros((height, width), dtype=np.float32)
            best_rgb = np.zeros_like(base)
            episode_index = int(frame_episode[frame_index])
            if episode_index >= 0 and int(observed.sum()) >= 100:
                target_tolerance = cv2.dilate(
                    observed.astype(np.uint8), tolerance_kernel
                ).astype(bool)
                for reference_index in references_by_episode[episode_index]:
                    reference_image, reference_gray, reference_mask, reference_distance = (
                        reference_data[reference_index]
                    )
                    if int(reference_mask.sum()) < 100:
                        continue
                    fallback = mask_affine(reference_mask, observed)
                    affine, inliers = orb_affine(
                        reference_gray,
                        reference_mask,
                        target_gray,
                        observed,
                        fallback,
                    )
                    orb_inlier_total += inliers
                    warped_mask = cv2.warpAffine(
                        reference_mask.astype(np.uint8), affine, (width, height),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                    ).astype(bool)
                    intersection = int(np.sum(warped_mask & target_tolerance))
                    union = int(np.sum(warped_mask | observed))
                    alignment_iou = intersection / max(union, 1)
                    if alignment_iou < args.minimum_alignment_iou:
                        continue
                    accepted_alignment_total += 1
                    warped_rgb = cv2.warpAffine(
                        reference_image, affine, (width, height),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    )
                    warped_distance = cv2.warpAffine(
                        reference_distance, affine, (width, height),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    )
                    donor = hidden & warped_mask
                    score = alignment_iou * np.clip(
                        warped_distance / max(args.reference_boundary_px, 1.0),
                        0.0,
                        1.0,
                    )
                    replace = donor & (score > best_score)
                    best_rgb[replace] = warped_rgb[replace]
                    best_score[replace] = score[replace]

            reference_fill = hidden & (best_score > 0)
            ddpm_fill = hidden & ~reference_fill
            output = base.copy()
            output[reference_fill] = best_rgb[reference_fill]
            output[ddpm_fill] = ddpm[ddpm_fill]
            output[observed] = base[observed]
            reference_counts[frame_index] = int(reference_fill.sum())
            ddpm_counts[frame_index] = int(ddpm_fill.sum())
            unresolved_counts[frame_index] = int(
                np.sum(hidden & ~reference_fill & ~ddpm_fill)
            )
            output_writer.write(output)

            debug = cv2.copyMakeBorder(
                output, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
            )
            tint = np.zeros_like(output)
            tint[reference_fill] = (0, 200, 0)
            tint[ddpm_fill] = (0, 0, 220)
            debug[64:] = cv2.addWeighted(debug[64:], 1.0, tint, 0.48, 0)
            cv2.putText(
                debug,
                f"frame {frame_index:04d} | green=real reference | red=DDPM residual",
                (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (240, 240, 240), 2, cv2.LINE_AA,
            )
            debug_writer.write(debug)
            if (frame_index + 1) % 50 == 0:
                print(f"[reference-first] {frame_index + 1}/{frames}", flush=True)
    finally:
        base_cap.release()
        ddpm_cap.release()
        output_writer.release()
        debug_writer.release()

    hidden_total = int(reference_counts.sum() + ddpm_counts.sum())
    output_report = {
        "schema_version": 1,
        "method": "real_reference_warp_first_ddpm_only_for_residual_holes",
        "frames": frames,
        "reference_frames_by_episode": references_by_episode,
        "real_reference_pixels": int(reference_counts.sum()),
        "ddpm_residual_pixels": int(ddpm_counts.sum()),
        "real_reference_fraction_of_hidden": (
            float(reference_counts.sum()) / max(hidden_total, 1)
        ),
        "ddpm_fraction_of_hidden": float(ddpm_counts.sum()) / max(hidden_total, 1),
        "unresolved_pixels": int(unresolved_counts.sum()),
        "accepted_reference_alignments": accepted_alignment_total,
        "orb_inliers_total": orb_inlier_total,
        "minimum_alignment_iou": args.minimum_alignment_iou,
        "invariants": {
            "reference_donors_are_trusted_observed_object_pixels": True,
            "ddpm_used_only_where_no_reference_donor_exists": True,
            "observed_target_object_rgb_unchanged": True,
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(output_report, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output_dir / "fill_counts.npz",
        real_reference_pixels=reference_counts,
        ddpm_residual_pixels=ddpm_counts,
    )
    print(json.dumps(output_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
