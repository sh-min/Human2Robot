"""Complete a hand-occluded sponge from a clean pre-contact reference frame.

The source frame only contains the sponge pixels that are not covered by the
human hand.  This utility tracks the planar sponge with SIFT + a robust affine
fit, warps a clean reference texture underneath the hand, and then restores the
currently visible source pixels on top.  The resulting video/mask can be used
as the object layer in the robot compositor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} from {video}")
    return frame


def _solid_mask(visible: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(visible))
    if not len(points):
        raise RuntimeError("reference sponge mask is empty")
    xy = points[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(xy)
    result = np.zeros_like(visible, dtype=np.uint8)
    cv2.fillConvexPoly(result, hull, 1)
    result = cv2.morphologyEx(
        result, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return result.astype(bool)


def _estimate_affine(
    matcher: cv2.BFMatcher,
    reference_points: list[cv2.KeyPoint],
    reference_descriptors: np.ndarray,
    current_points: list[cv2.KeyPoint],
    current_descriptors: np.ndarray | None,
    reference_center: np.ndarray,
    current_mask: np.ndarray,
    ratio: float,
    min_inliers: int,
) -> tuple[np.ndarray | None, int, int]:
    if current_descriptors is None or len(current_points) < 3:
        return None, 0, 0
    pairs = matcher.knnMatch(reference_descriptors, current_descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2
            and pair[0].distance < ratio * pair[1].distance]
    if len(good) < 3:
        return None, len(good), 0
    source = np.float32(
        [reference_points[item.queryIdx].pt for item in good]
    )
    target = np.float32([current_points[item.trainIdx].pt for item in good])
    affine, status = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.5,
        maxIters=3000, confidence=0.995,
    )
    inliers = int(status.sum()) if status is not None else 0
    if affine is None or inliers < min_inliers:
        return None, len(good), inliers
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    if not 0.55 <= scale <= 1.65:
        return None, len(good), inliers
    current_yx = np.column_stack(np.nonzero(current_mask))
    if len(current_yx):
        current_center = np.median(current_yx[:, ::-1], axis=0)
        projected = affine[:, :2] @ reference_center + affine[:, 2]
        if float(np.linalg.norm(projected - current_center)) > 115.0:
            return None, len(good), inliers
    return affine.astype(np.float32), len(good), inliers


def _smooth_affine(previous: np.ndarray, current: np.ndarray,
                   amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    return ((1.0 - amount) * previous + amount * current).astype(np.float32)


def _compose_affine(delta: np.ndarray, transform: np.ndarray) -> np.ndarray:
    delta_h = np.vstack([delta, [0.0, 0.0, 1.0]])
    transform_h = np.vstack([transform, [0.0, 0.0, 1.0]])
    return (delta_h @ transform_h)[:2].astype(np.float32)


def _estimate_flow_affine(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    if int(previous_mask.sum()) < 80 or int(current_mask.sum()) < 80:
        return None, 0
    points = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=240, qualityLevel=0.008, minDistance=3,
        blockSize=5, mask=previous_mask.astype(np.uint8) * 255,
    )
    if points is None or len(points) < 3:
        return None, 0
    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None, winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None:
        return None, 0
    source = points.reshape(-1, 2)
    target = tracked.reshape(-1, 2)
    keep = status.reshape(-1).astype(bool)
    keep &= error.reshape(-1) < 30.0
    support = cv2.dilate(
        current_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    ).astype(bool)
    x = np.clip(np.rint(target[:, 0]).astype(int), 0, support.shape[1] - 1)
    y = np.clip(np.rint(target[:, 1]).astype(int), 0, support.shape[0] - 1)
    keep &= support[y, x]
    source = source[keep]
    target = target[keep]
    if len(source) < 3:
        return None, len(source)
    delta, inlier_mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.5,
        maxIters=2000, confidence=0.995,
    )
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if delta is None or inliers < 3:
        return None, inliers
    scale = float(np.hypot(delta[0, 0], delta[1, 0]))
    angle = abs(float(np.degrees(np.arctan2(delta[1, 0], delta[0, 0]))))
    shift = float(np.linalg.norm(delta[:, 2]))
    if not 0.82 <= scale <= 1.18 or angle > 18.0 or shift > 35.0:
        return None, inliers
    return delta.astype(np.float32), inliers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--visible_mask", type=Path, required=True)
    parser.add_argument("--object_mask", type=Path, required=True)
    parser.add_argument("--reference_frame", type=int, default=315)
    parser.add_argument("--completion_start", type=int, default=320)
    parser.add_argument("--completion_end", type=int, default=520)
    parser.add_argument("--ratio", type=float, default=0.78)
    parser.add_argument("--min_inliers", type=int, default=4)
    parser.add_argument("--smoothing", type=float, default=0.45)
    parser.add_argument("--output_video", type=Path, required=True)
    parser.add_argument("--output_mask", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    visible = np.load(args.visible_mask, mmap_mode="r")
    base_object = np.load(args.object_mask, mmap_mode="r")
    reference = _read_frame(args.video, args.reference_frame)
    height, width = reference.shape[:2]
    reference_mask = _solid_mask(np.asarray(
        visible[args.reference_frame], dtype=bool
    ))
    reference_yx = np.column_stack(np.nonzero(reference_mask))
    reference_center = np.median(reference_yx[:, ::-1], axis=0).astype(np.float32)

    sift = cv2.SIFT_create(nfeatures=1400, contrastThreshold=0.015)
    reference_points, reference_descriptors = sift.detectAndCompute(
        reference, reference_mask.astype(np.uint8) * 255
    )
    if reference_descriptors is None or len(reference_points) < 4:
        raise RuntimeError("not enough SIFT features in reference sponge")
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    frame_count = min(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(visible), len(base_object)
    )
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"FFV1"), fps,
        (width, height),
    )
    preview_writer = None
    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview_writer = cv2.VideoWriter(
            str(args.preview), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )

    completed_mask = np.asarray(base_object, dtype=bool).copy()
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        dtype=np.float32)
    transform = identity.copy()
    reliable = 0
    flow_reliable = 0
    carried = 0
    match_total = 0
    inlier_total = 0
    flow_inlier_total = 0
    previous_gray = None
    previous_visible = None

    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"video decode stopped at frame {frame_index}")
        current_visible = np.asarray(visible[frame_index], dtype=bool)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        use_completion = args.completion_start <= frame_index <= args.completion_end
        if use_completion:
            flow_estimate = None
            flow_inliers = 0
            if previous_gray is not None and previous_visible is not None:
                flow_estimate, flow_inliers = _estimate_flow_affine(
                    previous_gray, current_gray, previous_visible,
                    current_visible,
                )
            flow_inlier_total += flow_inliers
            if flow_estimate is not None:
                transform = _compose_affine(flow_estimate, transform)
                flow_reliable += 1
            points, descriptors = sift.detectAndCompute(
                frame, current_visible.astype(np.uint8) * 255
            )
            estimate, matches, inliers = _estimate_affine(
                matcher, reference_points, reference_descriptors,
                points, descriptors, reference_center, current_visible,
                args.ratio, args.min_inliers,
            )
            match_total += matches
            inlier_total += inliers
            if estimate is not None:
                transform = _smooth_affine(transform, estimate, args.smoothing)
                reliable += 1
            elif flow_estimate is None:
                carried += 1

            warped_reference = cv2.warpAffine(
                reference, transform, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            warped_mask = cv2.warpAffine(
                reference_mask.astype(np.uint8), transform, (width, height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            ).astype(bool)
            completed = frame.copy()
            completed[warped_mask] = warped_reference[warped_mask]
            # Keep all genuinely visible pixels from the current frame so only
            # the hand-occluded interior comes from the reference texture.
            completed[current_visible] = frame[current_visible]
            completed_mask[frame_index] |= warped_mask | current_visible
        else:
            completed = frame
            warped_mask = np.zeros((height, width), dtype=bool)
            completed_mask[frame_index] |= current_visible

        writer.write(completed)
        if preview_writer is not None:
            preview_frame = completed.copy()
            outline = cv2.morphologyEx(
                warped_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ).astype(bool)
            preview_frame[outline] = (20, 20, 245)
            preview_writer.write(preview_frame)
        if (frame_index + 1) % 100 == 0:
            print(f"[frame] {frame_index + 1}/{frame_count}", flush=True)
        previous_gray = current_gray
        previous_visible = current_visible

    cap.release()
    writer.release()
    if preview_writer is not None:
        preview_writer.release()
    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_mask, completed_mask)
    print(
        f"[info] reference features={len(reference_points)}, "
        f"reference transforms={reliable}, flow transforms={flow_reliable}, "
        f"carried={carried}, matches={match_total}, inliers={inlier_total}, "
        f"flow inliers={flow_inlier_total}"
    )
    print(f"[ok] wrote {args.output_video}")
    print(f"[ok] wrote {args.output_mask}")


if __name__ == "__main__":
    main()
