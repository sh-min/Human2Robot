"""Union a conservative 2D hand silhouette into a SAM2 arm mask.

Object-negative SAM2 prompts can occasionally trim fingertips that touch a
held object.  The 21 hand keypoints remain reliable in those frames, so this
post-process fills the hand's convex hull, adds short capsules along every hand
bone, dilates the result slightly, and unions it with the existing arm mask.
The manipulated-object mask can still be passed as ``--protect_mask`` to the
inpainting stage, so object pixels inside this conservative hull are retained.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--hand_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=16)
    parser.add_argument("--bone_radius", type=int, default=8)
    args = parser.parse_args()

    masks = np.load(args.mask, mmap_mode="r")
    hand = np.load(args.hand_data)
    points = hand["kpts_2d"].astype(np.float32)
    detected = hand["hand_detected"].astype(bool)
    count = min(len(masks), len(points), len(detected))
    output = np.asarray(masks, dtype=bool).copy()
    kernel_size = max(1, 2 * args.dilate + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    added = np.zeros(count, dtype=np.int64)

    for frame_idx in range(count):
        if not detected[frame_idx]:
            continue
        pts = np.rint(points[frame_idx]).astype(np.int32)
        hand_mask = np.zeros(output.shape[1:], dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(hand_mask, hull, 1, lineType=cv2.LINE_AA)
        for start, end in CONNECTIONS:
            cv2.line(hand_mask, tuple(pts[start]), tuple(pts[end]), 1,
                     thickness=max(1, 2 * args.bone_radius), lineType=cv2.LINE_AA)
            cv2.circle(hand_mask, tuple(pts[end]), args.bone_radius, 1, -1,
                       cv2.LINE_AA)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1).astype(bool)
        before = int(output[frame_idx].sum())
        output[frame_idx] |= hand_mask
        added[frame_idx] = int(output[frame_idx].sum()) - before

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"[info] frames={count}, mean added={added.mean():.0f}px, "
          f"max added={added.max(initial=0)}px")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
