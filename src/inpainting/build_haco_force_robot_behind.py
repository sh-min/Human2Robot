"""Force the complete non-thumb XHand behind objects on HaCo contact frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact_state", type=Path, required=True)
    parser.add_argument("--robot_mask", type=Path, required=True)
    parser.add_argument(
        "--robot_hand_mask",
        type=Path,
        required=True,
        help="Visible semantic XHand mask including palm and all fingers.",
    )
    parser.add_argument("--thumb_mask", type=Path, required=True)
    parser.add_argument("--human_mask", type=Path, default=None,
                        help="Visible source-human mask. Contact hand pixels "
                             "absent here are treated as object occlusion.")
    parser.add_argument(
        "--object_mask",
        type=Path,
        default=None,
        help=(
            "Trusted visible/completed object support. When supplied, the "
            "object-front output is strictly limited to this mask so source "
            "background or missed human pixels can never be drawn over robot."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output_object_front", type=Path, default=None,
                        help="Optional direct object-front mask for source-hand "
                             "pixels hidden at HaCo contact.")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--strong_object_front",
        action="store_true",
        help=(
            "On HaCo contact frames, keep the object in front throughout the "
            "non-thumb robot/contact overlap instead of requiring the source "
            "human mask to be absent. This is still constrained by object support."
        ),
    )
    parser.add_argument(
        "--object_support_dilate_px",
        type=int,
        default=0,
        help=(
            "Fixed, video-independent expansion of trusted object support used "
            "only by --strong_object_front (pixels)."
        ),
    )
    args = parser.parse_args()
    if args.object_support_dilate_px < 0:
        raise ValueError("--object_support_dilate_px must be non-negative")

    robot = np.load(args.robot_mask, mmap_mode="r")
    robot_hand = np.load(args.robot_hand_mask, mmap_mode="r")
    thumb = np.load(args.thumb_mask, mmap_mode="r")
    human = (np.load(args.human_mask, mmap_mode="r")
             if args.human_mask is not None else None)
    objects = (np.load(args.object_mask, mmap_mode="r")
               if args.object_mask is not None else None)
    if args.strong_object_front and objects is None:
        raise ValueError("--strong_object_front requires --object_mask")
    state = np.load(args.contact_state)["state"]
    side_index = 0 if args.side == "left" else 1
    frame_count = min(
        len(robot), len(robot_hand), len(thumb), state.shape[0]
    )
    if human is not None:
        frame_count = min(frame_count, len(human))
    if objects is not None:
        frame_count = min(frame_count, len(objects))
    height, width = robot.shape[1:]
    expected_shape = (frame_count, height, width)
    for label, array in (
        ("robot", robot),
        ("robot_hand", robot_hand),
        ("thumb", thumb),
    ):
        if array.shape[:3] != expected_shape:
            raise ValueError(
                f"{label} mask shape {array.shape} does not match {expected_shape}"
            )
    contact = state[:frame_count, side_index, 1:].any(axis=1)
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=bool,
        shape=(frame_count, height, width),
    )
    output[:] = False
    object_front = None
    if args.output_object_front is not None:
        object_front = np.lib.format.open_memmap(
            args.output_object_front, mode="w+", dtype=bool,
            shape=(frame_count, height, width),
        )
        object_front[:] = False
    for frame_index in np.flatnonzero(contact):
        forced_hand = (
            np.asarray(robot_hand[frame_index], dtype=bool)
            & np.asarray(robot[frame_index], dtype=bool)
            & ~np.asarray(thumb[frame_index], dtype=bool)
        )
        output[frame_index] = forced_hand
        if object_front is not None:
            if args.strong_object_front:
                candidate = forced_hand
            elif human is None:
                candidate = forced_hand
            else:
                candidate = (
                    forced_hand
                    & ~np.asarray(human[frame_index], dtype=bool)
                )
            if objects is not None:
                object_support = np.asarray(objects[frame_index], dtype=bool)
                if (args.strong_object_front
                        and args.object_support_dilate_px > 0):
                    radius = args.object_support_dilate_px
                    support_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * radius + 1, 2 * radius + 1),
                    )
                    object_support = cv2.dilate(
                        object_support.astype(np.uint8),
                        support_kernel,
                        iterations=1,
                    ).astype(bool)
                candidate &= object_support
            object_front[frame_index] = candidate
    output.flush()
    if object_front is not None:
        object_front.flush()
    active = output.sum(axis=(1, 2)) > 0
    print(f"[ok] {args.output} contact={int(contact.sum())}/{frame_count} "
          f"active={int(active.sum())} pixels={int(output.sum())}")
    if object_front is not None:
        print(f"[ok] {args.output_object_front} pixels={int(object_front.sum())}")


if __name__ == "__main__":
    main()
