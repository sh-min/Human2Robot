"""Export the frames in which a static scene object stays behind the robot.

`refine_interaction_object_masks.py` restores a static table object such as the
sponge on *every* frame, because the human inpainting mask erases it whenever an
arm sweeps over it.  The compositor then draws those pixels in its object layers,
which sit in front of the rear robot, so the hand appears to be cut by an object
it is only passing above.

An object is a real interaction object only inside its annotated segment.  Every
other frame of its restored mask belongs behind the robot, and this script writes
exactly that mask for ``--behind_robot_object_mask``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_mask", type=Path, required=True,
                        help="Per-frame mask of the restored static object.")
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--object_name", required=True,
                        help="Segment name whose interaction interval is kept "
                             "out of the behind-robot mask.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    segments = {item["name"]: item for item in
                json.loads(args.segments_json.read_text(encoding="utf-8"))["segments"]}
    if args.object_name not in segments:
        raise SystemExit(f"unknown segment: {args.object_name}")
    segment = segments[args.object_name]
    start, end = int(segment["start_frame"]), int(segment["end_frame"])

    source = np.load(args.object_mask, mmap_mode="r")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=bool, shape=source.shape
    )
    kept = 0
    for frame_idx in range(len(source)):
        if start <= frame_idx <= end:
            result[frame_idx] = False
            continue
        frame = np.asarray(source[frame_idx], dtype=bool)
        result[frame_idx] = frame
        kept += int(frame.sum())
    result.flush()
    print(f"[info] {args.object_name} interaction interval {start}-{end} excluded")
    print(f"[info] behind-robot px={kept}, frames={len(source)}")
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
