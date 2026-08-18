"""Which grasps need a clean-frame paste, decided from the masks themselves.

``paste_object_from_reference`` fixes objects whose completion failed, but which
ones those are was being read off a comparison by eye and passed in by hand --
per clip, which is the manual step this pipeline keeps trying to remove.

The signal is in the data. A rigid object sits on the table fully visible before
the hand reaches it, so its area then is what it should still cover while held.
If the completed mask during the grasp is a fraction of that, the completion did
not rebuild it: either the hand covers most of it, or the visible remnant is a
rim and a handle whose convex hull spans the gap rather than the body. Those are
exactly the cases the reference paste exists for.

Prints the segment names to stdout, space separated, for the runner to pass on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_json", type=Path, required=True)
    parser.add_argument("--modal_mask", type=Path, required=True,
                        help="Visible-object mask, before completion.")
    parser.add_argument("--completed_mask", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.55,
                        help="Paste when the held area falls below this "
                             "fraction of the clean area.")
    parser.add_argument("--min_clean_px", type=int, default=1500,
                        help="Objects smaller than this are too small for the "
                             "warp to land accurately; leave them be.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    segments = json.loads(args.segments_json.read_text(encoding="utf-8"))["segments"]
    modal = np.load(args.modal_mask, mmap_mode="r")
    completed = np.load(args.completed_mask, mmap_mode="r")

    chosen = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = min(int(segment["end_frame"]), len(modal) - 1)
        contact = segment.get("contact_frames") or [start, end]
        grasp_start = int(contact[0])

        # Clean: tracked but not yet grasped. Held: the grasp itself.
        clean = [int(np.asarray(modal[t]).sum())
                 for t in range(start, min(grasp_start, end + 1))]
        held = [int(np.asarray(completed[t]).sum())
                for t in range(grasp_start, end + 1)]
        if not clean or not held:
            continue
        clean_area = float(np.percentile(clean, 90))
        held_area = float(np.median(held))
        if clean_area < args.min_clean_px:
            continue
        ratio = held_area / max(1.0, clean_area)
        if ratio < args.ratio:
            chosen.append(segment["name"])
        if args.verbose:
            mark = "PASTE" if ratio < args.ratio else "ok"
            print(f"[{mark}] {segment['name']}: clean {clean_area:.0f} px, "
                  f"held {held_area:.0f} px, ratio {ratio:.2f}",
                  flush=True)

    print(" ".join(chosen))


if __name__ == "__main__":
    main()
