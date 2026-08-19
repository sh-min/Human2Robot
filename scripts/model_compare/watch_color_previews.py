#!/usr/bin/env python3
"""Open each newly completed color robot-hands video in VLC."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--idle-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    seen = set(args.root.glob("*/color*_robot_hands.mp4"))
    sizes: dict[Path, int] = {}
    last_activity = time.monotonic()
    while time.monotonic() - last_activity < args.idle_timeout:
        for video in sorted(args.root.glob("*/color*_robot_hands.mp4")):
            if video in seen:
                continue
            size = video.stat().st_size
            if size <= 0 or sizes.get(video) != size:
                sizes[video] = size
                last_activity = time.monotonic()
                continue
            subprocess.Popen(
                [
                    "vlc",
                    "--no-one-instance",
                    "--play-and-exit",
                    "--no-video-title-show",
                    str(video.resolve()),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            seen.add(video)
            sizes.pop(video, None)
            last_activity = time.monotonic()
            print(f"[preview] {video}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
