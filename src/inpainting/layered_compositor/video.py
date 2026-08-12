"""Video output abstraction with VS Code/browser-compatible H.264 delivery."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


class CompatibleVideoWriter:
    """Write frames through OpenCV and optionally publish an H.264 MP4.

    OpenCV builds commonly expose ``mp4v`` but not an H.264 encoder.  Chromium
    (and therefore VS Code's preview) commonly has the opposite compatibility
    profile.  This class uses the reliable OpenCV encoder as an intermediate,
    then atomically publishes an ffmpeg/libx264 file on ``release``.
    """

    def __init__(self, path: str | Path, fps: float,
                 size: tuple[int, int], codec: str = "h264") -> None:
        self.path = Path(path)
        self.codec = codec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if codec not in {"h264", "mp4v"}:
            raise ValueError(f"unsupported video codec: {codec}")
        self._raw_path = (
            self.path if codec == "mp4v"
            else self.path.with_name(f".{self.path.stem}.raw_mp4v.mp4")
        )
        self._encoded_path = self.path.with_name(
            f".{self.path.stem}.encoded_h264.mp4"
        )
        self._writer = cv2.VideoWriter(
            str(self._raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        self._released = False

    def isOpened(self) -> bool:  # OpenCV-compatible spelling
        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        if self._released:
            return
        self._writer.release()
        self._released = True
        if self.codec == "mp4v" or not self._raw_path.exists():
            return
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required for --video_codec h264; use mp4v only "
                "when browser/VS Code preview compatibility is not required"
            )
        command = [
            ffmpeg, "-nostdin", "-y", "-v", "error",
            "-i", str(self._raw_path),
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self._encoded_path),
        ]
        try:
            subprocess.run(command, check=True)
            os.replace(self._encoded_path, self.path)
        finally:
            self._raw_path.unlink(missing_ok=True)
            self._encoded_path.unlink(missing_ok=True)
