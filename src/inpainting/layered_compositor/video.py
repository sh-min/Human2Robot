"""Video output abstraction with VS Code/browser-compatible H.264 delivery."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

# Visually lossless for this content while staying inside the H.264 High
# profile every browser decodes.  ``slow`` buys detail around the robot edges
# at no runtime cost that matters for a 553-frame clip.  Overridable so a
# stage whose output is itself re-encoded downstream can ask for CRF 0 and
# keep the chain to a single generation of loss.
H264_PRESET = os.environ.get("H264_PRESET", "slow")
H264_CRF = os.environ.get("H264_CRF", "14")


class CompatibleVideoWriter:
    """Write frames to a VS Code/browser-playable H.264 MP4.

    OpenCV builds commonly expose ``mp4v`` but not an H.264 encoder, while
    Chromium (and therefore VS Code's preview) commonly has the opposite
    compatibility profile.  For ``h264`` the frames are piped raw into ffmpeg,
    so the published file is a *first* encode of the composite rather than a
    re-encode of a lossy intermediate; the result is published atomically.
    ``mp4v`` still goes through OpenCV for callers that want no ffmpeg
    dependency.
    """

    def __init__(self, path: str | Path, fps: float,
                 size: tuple[int, int], codec: str = "h264") -> None:
        self.path = Path(path)
        self.codec = codec
        self.size = (int(size[0]), int(size[1]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if codec not in {"h264", "mp4v"}:
            raise ValueError(f"unsupported video codec: {codec}")
        self._released = False
        self._writer = None
        self._process = None
        self._encoded_path = self.path.with_name(
            f".{self.path.stem}.encoded_h264.mp4"
        )
        if codec == "mp4v":
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )
            return

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg is required for --video_codec h264; use mp4v only "
                "when browser/VS Code preview compatibility is not required"
            )
        self._process = subprocess.Popen(
            [
                ffmpeg, "-nostdin", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.size[0]}x{self.size[1]}",
                "-r", f"{fps:.6f}", "-i", "-",
                "-an", "-c:v", "libx264",
                "-preset", H264_PRESET, "-crf", H264_CRF,
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self._encoded_path),
            ],
            stdin=subprocess.PIPE,
        )

    def isOpened(self) -> bool:  # OpenCV-compatible spelling
        if self.codec == "mp4v":
            return self._writer.isOpened()
        return self._process is not None and self._process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if self.codec == "mp4v":
            self._writer.write(frame)
            return
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape[:2] != (self.size[1], self.size[0]):
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"writer expects {self.size[0]}x{self.size[1]}"
            )
        self._process.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.codec == "mp4v":
            self._writer.release()
            return
        self._process.stdin.close()
        code = self._process.wait()
        if code != 0:
            self._encoded_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed with exit code {code}")
        os.replace(self._encoded_path, self.path)
