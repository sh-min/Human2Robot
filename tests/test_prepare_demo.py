"""Regression tests for the demo-layout preparation step."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
INPAINTING_DIR = REPO_ROOT / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

import prepare_demo  # noqa: E402


class PrepareDemoCopyTests(unittest.TestCase):
    def test_existing_processed_dir_without_video_receives_raw_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "raw"
            processed_root = root / "processed"
            raw_video = data_root / "cam0" / "0" / "video_L.mp4"
            raw_video.parent.mkdir(parents=True)
            raw_video.write_bytes(b"already-built-video")

            processed_dir = processed_root / "cam0" / "0"
            processed_dir.mkdir(parents=True)
            marker = processed_dir / "existing-stage-output.txt"
            marker.write_text("keep me")
            processed_video = processed_dir / "video_L.mp4"
            self.assertFalse(processed_video.exists())

            argv = [
                "prepare_demo.py",
                "--input",
                str(root / "input-does-not-need-to-exist"),
                "--data_root",
                str(data_root),
                "--processed_root",
                str(processed_root),
                "--demo_name",
                "cam0",
                "--demo_num",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(
                io.StringIO()
            ):
                prepare_demo.main()

            self.assertEqual(
                processed_video.read_bytes(),
                raw_video.read_bytes(),
            )
            self.assertEqual(marker.read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
