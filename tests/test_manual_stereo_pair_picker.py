"""Tests for the manual stereo calibration frame picker."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "src" / "calibration"
sys.path.insert(0, str(CALIBRATION_DIR))

import manual_stereo_pair_picker as picker  # noqa: E402


def _write_video(path: Path, colours: list[tuple[int, int, int]]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 24),
    )
    if not writer.isOpened():
        raise RuntimeError("could not create synthetic test video")
    try:
        for colour in colours:
            frame = np.full((24, 32, 3), colour, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class ManualStereoPairPickerTests(unittest.TestCase):
    def test_pair_names_match_calibration_input_contract(self):
        self.assertEqual(picker.pair_filename(1), "1_Color.png")
        self.assertEqual(picker.pair_filename(38), "38_Color.png")
        for invalid in (0, -1, True, "bad"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                picker.pair_filename(invalid)

    def test_next_pair_id_scans_both_camera_folders_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "camera_1").mkdir()
            (root / "camera_2").mkdir()
            (root / "camera_1" / "1_Color.png").touch()
            (root / "camera_2" / "3_Color.png").touch()
            (root / "camera_2" / "ignore.png").touch()
            (root / "pairs.json").write_text(
                json.dumps({"schema_version": 1, "pairs": [{"pair_id": 7}]}),
                encoding="utf-8",
            )
            self.assertEqual(picker.next_pair_id(root), 8)

    def test_browser_proxy_preserves_source_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "camera.avi"
            _write_video(video, [(0, 0, 255), (0, 255, 0), (255, 0, 0), (80, 90, 100)])
            preview, source_info = picker.ensure_browser_preview(video, root / "cache")
            preview_info = picker.probe_video(preview)
            self.assertEqual(source_info["frames"], 4)
            self.assertEqual(preview_info["frames"], 4)
            self.assertTrue(preview.is_file())

    def test_exact_pair_extraction_uses_same_filename_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "camera1.avi"
            second = root / "camera2.avi"
            _write_video(first, [(0, 0, 255), (0, 255, 0), (255, 0, 0), (50, 60, 70)])
            _write_video(second, [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0)])
            output = root / "output"
            (output / "camera_1").mkdir(parents=True)
            (output / "camera_2").mkdir()
            info_1 = picker.probe_video(first)
            info_2 = picker.probe_video(second)
            config = picker.PickerConfig(
                camera_1=first,
                camera_2=second,
                output_root=output,
                preview_1=first,
                preview_2=second,
                info_1=info_1,
                info_2=info_2,
            )

            result = picker.save_frame_pair(
                config,
                frame_1=1,
                frame_2=3,
                pair_id=4,
            )
            saved_1 = output / "camera_1" / "4_Color.png"
            saved_2 = output / "camera_2" / "4_Color.png"
            self.assertTrue(saved_1.is_file())
            self.assertTrue(saved_2.is_file())
            image_1 = cv2.imread(str(saved_1))
            image_2 = cv2.imread(str(saved_2))
            self.assertEqual(int(np.argmax(image_1.mean(axis=(0, 1)))), 1)
            self.assertEqual(int(np.argmax(image_2.mean(axis=(0, 1)))), 0)
            manifest = json.loads((output / "pairs.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pairs"][0]["filename"], "4_Color.png")
            self.assertEqual(manifest["pairs"][0]["camera_1"]["frame"], 1)
            self.assertEqual(manifest["pairs"][0]["camera_2"]["frame"], 3)
            self.assertEqual(result["next_pair_id"], 5)

            with self.assertRaises(FileExistsError):
                picker.save_frame_pair(
                    config,
                    frame_1=0,
                    frame_2=0,
                    pair_id=4,
                )

    def test_frame_range_is_validated_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "camera.avi"
            _write_video(video, [(0, 0, 0)])
            output = root / "output"
            (output / "camera_1").mkdir(parents=True)
            (output / "camera_2").mkdir()
            info = picker.probe_video(video)
            config = picker.PickerConfig(
                camera_1=video,
                camera_2=video,
                output_root=output,
                preview_1=video,
                preview_2=video,
                info_1=info,
                info_2=info,
            )
            with self.assertRaisesRegex(ValueError, "camera 1 frame"):
                picker.save_frame_pair(
                    config,
                    frame_1=1,
                    frame_2=0,
                    pair_id=1,
                )


if __name__ == "__main__":
    unittest.main()
