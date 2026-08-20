import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "08_04" / "paired_annotation.py"
SPEC = importlib.util.spec_from_file_location("paired_annotation", MODULE_PATH)
paired_annotation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(paired_annotation)


class PairedAnnotationTests(unittest.TestCase):
    def test_discovers_all_numeric_pairs_in_natural_order(self):
        pairs = paired_annotation.discover_pairs()
        self.assertEqual(list(pairs), [str(index) for index in range(1, 25)])
        self.assertTrue(
            all(left.stem == right.stem == name for name, (left, right) in pairs.items())
        )

    def test_segment_validation_sorts_and_accepts_valid_ranges(self):
        segments = [
            {"start_frame": 10, "end_frame": 19, "label": "Sweep"},
            {"start_frame": 0, "end_frame": 9, "label": "Cup"},
        ]
        self.assertEqual(
            paired_annotation.validate_segments(segments, 20),
            list(reversed(segments)),
        )

    def test_can_pair_different_camera_filenames_by_natural_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "mh"
            right = root / "sh"
            left.mkdir()
            right.mkdir()
            for name in ("A_C057.mov", "A_C056.mov"):
                (left / name).touch()
            for name in ("Sh_C056.mov", "Sh_C055.mov"):
                (right / name).touch()
            original_left = paired_annotation.CAMERA_LEFT
            original_right = paired_annotation.CAMERA_RIGHT
            paired_annotation.CAMERA_LEFT = left
            paired_annotation.CAMERA_RIGHT = right
            try:
                pairs = paired_annotation.discover_pairs(pair_by_order=True)
            finally:
                paired_annotation.CAMERA_LEFT = original_left
                paired_annotation.CAMERA_RIGHT = original_right
            self.assertEqual(list(pairs), ["1", "2"])
            self.assertEqual(pairs["1"][0].name, "A_C056.mov")
            self.assertEqual(pairs["1"][1].name, "Sh_C055.mov")

    def test_segment_validation_rejects_invalid_ranges(self):
        invalid_cases = [
            [{"start_frame": 0, "end_frame": 20, "label": "Cup"}],
            [{"start_frame": 0, "end_frame": 2, "label": "Unknown"}],
            [
                {"start_frame": 0, "end_frame": 5, "label": "Cup"},
                {"start_frame": 5, "end_frame": 9, "label": "Sweep"},
            ],
        ]
        for segments in invalid_cases:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                paired_annotation.validate_segments(segments, 20)

    def test_annotation_round_trip_uses_requested_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            original_directory = paired_annotation.ANNOTATION_DIR
            paired_annotation.ANNOTATION_DIR = Path(directory)
            try:
                destination = paired_annotation.save_annotation(
                    "1",
                    [{"start_frame": 0, "end_frame": 9, "label": "Cup"}],
                    {"frames": 695},
                )
                payload = json.loads(destination.read_text(encoding="utf-8"))
                self.assertEqual(payload["episode"], "1")
                self.assertEqual(
                    set(payload), {"episode", "num_frames", "fps", "segments"}
                )
                self.assertEqual(payload["segments"][0]["label"], "Cup")
                self.assertEqual(destination.relative_to(Path(directory)), Path("1/gt_labels.json"))
                self.assertEqual(paired_annotation.load_annotation("1"), payload)
            finally:
                paired_annotation.ANNOTATION_DIR = original_directory


if __name__ == "__main__":
    unittest.main()
