import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prepare_0804_stereo_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_0804", MODULE_PATH)
prepare_0804 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_0804)


class Prepare0804DatasetTests(unittest.TestCase):
    def test_discovers_all_24_pairs_in_natural_order(self):
        episodes = prepare_0804.discover_episodes(prepare_0804.DEFAULT_SOURCE)
        self.assertEqual(episodes, [str(index) for index in range(1, 25)])

    def test_all_gt_files_are_complete_and_match_common_raw_length(self):
        source = prepare_0804.DEFAULT_SOURCE
        for name in prepare_0804.discover_episodes(source):
            with self.subTest(episode=name):
                gt = prepare_0804.load_and_validate_gt(
                    source / "annotations" / name / "gt_labels.json"
                )
                mh = prepare_0804.probe_frame_count(source / "mh" / f"{name}.mov")
                sh = prepare_0804.probe_frame_count(source / "sh" / f"{name}.mov")
                self.assertEqual(int(gt["num_frames"]), min(mh, sh))


if __name__ == "__main__":
    unittest.main()
