import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skill2policy-test-mpl")

import numpy as np
import torch
from PIL import Image

from src.skill_classifier import infer_long_horizon as inference


class LongHorizonAlignmentTests(unittest.TestCase):
    def test_frame_mapping_covers_full_tail(self):
        mapping = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        actual = inference._token_mapping(9, 3, 2, mapping)
        np.testing.assert_array_equal(actual, mapping)
        predictions = np.array([4, 2, 1])
        np.testing.assert_array_equal(
            predictions[actual],
            [4, 4, 4, 2, 2, 1, 1, 1, 1],
        )

    def test_discovers_jpg_episode_with_hawor_or_feature_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "1"
            rgb = episode / "rgb"
            rgb.mkdir(parents=True)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                rgb / "rgb_frame000000.jpg"
            )
            (episode / "features.pt").write_bytes(b"placeholder")
            found = inference.discover_episodes(directory)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["frame_names"], ["rgb_frame000000"])
            self.assertIsNotNone(found[0]["features_path"])

    def test_aligned_classifier_accepts_matching_token_features(self):
        class Dummy(torch.nn.Module):
            def forward(self, vjepa, hand):
                logits = torch.zeros(len(vjepa), 3, device=vjepa.device)
                logits[:, 1] = 2.0
                return logits

        preds, probabilities = inference.run_classifier_aligned(
            Dummy(),
            torch.zeros(3, 4),
            torch.zeros(3, 2),
            window_size=2,
            device=torch.device("cpu"),
        )
        np.testing.assert_array_equal(preds, [1, 1, 1])
        self.assertEqual(probabilities.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
