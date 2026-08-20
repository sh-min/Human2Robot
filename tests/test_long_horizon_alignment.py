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

    def test_aligned_classifier_accepts_dense_token_features(self):
        class Dummy(torch.nn.Module):
            def forward(self, vjepa, hand):
                self.seen = (tuple(vjepa.shape), tuple(hand.shape))
                return torch.zeros(len(vjepa), 2, device=vjepa.device)

        model = Dummy()
        inference.run_classifier_aligned(
            model,
            torch.zeros(3, 5, 4),
            torch.zeros(3, 7),
            window_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(model.seen, ((1, 2, 5, 4), (1, 2, 7)))

    def test_object_context_sidecar_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            feature_path = Path(directory) / "features.pt"
            feature_path.touch()
            torch.save(
                {
                    "schema_version": 1,
                    "object_names": ["cup", "sponge"],
                    "masks": torch.ones(3, 2, 4),
                    "confidence": torch.full((3, 2), 0.5),
                    "token_center_frame_indices": torch.tensor([1, 3, 5]),
                },
                Path(directory) / "vlm_sam.pt",
            )
            context = inference.load_object_context_sidecar(
                feature_path,
                {
                    "num_tokens": 3,
                    "token_center_frame_indices": torch.tensor([1, 3, 5]),
                },
                {
                    "object_context_key": "vlm_sam",
                    "object_names": ["cup", "sponge"],
                    "object_prompt_count": 2,
                    "object_mask_spatial_tokens": 4,
                },
            )
            self.assertEqual(tuple(context.shape), (3, 10))


if __name__ == "__main__":
    unittest.main()
