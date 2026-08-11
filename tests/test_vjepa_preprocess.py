import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data_preprocess import preprocess


class VjepaAlignmentTests(unittest.TestCase):
    def test_vjepa2_profile_samples_4fps_pairs_and_preserves_tail(self):
        alignment = preprocess.build_token_alignment(
            25,
            24.0,
            sampling_profile=preprocess.VJEPA2_4FPS_PROFILE,
            sample_fps=4.0,
        )
        np.testing.assert_array_equal(
            alignment.sampled_frame_indices,
            [0, 6, 12, 18, 24],
        )
        np.testing.assert_array_equal(
            alignment.token_frame_indices,
            [[0, 6], [12, 18], [24, 24]],
        )
        np.testing.assert_array_equal(
            alignment.token_center_frame_indices,
            [3, 15, 24],
        )
        self.assertEqual(alignment.num_tokens, 3)
        self.assertEqual(len(alignment.frame_to_token), 25)
        self.assertEqual(int(alignment.frame_to_token[-1]), 2)

    def test_episode_one_contract_has_58_tokens(self):
        alignment = preprocess.build_token_alignment(
            695,
            24.0,
            sampling_profile=preprocess.VJEPA2_4FPS_PROFILE,
            sample_fps=4.0,
        )
        self.assertEqual(len(alignment.sampled_frame_indices), 116)
        self.assertEqual(alignment.num_tokens, 58)

    def test_official_spatial_profile_center_crops_without_square_warp(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        image[:, :640, 0] = 255
        image[:, 640:, 2] = 255
        result = preprocess.preprocess_rgb_frame(
            image,
            256,
            preprocess.VJEPA2_EVAL_CROP,
        )
        self.assertEqual(result.shape, (256, 256, 3))
        self.assertGreater(float(result[:, :32, 0].mean()), 240.0)
        self.assertGreater(float(result[:, -32:, 2].mean()), 240.0)

    def test_boundary_tokens_are_ignored_or_center_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gt_labels.json").write_text(
                json.dumps(
                    {
                        "fps": 24.0,
                        "segments": [
                            {"label": "Cup", "start_frame": 0, "end_frame": 14},
                            {"label": "Milk", "start_frame": 15, "end_frame": 24},
                        ],
                    }
                )
            )
            alignment = preprocess.build_token_alignment(
                25,
                24.0,
                sampling_profile=preprocess.VJEPA2_4FPS_PROFILE,
                sample_fps=4.0,
            )
            ignored = preprocess.labels_for_alignment(
                root, alignment, boundary_policy="ignore"
            )
            centered = preprocess.labels_for_alignment(
                root, alignment, boundary_policy="center"
            )
            cup = preprocess.ACTION_LABELS.index("Cup")
            milk = preprocess.ACTION_LABELS.index("Milk")
            torch.testing.assert_close(
                ignored,
                torch.tensor([cup, -1, milk], dtype=torch.int32),
            )
            torch.testing.assert_close(
                centered,
                torch.tensor([cup, milk, milk], dtype=torch.int32),
            )

    def test_dataset_scoped_choco_vocabulary_keeps_six_contiguous_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gt_labels.json").write_text(
                json.dumps(
                    {
                        "fps": 24.0,
                        "segments": [
                            {"label": "Choco", "start_frame": 0, "end_frame": 24}
                        ],
                    }
                )
            )
            alignment = preprocess.build_token_alignment(
                25,
                24.0,
                sampling_profile=preprocess.VJEPA2_4FPS_PROFILE,
                sample_fps=4.0,
            )
            labels = ["Cup", "Lock", "Choco", "Snack", "Sweep", "Trans"]
            actual = preprocess.labels_for_alignment(
                root,
                alignment,
                action_labels=labels,
            )
            torch.testing.assert_close(
                actual,
                torch.full((3,), 2, dtype=torch.int32),
            )

    def test_mano_uses_center_frame_and_never_averages_invalid_pose(self):
        alignment = preprocess.build_token_alignment(
            25,
            24.0,
            sampling_profile=preprocess.VJEPA2_4FPS_PROFILE,
            sample_fps=4.0,
        )
        features = np.zeros((25, 2, 48), dtype=np.float32)
        valid = np.ones((25, 2), dtype=bool)
        for frame in range(25):
            features[frame] = frame
        valid[15, 1] = False
        mano, token_valid = preprocess.align_mano_to_tokens(
            features, valid, alignment
        )
        self.assertTrue(torch.all(mano[0] == 3))
        self.assertTrue(torch.all(mano[1, :48] == 15))
        self.assertTrue(torch.all(mano[1, 48:] == 0))
        self.assertFalse(bool(token_valid[1, 1]))

    def test_aligned_extractor_returns_declared_token_count(self):
        class FakeExtractor(torch.nn.Module):
            def forward(self, batch):
                batch_size = len(batch)
                tokens = (batch.shape[2] // 2) * (batch.shape[3] // 16) ** 2
                return torch.zeros(batch_size, tokens, 1024)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for frame in range(25):
                image = np.full((20, 32, 3), frame, dtype=np.uint8)
                from PIL import Image

                Image.fromarray(image).save(root / f"rgb_frame{frame:06d}.jpg")
            alignment = preprocess.build_token_alignment(
                25,
                24.0,
                sampling_profile=preprocess.VJEPA2_4FPS_PROFILE,
                sample_fps=4.0,
            )
            features, source_count = preprocess.extract_vjepa(
                root,
                FakeExtractor(),
                "cpu",
                16,
                16,
                2,
                alignment=alignment,
                spatial_profile=preprocess.VJEPA2_EVAL_CROP,
            )
            self.assertEqual(source_count, 25)
            self.assertEqual(tuple(features.shape), (3, 1024))


class FeatureBundleValidationTests(unittest.TestCase):
    def _bundle(self):
        return {
            "feature_schema_version": 2,
            "vjepa_orig": torch.zeros(3, 1024),
            "mano": torch.zeros(3, 96),
            "mano_valid_per_token": torch.ones(3, 2, dtype=torch.bool),
            "labels_per_token": torch.zeros(3, dtype=torch.int32),
            "num_frames": 25,
            "num_tokens": 3,
            "recording": "demo",
            "sampling_profile": preprocess.VJEPA2_4FPS_PROFILE,
            "source_fps": 24.0,
            "sample_fps": 4.0,
            "token_rate_hz": 2.0,
            "clip_frames": 16,
            "tubelet_size": 2,
            "spatial_profile": preprocess.VJEPA2_EVAL_CROP,
            "sampled_frame_indices": torch.tensor([0, 6, 12, 18, 24]),
            "token_frame_indices": torch.tensor([[0, 6], [12, 18], [24, 24]]),
            "token_center_frame_indices": torch.tensor([3, 15, 24]),
            "frame_to_token": torch.tensor(
                [0] * 10 + [1] * 10 + [2] * 5
            ),
        }

    def test_valid_bundle(self):
        preprocess.validate_feature_bundle(self._bundle())

    def test_rejects_misaligned_mano(self):
        bundle = self._bundle()
        bundle["mano"] = torch.zeros(2, 96)
        with self.assertRaisesRegex(ValueError, "mano shape"):
            preprocess.validate_feature_bundle(bundle)

    def test_rejects_label_outside_bundle_vocabulary(self):
        bundle = self._bundle()
        bundle["action_labels"] = ["Cup", "Trans"]
        bundle["labels_per_token"][0] = 2
        with self.assertRaisesRegex(ValueError, "outside action_labels"):
            preprocess.validate_feature_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
