import unittest

import torch

from src.skill_classifier.skill_dataset import (
    SkillWindowDataset,
    axis_angle_to_rotation_6d,
    mano_axis_angle_to_rotation_6d,
)


def recording(profile="vjepa2_4fps", tokens=3):
    return {
        "mano": torch.zeros(tokens, 96),
        "mano_valid_per_token": torch.ones(tokens, 2, dtype=torch.bool),
        "vjepa_orig": torch.zeros(tokens, 1024),
        "labels_per_token": torch.zeros(tokens, dtype=torch.int32),
        "sampling_profile": profile,
        "sample_fps": 4.0 if profile == "vjepa2_4fps" else 24.0,
        "token_rate_hz": 2.0 if profile == "vjepa2_4fps" else 12.0,
        "clip_frames": 16,
        "tubelet_size": 2,
        "spatial_profile": (
            "vjepa2_eval_center_crop"
            if profile == "vjepa2_4fps"
            else "legacy_stretch"
        ),
    }


class SkillDatasetSamplingTests(unittest.TestCase):
    def test_rejects_mixed_sampling_contracts(self):
        with self.assertRaisesRegex(ValueError, "mixed feature sampling"):
            SkillWindowDataset(
                [recording(), recording("legacy_dense")],
                variant="vjepa_orig",
            )

    def test_rejects_feature_mano_length_mismatch(self):
        value = recording()
        value["vjepa_orig"] = torch.zeros(2, 1024)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            SkillWindowDataset([value], variant="vjepa_orig")

    def test_exposes_sampling_signature(self):
        dataset = SkillWindowDataset([recording()], variant="vjepa_orig")
        self.assertEqual(dataset.sampling_signature[0], "vjepa2_4fps")
        self.assertEqual(dataset.sampling_signature[2], 2.0)

    def test_rot6d_is_continuous_across_equivalent_pi_branches(self):
        first = torch.tensor([[3.13, 0.0, 0.0]], dtype=torch.float32)
        equivalent = torch.tensor(
            [[-(2.0 * torch.pi - 3.13), 0.0, 0.0]], dtype=torch.float32
        )
        converted = axis_angle_to_rotation_6d(torch.cat([first, equivalent]))
        torch.testing.assert_close(converted[0], converted[1], atol=1.0e-6, rtol=0)

    def test_rot6d_masks_invalid_hands_back_to_zero(self):
        mano = torch.zeros(1, 96)
        validity = torch.tensor([[True, False]])
        converted = mano_axis_angle_to_rotation_6d(mano, validity)
        self.assertEqual(tuple(converted.shape), (1, 192))
        expected_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        torch.testing.assert_close(converted[0, :6], expected_rotation)
        self.assertTrue(torch.equal(converted[0, 96:], torch.zeros(96)))

    def test_dataset_rot6d_doubles_hand_dimension(self):
        dataset = SkillWindowDataset(
            [recording()],
            variant="vjepa_orig",
            hand_representation="rot6d",
        )
        self.assertEqual(dataset.hand_dim, 192)
        _, hand, _ = dataset[0]
        self.assertEqual(tuple(hand.shape), (8, 192))

    def test_none_hand_representation_builds_vjepa_only_windows(self):
        dataset = SkillWindowDataset(
            [recording()],
            variant="vjepa_orig",
            hand_representation="none",
        )
        self.assertEqual(dataset.hand_dim, 0)
        vjepa, hand, _ = dataset[0]
        self.assertEqual(tuple(vjepa.shape), (8, 1024))
        self.assertEqual(tuple(hand.shape), (8, 0))

    def test_dense_variant_preserves_patch_axis_and_dtype_when_padding(self):
        value = recording()
        value["vjepa_orig_dense"] = torch.ones(3, 4, 1024, dtype=torch.float16)
        dataset = SkillWindowDataset(
            [value],
            variant="vjepa_orig_dense",
            hand_representation="none",
        )
        self.assertEqual(dataset.vjepa_dim, 1024)
        self.assertEqual(dataset.vjepa_spatial_tokens, 4)
        vjepa, _, _ = dataset[0]
        self.assertEqual(tuple(vjepa.shape), (8, 4, 1024))
        self.assertEqual(vjepa.dtype, torch.float16)
        self.assertTrue(torch.equal(vjepa[:-1], torch.zeros_like(vjepa[:-1])))


if __name__ == "__main__":
    unittest.main()
