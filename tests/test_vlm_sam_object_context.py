import unittest

import numpy as np

from src.data_preprocess.extract_vlm_sam_object_context import (
    grounding_caption,
    mask_to_patch_occupancy,
    match_object_name,
    suppress_cross_object_duplicate_boxes,
)


PROMPTS = [
    {
        "name": "cup",
        "prompts": ["cup", "mug"],
        "grounding_queries": ["cup"],
        "max_instances": 2,
    },
    {
        "name": "cup_holder",
        "prompts": ["cup holder", "mug rack"],
        "grounding_queries": ["cup holder"],
        "max_instances": 1,
    },
]


class VlmSamObjectContextTests(unittest.TestCase):
    def test_prompt_bank_is_fixed_and_period_separated(self):
        self.assertEqual(grounding_caption(PROMPTS), "cup . cup holder .")

    def test_longer_exact_object_phrase_wins(self):
        self.assertEqual(match_object_name("cup holder", PROMPTS), "cup_holder")
        self.assertEqual(match_object_name("a mug", PROMPTS), "cup")
        self.assertIsNone(match_object_name("person", PROMPTS))

    def test_mask_is_aligned_to_patch_occupancy(self):
        mask = np.ones((80, 120), dtype=np.float32)
        occupancy = mask_to_patch_occupancy(
            mask,
            crop_size=32,
            spatial_profile="vjepa2_eval_center_crop",
        )
        self.assertEqual(occupancy.shape, (4,))
        np.testing.assert_allclose(occupancy, np.ones(4), atol=1.0e-6)

    def test_same_box_keeps_higher_scoring_semantic_name(self):
        detections = [
            {
                "name": "food_container",
                "score": 0.57,
                "box": np.array([0, 0, 10, 10], dtype=np.float32),
            },
            {
                "name": "trash_bin",
                "score": 0.83,
                "box": np.array([0, 0, 10, 10], dtype=np.float32),
            },
            {
                "name": "food_container",
                "score": 0.52,
                "box": np.array([20, 20, 30, 30], dtype=np.float32),
            },
        ]
        kept = suppress_cross_object_duplicate_boxes(detections)
        self.assertEqual(
            [(item["name"], item["score"]) for item in kept],
            [("trash_bin", 0.83), ("food_container", 0.52)],
        )


if __name__ == "__main__":
    unittest.main()
