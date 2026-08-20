import unittest

import torch

from src.data_preprocess.derive_object_context_ablations import (
    derive_sidecar,
    masks_to_patch_boxes,
)


class ObjectContextAblationTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "schema_version": 1,
            "masks": torch.zeros(4, 3, 16, dtype=torch.float16),
            "confidence": torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12,
            "object_names": ["a", "b", "c"],
        }
        self.source["masks"][0, 0, [5, 10]] = 1

    def test_bbox_encloses_mask_on_patch_grid(self):
        boxed = masks_to_patch_boxes(self.source["masks"])
        self.assertEqual(int(boxed[0, 0].sum()), 4)
        self.assertTrue(torch.all(boxed[0, 0, [5, 6, 9, 10]] == 1))

    def test_zero_is_true_negative_control(self):
        derived = derive_sidecar(self.source, "zero", "episode")
        self.assertEqual(int(derived["masks"].count_nonzero()), 0)
        self.assertEqual(int(derived["confidence"].count_nonzero()), 0)
        self.assertGreater(int(self.source["confidence"].count_nonzero()), 0)

    def test_random_controls_are_reproducible(self):
        for variant in ("channel_shuffle", "temporal_shuffle"):
            first = derive_sidecar(self.source, variant, "episode")
            second = derive_sidecar(self.source, variant, "episode")
            self.assertTrue(torch.equal(first["masks"], second["masks"]))
            self.assertTrue(torch.equal(first["confidence"], second["confidence"]))


if __name__ == "__main__":
    unittest.main()
