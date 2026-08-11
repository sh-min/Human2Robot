import unittest

import torch

from src.skill_classifier.models.spatial_attention_mlp import SpatialAttentionMLP


class SpatialAttentionMLPTests(unittest.TestCase):
    def test_forward_and_attention_probability_contract(self):
        model = SpatialAttentionMLP(
            vjepa_dim=16,
            hand_dim=0,
            window_size=3,
            num_classes=6,
            hidden_dims=(8,),
        )
        dense = torch.randn(2, 3, 5, 16, dtype=torch.float16)
        hand = torch.zeros(2, 3, 0)
        pooled, weights = model.pool_dense(dense)
        logits = model(dense, hand)
        self.assertEqual(tuple(pooled.shape), (2, 16))
        self.assertEqual(tuple(weights.shape), (2, 3, 5))
        self.assertEqual(tuple(logits.shape), (2, 6))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))

    def test_rejects_spatially_pooled_input(self):
        model = SpatialAttentionMLP(vjepa_dim=16, hand_dim=0)
        with self.assertRaisesRegex(ValueError, "expects"):
            model(torch.randn(2, 8, 16), torch.zeros(2, 8, 0))


if __name__ == "__main__":
    unittest.main()
