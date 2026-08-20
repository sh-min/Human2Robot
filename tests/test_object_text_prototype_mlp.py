import unittest

import torch

from src.skill_classifier.models.object_text_prototype_mlp import (
    ObjectTextPrototypeMLP,
)


class ObjectTextPrototypeMLPTests(unittest.TestCase):
    def build_model(self, mode="prototype"):
        return ObjectTextPrototypeMLP(
            vjepa_dim=16,
            hand_dim=12,
            window_size=3,
            num_classes=4,
            hidden_dims=(8,),
            object_prompt_count=2,
            object_mask_spatial_tokens=5,
            object_projection_dim=3,
            text_embedding_dim=6,
            text_head_mode=mode,
        )

    def test_prototype_and_hybrid_forward(self):
        dense = torch.randn(2, 3, 5, 16)
        context = torch.rand(2, 3, 12)
        for mode in ("prototype", "hybrid"):
            with self.subTest(mode=mode):
                model = self.build_model(mode)
                model.set_action_text_embeddings(torch.randn(4, 6))
                self.assertEqual(tuple(model(dense, context).shape), (2, 4))

    def test_rejects_wrong_text_shape(self):
        with self.assertRaisesRegex(ValueError, "must have shape"):
            self.build_model().set_action_text_embeddings(torch.randn(3, 6))


if __name__ == "__main__":
    unittest.main()
