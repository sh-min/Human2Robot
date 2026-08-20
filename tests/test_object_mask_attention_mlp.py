import unittest

import torch

from src.skill_classifier.models.object_mask_attention_mlp import (
    ObjectMaskAttentionMLP,
)


class ObjectMaskAttentionMLPTests(unittest.TestCase):
    def test_forward_and_diagnostic_shapes(self):
        batch, window, objects, patches, dimension = 2, 3, 4, 5, 16
        model = ObjectMaskAttentionMLP(
            vjepa_dim=dimension,
            hand_dim=objects * (patches + 1),
            window_size=window,
            num_classes=6,
            hidden_dims=(8,),
            object_prompt_count=objects,
            object_mask_spatial_tokens=patches,
            object_projection_dim=6,
        )
        dense = torch.randn(batch, window, patches, dimension)
        masks = torch.rand(batch, window, objects, patches)
        confidence = torch.rand(batch, window, objects)
        context = torch.cat([masks.flatten(2), confidence], dim=-1)
        fused, diagnostics = model.pool_dense(dense, context)
        logits = model(dense, context)
        self.assertEqual(tuple(logits.shape), (batch, 6))
        self.assertEqual(
            tuple(fused.shape),
            (batch, dimension + objects * (6 + 2)),
        )
        self.assertEqual(
            tuple(diagnostics["object_masks"].shape),
            (batch, window, objects, patches),
        )

    def test_rejects_wrong_context_dimension(self):
        with self.assertRaisesRegex(ValueError, "context dimension"):
            ObjectMaskAttentionMLP(
                vjepa_dim=16,
                hand_dim=10,
                object_prompt_count=2,
                object_mask_spatial_tokens=5,
            )

    def test_ablation_flags_change_only_the_fused_inputs(self):
        batch, window, objects, patches, dimension = 2, 3, 4, 5, 16
        context = torch.rand(batch, window, objects * (patches + 1))
        dense = torch.randn(batch, window, patches, dimension)
        cases = [
            ({"use_object_features": False}, dimension + objects * 2),
            (
                {
                    "use_global_features": False,
                    "use_confidence_features": False,
                    "use_occupancy_features": False,
                },
                objects * 6,
            ),
            (
                {
                    "use_global_features": False,
                    "use_object_features": False,
                },
                objects * 2,
            ),
        ]
        for flags, expected_dim in cases:
            with self.subTest(flags=flags):
                model = ObjectMaskAttentionMLP(
                    vjepa_dim=dimension,
                    hand_dim=objects * (patches + 1),
                    window_size=window,
                    num_classes=6,
                    hidden_dims=(8,),
                    object_prompt_count=objects,
                    object_mask_spatial_tokens=patches,
                    object_projection_dim=6,
                    **flags,
                )
                fused, _ = model.pool_dense(dense, context)
                self.assertEqual(tuple(fused.shape), (batch, expected_dim))
                self.assertEqual(tuple(model(dense, context).shape), (batch, 6))

    def test_rejects_ablation_with_no_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ObjectMaskAttentionMLP(
                vjepa_dim=16,
                hand_dim=12,
                object_prompt_count=2,
                object_mask_spatial_tokens=5,
                use_global_features=False,
                use_object_features=False,
                use_confidence_features=False,
                use_occupancy_features=False,
            )


if __name__ == "__main__":
    unittest.main()
