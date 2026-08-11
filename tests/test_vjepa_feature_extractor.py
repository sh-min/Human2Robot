import unittest

import torch

from src.data_preprocess.feature_extractor import (
    _clean_target_encoder_state_dict,
    _clean_vjepa21_encoder_state_dict,
)


class VjepaCheckpointCleaningTests(unittest.TestCase):
    def test_removes_ddp_prefix_and_only_known_rope_position_key(self):
        weight = torch.ones(2, 3)
        cleaned = _clean_target_encoder_state_dict(
            {
                "module.backbone.pos_embed": torch.zeros(1, 4, 8),
                "module.backbone.patch_embed.proj.weight": weight,
            }
        )
        self.assertEqual(
            set(cleaned),
            {"backbone.patch_embed.proj.weight"},
        )
        self.assertIs(cleaned["backbone.patch_embed.proj.weight"], weight)

    def test_rejects_duplicate_key_created_by_prefix_cleanup(self):
        with self.assertRaisesRegex(ValueError, "duplicate checkpoint key"):
            _clean_target_encoder_state_dict(
                {
                    "module.backbone.norm.weight": torch.ones(2),
                    "backbone.norm.weight": torch.ones(2),
                }
            )

    def test_rejects_non_mapping_payload(self):
        with self.assertRaisesRegex(TypeError, "state-dict mapping"):
            _clean_target_encoder_state_dict([("weight", torch.ones(1))])

    def test_vjepa21_maps_ddp_ema_encoder_to_bare_encoder(self):
        weight = torch.ones(2, 3)
        cleaned = _clean_vjepa21_encoder_state_dict(
            {"module.backbone.patch_embed.proj.weight": weight}
        )
        self.assertEqual(set(cleaned), {"patch_embed.proj.weight"})
        self.assertIs(cleaned["patch_embed.proj.weight"], weight)


if __name__ == "__main__":
    unittest.main()
