"""Fixture tests for the 08-05 classifier feature A/B validator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_0805_feature_ab as validator  # noqa: E402


def _bundle() -> dict:
    return {
        "feature_schema_version": 2,
        "recording": "1",
        "action_labels": ["Cup", "Choco", "Trans"],
        "num_frames": 12,
        "num_tokens": 2,
        "sampling_profile": "vjepa2_4fps",
        "source_fps": 24.0,
        "sample_fps": 4.0,
        "token_rate_hz": 2.0,
        "clip_frames": 16,
        "tubelet_size": 2,
        "spatial_profile": "vjepa2_eval_center_crop",
        "label_boundary_policy": "token_center",
        "labels_per_token": torch.tensor([0, 1], dtype=torch.int32),
        "sampled_frame_indices": torch.tensor([0, 6, 11], dtype=torch.int64),
        "token_frame_indices": torch.tensor([[0, 6], [11, 11]]),
        "token_center_frame_indices": torch.tensor([3, 11]),
        "frame_to_token": torch.tensor([0] * 7 + [1] * 5),
        "vjepa_orig": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "mano": torch.zeros(2, 6, dtype=torch.float32),
        "mano_valid_per_token": torch.tensor(
            [[True, False], [True, True]], dtype=torch.bool
        ),
    }


class FeatureABValidatorTests(unittest.TestCase):
    def _roots(self, directory: str) -> tuple[Path, Path]:
        root = Path(directory)
        approx = root / "approx"
        calibrated = root / "calibrated"
        (approx / "1").mkdir(parents=True)
        (calibrated / "1").mkdir(parents=True)
        return approx, calibrated

    def test_fixture_passes_and_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            approx_root, calibrated_root = self._roots(directory)
            approx = _bundle()
            calibrated = _bundle()
            calibrated["vjepa_orig"] += 5.0e-7
            calibrated["mano"][0, 0] = 2.0
            calibrated["mano_valid_per_token"][0, 1] = True
            torch.save(approx, approx_root / "1" / "features.pt")
            torch.save(calibrated, calibrated_root / "1" / "features.pt")
            output = Path(directory) / "report"

            report = validator.validate_feature_ab(
                approx_root,
                calibrated_root,
                episodes=["1"],
                atol=1.0e-6,
                output_dir=output,
            )

            self.assertEqual(report["status"], "passed")
            self.assertGreater(
                report["episodes"][0]["mano_difference"]["max_abs_difference"],
                0.0,
            )
            self.assertAlmostEqual(
                report["episodes"][0]["mano_validity"]["agreement_rate"],
                0.75,
            )
            json_report = json.loads(
                (output / "feature_ab_validation.json").read_text()
            )
            self.assertEqual(json_report["aggregate"]["num_episodes"], 1)
            markdown = (output / "feature_ab_validation.md").read_text()
            self.assertIn("Status: PASS", markdown)
            self.assertIn(str((approx_root / "1" / "features.pt").resolve()), markdown)

    def test_identical_mano_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            approx_root, calibrated_root = self._roots(directory)
            torch.save(_bundle(), approx_root / "1" / "features.pt")
            torch.save(_bundle(), calibrated_root / "1" / "features.pt")
            report = validator.validate_feature_ab(
                approx_root,
                calibrated_root,
                episodes=["1"],
                output_dir=None,
            )
            self.assertTrue(report["aggregate"]["mano_difference"]["identical"])

    def test_rejects_token_alignment_mismatch_with_field_name(self):
        with tempfile.TemporaryDirectory() as directory:
            approx_root, calibrated_root = self._roots(directory)
            approx = _bundle()
            calibrated = _bundle()
            calibrated["token_center_frame_indices"][1] = 10
            torch.save(approx, approx_root / "1" / "features.pt")
            torch.save(calibrated, calibrated_root / "1" / "features.pt")
            with self.assertRaisesRegex(
                validator.FeatureABValidationError,
                "token_center_frame_indices.*differs",
            ):
                validator.validate_feature_ab(
                    approx_root,
                    calibrated_root,
                    episodes=["1"],
                    output_dir=None,
                )

    def test_rejects_vjepa_difference_above_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            approx_root, calibrated_root = self._roots(directory)
            approx = _bundle()
            calibrated = _bundle()
            calibrated["vjepa_orig"][1, 2] += 2.0e-4
            torch.save(approx, approx_root / "1" / "features.pt")
            torch.save(calibrated, calibrated_root / "1" / "features.pt")
            with self.assertRaisesRegex(
                validator.FeatureABValidationError,
                "vjepa_orig max absolute difference.*exceeds atol",
            ):
                validator.validate_feature_ab(
                    approx_root,
                    calibrated_root,
                    episodes=["1"],
                    atol=1.0e-6,
                    output_dir=None,
                )

    def test_missing_bundle_has_clear_path_error(self):
        with tempfile.TemporaryDirectory() as directory:
            approx_root, calibrated_root = self._roots(directory)
            torch.save(_bundle(), approx_root / "1" / "features.pt")
            missing = calibrated_root / "1" / "features.pt"
            with self.assertRaisesRegex(
                validator.FeatureABValidationError,
                f"missing feature bundle: {missing}",
            ):
                validator.validate_feature_ab(
                    approx_root,
                    calibrated_root,
                    episodes=["1"],
                    output_dir=None,
                )


if __name__ == "__main__":
    unittest.main()
