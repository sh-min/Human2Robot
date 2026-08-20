"""Fixture tests for the robust rot6d calibration A/B summarizer."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_0805_rot6d_calibration_training.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_0805_rot6d", MODULE_PATH)
summary_tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary_tool)


SEEDS = (42, 43, 44)
APPROX_ACCURACY = {
    42: (0.4, 0.6),
    43: (0.5, 0.5),
    44: (0.3, 0.7),
}
CALIBRATED_ACCURACY = {
    42: (0.5, 0.8),  # two-fold delta +0.15
    43: (0.6, 0.6),  # two-fold delta +0.10
    44: (0.5, 0.9),  # two-fold delta +0.20
}


def write_fixture_tree(root: Path) -> None:
    for variant, accuracies in (
        ("approx", APPROX_ACCURACY),
        ("calibrated", CALIBRATED_ACCURACY),
    ):
        for seed in SEEDS:
            for fold_index, fold in enumerate(summary_tool.FOLDS):
                accuracy = accuracies[seed][fold_index]
                path = summary_tool.evaluation_path(root, variant, seed, fold)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "experiment": f"fixture_{variant}_{seed}_{fold}",
                    "variant": "vjepa_mano",
                    "hand_representation": "rot6d",
                    "seed": seed,
                    "deterministic": True,
                    "best_epoch": 7,
                    "train_recordings": 1,
                    "validation_recordings": 1,
                    "train_samples": 10,
                    "validation_samples": 10,
                    "validation_loss": 0.75,
                    "validation_accuracy": accuracy,
                    "validation_f1_macro": accuracy - 0.1,
                    "validation_f1_weighted": accuracy - 0.05,
                    "per_class": {
                        "Cup": {"support": 4, "accuracy": 0.5},
                        "Lock": {"support": 6, "accuracy": 0.5},
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")


class Rot6dCalibrationSummaryTests(unittest.TestCase):
    def test_paired_fold_seed_and_across_seed_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "classifier"
            write_fixture_tree(root)
            report = summary_tool.build_summary(root, SEEDS)

        self.assertEqual(report["seeds"], [42, 43, 44])
        self.assertAlmostEqual(
            report["variants"]["calibrated"]["seeds"]["42"]["two_fold_mean"][
                "validation_accuracy"
            ],
            0.65,
        )
        paired = report["paired_calibrated_minus_approx"]
        self.assertAlmostEqual(
            paired["seeds"]["42"]["folds"]["2_to_1"]["validation_accuracy"],
            0.2,
        )
        self.assertAlmostEqual(
            paired["seeds"]["42"]["two_fold_mean_delta"][
                "validation_accuracy"
            ],
            0.15,
        )
        stats = paired["across_seed_two_fold_delta"]["validation_accuracy"]
        self.assertAlmostEqual(stats["mean"], 0.15)
        self.assertAlmostEqual(stats["sample_std"], 0.05)
        self.assertAlmostEqual(stats["min"], 0.10)
        self.assertAlmostEqual(stats["max"], 0.20)
        self.assertEqual(stats["seed_count"], 3)

    def test_validation_support_and_correct_count_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture_tree(root)
            path = summary_tool.evaluation_path(root, "approx", 42, "1_to_2")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["validation_accuracy"] = 0.333
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = summary_tool.build_summary(root, SEEDS)

        fold = report["variants"]["approx"]["seeds"]["42"]["folds"]["1_to_2"]
        count = fold["accuracy_correct_count_check"]
        self.assertEqual(count["support"], 10)
        self.assertAlmostEqual(count["accuracy_times_support"], 3.33)
        self.assertEqual(count["nearest_integer"], 3)
        self.assertFalse(count["is_near_integer"])
        self.assertIsNone(count["inferred_correct_count"])
        self.assertFalse(
            report["integrity_checks"]["all_accuracy_correct_counts_near_integer"]
        )
        self.assertTrue(
            report["integrity_checks"]["all_paired_validation_sample_counts_match"]
        )

    def test_paired_sample_count_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture_tree(root)
            path = summary_tool.evaluation_path(root, "calibrated", 43, "2_to_1")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["validation_samples"] = 12
            payload["per_class"]["Lock"]["support"] = 8
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = summary_tool.build_summary(root, SEEDS)

        paired = report["paired_calibrated_minus_approx"]["seeds"]["43"]
        self.assertEqual(
            paired["folds"]["2_to_1"]["validation_samples"],
            {"approx": 10, "calibrated": 12, "match": False},
        )
        self.assertFalse(
            report["integrity_checks"]["all_paired_validation_sample_counts_match"]
        )

    def test_writes_json_csv_markdown_and_caveats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classifier_root = root / "classifier"
            output_dir = root / "reports"
            write_fixture_tree(classifier_root)
            report = summary_tool.build_summary(classifier_root, SEEDS)
            outputs = summary_tool.write_reports(report, output_dir, "fixture")
            saved = json.loads(outputs["json"].read_text(encoding="utf-8"))
            with outputs["csv"].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            markdown = outputs["markdown"].read_text(encoding="utf-8")

        self.assertEqual(set(outputs), {"json", "csv", "markdown"})
        self.assertEqual(saved["representation"], "rot6d")
        self.assertTrue(any(row["row_type"] == "variant_fold" for row in rows))
        self.assertTrue(
            any(
                row["row_type"] == "paired_across_seed_two_fold_delta"
                and row["statistic"] == "sample_std"
                for row in rows
            )
        )
        self.assertIn("Only two recordings", markdown)
        self.assertIn("best epoch is selected", markdown)
        self.assertIn("MH scalar-focal-length", markdown)
        self.assertIn("Accuracy correct-count audit", markdown)

    def test_missing_file_cli_error_identifies_variant_seed_fold_and_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
                summary_tool.main(
                    ["--classifier-root", str(root), "--seeds", "42"]
                )

        self.assertEqual(ctx.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn("variant='approx', seed=42, fold='1_to_2'", message)
        self.assertIn("evaluation_summary.json", message)

    def test_invalid_validation_samples_and_metric_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture_tree(root)
            path = summary_tool.evaluation_path(root, "approx", 42, "1_to_2")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["validation_samples"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                summary_tool.SummaryInputError,
                r"validation_samples.*positive integer",
            ):
                summary_tool.build_summary(root, SEEDS)

            payload["validation_samples"] = 10
            payload["validation_f1_macro"] = float("nan")
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                summary_tool.SummaryInputError,
                r"validation_f1_macro.*finite number in \[0, 1\]",
            ):
                summary_tool.build_summary(root, SEEDS)

    def test_cli_seeds_default_and_explicit_values(self):
        parser = summary_tool.build_parser()
        self.assertEqual(parser.parse_args([]).seeds, [42, 43, 44, 45, 46])
        self.assertEqual(parser.parse_args(["--seeds", "7", "9"]).seeds, [7, 9])


if __name__ == "__main__":
    unittest.main()
