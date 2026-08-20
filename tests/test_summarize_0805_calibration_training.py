"""Tests for the reproducible 08-05 classifier calibration A/B summary."""

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
    / "summarize_0805_calibration_training.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_0805_training", MODULE_PATH)
summary_tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary_tool)


FOLD_METRICS = {
    ("approx", "1_to_2"): (0.40, 0.30, 0.35),
    ("approx", "2_to_1"): (0.60, 0.50, 0.55),
    ("calibrated", "1_to_2"): (0.50, 0.40, 0.45),
    ("calibrated", "2_to_1"): (0.80, 0.70, 0.75),
}


def write_fixture_tree(root: Path, *, include_per_class: bool = True) -> None:
    supports = {"1_to_2": (2, 3), "2_to_1": (8, 0)}
    cup_accuracy = {
        ("approx", "1_to_2"): 0.50,
        ("approx", "2_to_1"): 0.75,
        ("calibrated", "1_to_2"): 1.00,
        ("calibrated", "2_to_1"): 0.875,
    }
    for (variant, fold), metrics in FOLD_METRICS.items():
        path = summary_tool.evaluation_path(root, variant, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": f"fixture_{variant}_{fold}",
            "variant": "vjepa_mano",
            "best_epoch": 7,
            "validation_accuracy": metrics[0],
            "validation_f1_macro": metrics[1],
            "validation_f1_weighted": metrics[2],
        }
        if include_per_class:
            cup_support, lock_support = supports[fold]
            payload["per_class"] = {
                "Cup": {
                    "support": cup_support,
                    "accuracy": cup_accuracy[(variant, fold)],
                },
                "Lock": {
                    "support": lock_support,
                    "accuracy": 1.0 if lock_support else None,
                },
            }
        path.write_text(json.dumps(payload), encoding="utf-8")


class CalibrationTrainingSummaryTests(unittest.TestCase):
    def test_fold_metrics_means_deltas_and_per_class_weighting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "classifier"
            write_fixture_tree(root)
            report = summary_tool.build_summary(root)

        approx = report["variants"]["approx"]
        calibrated = report["variants"]["calibrated"]
        self.assertAlmostEqual(approx["two_fold_mean"]["validation_accuracy"], 0.50)
        self.assertAlmostEqual(approx["two_fold_mean"]["validation_f1_macro"], 0.40)
        self.assertAlmostEqual(
            calibrated["two_fold_mean"]["validation_f1_weighted"], 0.60
        )
        self.assertAlmostEqual(
            report["calibrated_minus_approx"]["folds"]["2_to_1"][
                "validation_accuracy"
            ],
            0.20,
        )
        self.assertAlmostEqual(
            report["calibrated_minus_approx"]["two_fold_mean_delta"][
                "validation_accuracy"
            ],
            0.15,
        )

        cup = approx["per_class_two_fold"]["Cup"]
        self.assertEqual(cup["total_support"], 10)
        self.assertEqual(cup["folds_with_support"], 2)
        self.assertAlmostEqual(cup["accuracy_mean_across_supported_folds"], 0.625)
        self.assertAlmostEqual(cup["accuracy_support_weighted"], 0.70)
        lock = approx["per_class_two_fold"]["Lock"]
        self.assertEqual(lock["folds_with_support"], 1)
        self.assertAlmostEqual(lock["accuracy_mean_across_supported_folds"], 1.0)

    def test_writes_json_csv_and_markdown_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classifier_root = root / "classifier"
            output_dir = root / "reports"
            write_fixture_tree(classifier_root)
            report = summary_tool.build_summary(classifier_root)
            outputs = summary_tool.write_reports(report, output_dir, "fixture")

            self.assertEqual(set(outputs), {"json", "csv", "markdown"})
            saved = json.loads(outputs["json"].read_text(encoding="utf-8"))
            self.assertEqual(saved["comparison"], "calibrated_minus_approx")
            with outputs["csv"].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            markdown = outputs["markdown"].read_text(encoding="utf-8")

        self.assertTrue(any(row["row_type"] == "overall_fold" for row in rows))
        self.assertTrue(
            any(row["row_type"] == "per_class_two_fold" for row in rows)
        )
        self.assertIn("Calibrated minus approximate", markdown)
        self.assertIn("+15.00%", markdown)
        self.assertIn("positive classifier delta does not", markdown)

    def test_can_summarize_legacy_files_without_per_class_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture_tree(root, include_per_class=False)
            report = summary_tool.build_summary(root)

        self.assertEqual(report["class_names"], [])
        self.assertNotIn("per_class_two_fold", report["variants"]["approx"])

    def test_mixed_per_class_availability_is_rejected_with_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture_tree(root)
            missing_path = summary_tool.evaluation_path(root, "calibrated", "2_to_1")
            payload = json.loads(missing_path.read_text(encoding="utf-8"))
            del payload["per_class"]
            missing_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                summary_tool.SummaryInputError,
                r"per-class comparison requires.*0805_calibrated.*fold_2_to_1",
            ):
                summary_tool.build_summary(root)

    def test_missing_file_cli_error_identifies_variant_fold_and_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
                summary_tool.main(["--classifier-root", str(root)])

        self.assertEqual(ctx.exception.code, 2)
        message = stderr.getvalue()
        self.assertIn("variant='approx', fold='1_to_2'", message)
        self.assertIn("evaluation_summary.json", message)

    def test_invalid_metric_reports_exact_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture_tree(root)
            path = summary_tool.evaluation_path(root, "approx", "1_to_2")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["validation_f1_macro"] = 1.5
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                summary_tool.SummaryInputError,
                r"validation_f1_macro.*finite number in \[0, 1\]",
            ):
                summary_tool.build_summary(root)


if __name__ == "__main__":
    unittest.main()
