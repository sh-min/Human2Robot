#!/usr/bin/env python3
"""Summarize the 08-05 calibrated-vs-approximate classifier experiments.

The expected experiment layout is::

    output/skill_classifier/0805_{approx,calibrated}/
        fold_{1_to_2,2_to_1}/evaluation_summary.json

The report keeps every held-out-fold metric, computes an unweighted two-fold
mean, and reports all calibration deltas as ``calibrated - approx``.  If the
training summaries contain per-class results, both the mean across supported
folds and a support-weighted accuracy are included.

This comparison measures sensitivity to the calibration branch.  It does not
by itself establish which branch is geometrically more accurate because the
classifier validation labels do not contain calibration ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER_ROOT = REPO_ROOT / "output" / "skill_classifier"
DEFAULT_OUTPUT_DIR = DEFAULT_CLASSIFIER_ROOT / "0805_calibration_comparison"
DEFAULT_PREFIX = "0805_calibration_training_summary"

VARIANTS = ("approx", "calibrated")
FOLDS = ("1_to_2", "2_to_1")
METRICS = (
    "validation_accuracy",
    "validation_f1_macro",
    "validation_f1_weighted",
)


class SummaryInputError(ValueError):
    """Raised when an experiment summary is missing or malformed."""


def evaluation_path(classifier_root: Path, variant: str, fold: str) -> Path:
    return (
        classifier_root
        / f"0805_{variant}"
        / f"fold_{fold}"
        / "evaluation_summary.json"
    )


def _required_number(payload: dict[str, Any], field: str, path: Path) -> float:
    if field not in payload:
        raise SummaryInputError(f"{path}: missing required field {field!r}")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryInputError(
            f"{path}: field {field!r} must be a finite number in [0, 1], "
            f"got {value!r}"
        )
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SummaryInputError(
            f"{path}: field {field!r} must be a finite number in [0, 1], "
            f"got {value!r}"
        )
    return value


def _per_class_metrics(payload: dict[str, Any], path: Path) -> dict[str, dict[str, Any]] | None:
    if "per_class" not in payload:
        return None
    raw = payload["per_class"]
    if not isinstance(raw, dict) or not raw:
        raise SummaryInputError(
            f"{path}: field 'per_class' must be a non-empty object when present"
        )

    parsed: dict[str, dict[str, Any]] = {}
    for class_name, class_payload in raw.items():
        if not isinstance(class_name, str) or not class_name:
            raise SummaryInputError(f"{path}: per_class contains an invalid class name")
        if not isinstance(class_payload, dict):
            raise SummaryInputError(
                f"{path}: per_class[{class_name!r}] must be an object"
            )
        if "support" not in class_payload or "accuracy" not in class_payload:
            raise SummaryInputError(
                f"{path}: per_class[{class_name!r}] requires 'support' and 'accuracy'"
            )
        support = class_payload["support"]
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise SummaryInputError(
                f"{path}: per_class[{class_name!r}].support must be a "
                f"non-negative integer, got {support!r}"
            )
        accuracy = class_payload["accuracy"]
        if accuracy is None:
            if support > 0:
                raise SummaryInputError(
                    f"{path}: per_class[{class_name!r}].accuracy is null despite "
                    f"support={support}"
                )
        else:
            accuracy = _required_number(
                {"accuracy": accuracy}, "accuracy",
                Path(f"{path}:per_class[{class_name!r}]"),
            )
        parsed[class_name] = {"support": support, "accuracy": accuracy}
    return parsed


def load_evaluation_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SummaryInputError(f"missing evaluation summary: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SummaryInputError(
            f"{path}: invalid JSON at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error
    except OSError as error:
        raise SummaryInputError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SummaryInputError(f"{path}: expected a JSON object")

    result: dict[str, Any] = {
        metric: _required_number(payload, metric, path) for metric in METRICS
    }
    result["per_class"] = _per_class_metrics(payload, path)
    for optional in (
        "experiment",
        "variant",
        "hand_representation",
        "seed",
        "deterministic",
        "best_epoch",
        "train_recordings",
        "validation_recordings",
        "train_samples",
        "validation_samples",
        "validation_loss",
    ):
        if optional in payload:
            result[optional] = payload[optional]
    return result


def _mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(values))


def _validate_class_schema(
    evaluations: dict[str, dict[str, dict[str, Any]]],
    paths: dict[str, dict[str, Path]],
) -> list[str]:
    present = [
        evaluations[variant][fold]["per_class"] is not None
        for variant in VARIANTS
        for fold in FOLDS
    ]
    if not any(present):
        return []
    if not all(present):
        missing = [
            str(paths[variant][fold])
            for variant in VARIANTS
            for fold in FOLDS
            if evaluations[variant][fold]["per_class"] is None
        ]
        raise SummaryInputError(
            "per-class comparison requires 'per_class' in all four summaries; "
            f"missing from: {', '.join(missing)}"
        )

    reference = list(evaluations[VARIANTS[0]][FOLDS[0]]["per_class"])
    reference_set = set(reference)
    for variant in VARIANTS:
        for fold in FOLDS:
            actual = set(evaluations[variant][fold]["per_class"])
            if actual != reference_set:
                missing = sorted(reference_set - actual)
                extra = sorted(actual - reference_set)
                raise SummaryInputError(
                    f"{paths[variant][fold]}: per_class vocabulary differs from "
                    f"the other folds (missing={missing}, extra={extra})"
                )
    return reference


def build_summary(classifier_root: Path) -> dict[str, Any]:
    classifier_root = classifier_root.expanduser().resolve()
    paths = {
        variant: {
            fold: evaluation_path(classifier_root, variant, fold)
            for fold in FOLDS
        }
        for variant in VARIANTS
    }
    evaluations: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANTS:
        evaluations[variant] = {}
        for fold in FOLDS:
            path = paths[variant][fold]
            try:
                evaluations[variant][fold] = load_evaluation_summary(path)
            except SummaryInputError as error:
                raise SummaryInputError(
                    f"variant={variant!r}, fold={fold!r}: {error}"
                ) from error

    class_names = _validate_class_schema(evaluations, paths)
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        fold_reports: dict[str, Any] = {}
        for fold in FOLDS:
            evaluation = evaluations[variant][fold]
            fold_report = {
                "source": str(paths[variant][fold]),
                **{metric: evaluation[metric] for metric in METRICS},
            }
            for optional in (
                "experiment",
                "variant",
                "hand_representation",
                "seed",
                "deterministic",
                "best_epoch",
                "train_recordings",
                "validation_recordings",
                "train_samples",
                "validation_samples",
                "validation_loss",
            ):
                if optional in evaluation:
                    fold_report[optional] = evaluation[optional]
            if class_names:
                fold_report["per_class"] = evaluation["per_class"]
            fold_reports[fold] = fold_report

        variant_report: dict[str, Any] = {
            "folds": fold_reports,
            "two_fold_mean": {
                metric: _mean(evaluations[variant][fold][metric] for fold in FOLDS)
                for metric in METRICS
            },
        }
        if class_names:
            per_class: dict[str, Any] = {}
            for class_name in class_names:
                supported = [
                    evaluations[variant][fold]["per_class"][class_name]
                    for fold in FOLDS
                    if evaluations[variant][fold]["per_class"][class_name]["support"] > 0
                ]
                total_support = sum(row["support"] for row in supported)
                per_class[class_name] = {
                    "total_support": total_support,
                    "folds_with_support": len(supported),
                    "accuracy_mean_across_supported_folds": (
                        _mean(row["accuracy"] for row in supported) if supported else None
                    ),
                    "accuracy_support_weighted": (
                        sum(row["accuracy"] * row["support"] for row in supported)
                        / total_support
                        if total_support
                        else None
                    ),
                }
            variant_report["per_class_two_fold"] = per_class
        variants[variant] = variant_report

    fold_deltas = {
        fold: {
            metric: (
                evaluations["calibrated"][fold][metric]
                - evaluations["approx"][fold][metric]
            )
            for metric in METRICS
        }
        for fold in FOLDS
    }
    mean_delta = {
        metric: (
            variants["calibrated"]["two_fold_mean"][metric]
            - variants["approx"]["two_fold_mean"][metric]
        )
        for metric in METRICS
    }

    return {
        "schema_version": 1,
        "comparison": "calibrated_minus_approx",
        "fold_mean_definition": "unweighted arithmetic mean across 1_to_2 and 2_to_1",
        "interpretation": (
            "Legacy raw-axis-angle diagnostic. Episode 1 contains a +/-pi "
            "representation-branch discontinuity, so use the paired rot6d report "
            "for the primary calibration-sensitivity conclusion. Without geometric "
            "ground truth, a positive delta is not proof of better calibration."
        ),
        "classifier_root": str(classifier_root),
        "class_names": class_names,
        "variants": variants,
        "calibrated_minus_approx": {
            "folds": fold_deltas,
            "two_fold_mean_delta": mean_delta,
        },
    }


def _format_float(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_json(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "row_type",
        "variant",
        "fold",
        "class_name",
        "support",
        "folds_with_support",
        *METRICS,
        "class_accuracy_mean",
        "class_accuracy_support_weighted",
    ]
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_report = report["variants"][variant]
        for fold in FOLDS:
            fold_report = variant_report["folds"][fold]
            rows.append({
                "row_type": "overall_fold",
                "variant": variant,
                "fold": fold,
                **{metric: fold_report[metric] for metric in METRICS},
            })
            for class_name, metrics in fold_report.get("per_class", {}).items():
                rows.append({
                    "row_type": "per_class_fold",
                    "variant": variant,
                    "fold": fold,
                    "class_name": class_name,
                    "support": metrics["support"],
                    "class_accuracy_mean": metrics["accuracy"],
                })
        rows.append({
            "row_type": "overall_two_fold_mean",
            "variant": variant,
            "fold": "mean",
            **variant_report["two_fold_mean"],
        })
        for class_name, metrics in variant_report.get("per_class_two_fold", {}).items():
            rows.append({
                "row_type": "per_class_two_fold",
                "variant": variant,
                "fold": "mean",
                "class_name": class_name,
                "support": metrics["total_support"],
                "folds_with_support": metrics["folds_with_support"],
                "class_accuracy_mean": metrics["accuracy_mean_across_supported_folds"],
                "class_accuracy_support_weighted": metrics["accuracy_support_weighted"],
            })

    comparison = report["calibrated_minus_approx"]
    for fold in FOLDS:
        rows.append({
            "row_type": "overall_delta_fold",
            "variant": "calibrated_minus_approx",
            "fold": fold,
            **comparison["folds"][fold],
        })
    rows.append({
        "row_type": "overall_two_fold_mean_delta",
        "variant": "calibrated_minus_approx",
        "fold": "mean",
        **comparison["two_fold_mean_delta"],
    })

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: _format_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            })


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 08-05 calibration classifier A/B summary",
        "",
        "The two-fold mean is the unweighted arithmetic mean of folds `1_to_2` "
        "and `2_to_1`.",
        "",
        "> This reports classifier sensitivity to the calibration branch. Without "
        "geometric ground truth, a positive classifier delta does not by itself prove "
        "that calibration is more accurate.",
        "",
        "> This is the legacy raw-axis-angle diagnostic. Episode 1 crosses the +/-pi "
        "axis-angle branch at three token centres; use the paired rot6d report as the "
        "primary robustness result.",
        "",
        "## Overall metrics",
        "",
        "| Variant | Fold | Validation accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        variant_report = report["variants"][variant]
        for fold in FOLDS:
            metrics = variant_report["folds"][fold]
            lines.append(
                f"| {variant} | {fold} | {_percent(metrics['validation_accuracy'])} "
                f"| {_percent(metrics['validation_f1_macro'])} "
                f"| {_percent(metrics['validation_f1_weighted'])} |"
            )
        metrics = variant_report["two_fold_mean"]
        lines.append(
            f"| **{variant}** | **two-fold mean** | "
            f"**{_percent(metrics['validation_accuracy'])}** | "
            f"**{_percent(metrics['validation_f1_macro'])}** | "
            f"**{_percent(metrics['validation_f1_weighted'])}** |"
        )

    lines.extend([
        "",
        "## Calibrated minus approximate",
        "",
        "| Fold | Validation accuracy delta | Macro F1 delta | Weighted F1 delta |",
        "|---|---:|---:|---:|",
    ])
    comparison = report["calibrated_minus_approx"]
    for fold in FOLDS:
        metrics = comparison["folds"][fold]
        lines.append(
            f"| {fold} | {_percent(metrics['validation_accuracy'], signed=True)} "
            f"| {_percent(metrics['validation_f1_macro'], signed=True)} "
            f"| {_percent(metrics['validation_f1_weighted'], signed=True)} |"
        )
    metrics = comparison["two_fold_mean_delta"]
    lines.append(
        f"| **two-fold mean** | **{_percent(metrics['validation_accuracy'], signed=True)}** "
        f"| **{_percent(metrics['validation_f1_macro'], signed=True)}** "
        f"| **{_percent(metrics['validation_f1_weighted'], signed=True)}** |"
    )

    if report["class_names"]:
        lines.extend([
            "",
            "## Per-class two-fold accuracy",
            "",
            "The fold mean excludes folds where the class has zero validation support; "
            "the weighted value pools fold accuracies by support.",
            "",
            "| Variant | Class | Total support | Supported folds | Fold mean | Support weighted |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for variant in VARIANTS:
            per_class = report["variants"][variant]["per_class_two_fold"]
            for class_name in report["class_names"]:
                metrics = per_class[class_name]
                lines.append(
                    f"| {variant} | {class_name} | {metrics['total_support']} "
                    f"| {metrics['folds_with_support']} "
                    f"| {_percent(metrics['accuracy_mean_across_supported_folds'])} "
                    f"| {_percent(metrics['accuracy_support_weighted'])} |"
                )

    lines.extend(["", "## Inputs", ""])
    for variant in VARIANTS:
        for fold in FOLDS:
            source = report["variants"][variant]["folds"][fold]["source"]
            lines.append(f"- `{variant}` `{fold}`: `{source}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    report: dict[str, Any], output_dir: Path, prefix: str = DEFAULT_PREFIX
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / f"{prefix}.json",
        "csv": output_dir / f"{prefix}.csv",
        "markdown": output_dir / f"{prefix}.md",
    }
    write_json(report, paths["json"])
    write_csv(report, paths["csv"])
    write_markdown(report, paths["markdown"])
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize four 08-05 calibration A/B classifier experiments."
    )
    parser.add_argument(
        "--classifier-root",
        type=Path,
        default=DEFAULT_CLASSIFIER_ROOT,
        help=(
            "Directory containing 0805_approx and 0805_calibrated "
            f"(default: {DEFAULT_CLASSIFIER_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Output filename prefix (default: {DEFAULT_PREFIX})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prefix or Path(args.prefix).name != args.prefix:
        parser.error("--prefix must be a non-empty filename stem, not a path")
    try:
        report = build_summary(args.classifier_root)
        paths = write_reports(report, args.output_dir, args.prefix)
    except SummaryInputError as error:
        parser.error(str(error))
    except OSError as error:
        parser.error(f"cannot write reports under {args.output_dir}: {error}")
    for output_type, path in paths.items():
        print(f"{output_type}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
