#!/usr/bin/env python3
"""Summarize the paired, multi-seed 08-05 rot6d calibration experiment.

Expected input layout::

    output/skill_classifier/0805_rot6d_{approx,calibrated}/
        seed_{seed}/fold_{1_to_2,2_to_1}/evaluation_summary.json

All deltas are ``calibrated - approx``.  The two-fold result for one seed is
the unweighted arithmetic mean of the two held-out-fold metrics.  Across-seed
dispersion is the sample standard deviation (``n - 1`` denominator).
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
DEFAULT_PREFIX = "0805_rot6d_calibration_training_summary"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

VARIANTS = ("approx", "calibrated")
FOLDS = ("1_to_2", "2_to_1")
METRICS = (
    "validation_accuracy",
    "validation_f1_macro",
    "validation_f1_weighted",
)
METRIC_TITLES = {
    "validation_accuracy": "Accuracy",
    "validation_f1_macro": "Macro F1",
    "validation_f1_weighted": "Weighted F1",
}
CORRECT_COUNT_ATOL = 1e-5

CAVEATS = (
    "Only two recordings are available: each fold trains on one recording and "
    "validates on the other, so the estimates are small-sample and high-variance.",
    "The best epoch is selected using the same validation fold reported here; "
    "these are selection-biased validation results, not independent test results.",
    "This isolates sensitivity to the MH scalar-focal-length input to HaWoR and "
    "the resulting MANO features only; it does not evaluate full stereo "
    "calibration, rectification, distortion, or extrinsics.",
)


class SummaryInputError(ValueError):
    """Raised when an evaluation summary is absent or malformed."""


def evaluation_path(
    classifier_root: Path, variant: str, seed: int, fold: str
) -> Path:
    return (
        classifier_root
        / f"0805_rot6d_{variant}"
        / f"seed_{seed}"
        / f"fold_{fold}"
        / "evaluation_summary.json"
    )


def _required_probability(payload: dict[str, Any], field: str, path: Path) -> float:
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


def _required_positive_integer(
    payload: dict[str, Any], field: str, path: Path
) -> int:
    if field not in payload:
        raise SummaryInputError(f"{path}: missing required field {field!r}")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SummaryInputError(
            f"{path}: field {field!r} must be a positive integer, got {value!r}"
        )
    return value


def _optional_nonnegative_integer(
    payload: dict[str, Any], field: str, path: Path
) -> int | None:
    if field not in payload:
        return None
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SummaryInputError(
            f"{path}: field {field!r} must be a non-negative integer, got {value!r}"
        )
    return value


def _optional_finite_number(
    payload: dict[str, Any], field: str, path: Path
) -> float | None:
    if field not in payload:
        return None
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryInputError(
            f"{path}: field {field!r} must be a finite number, got {value!r}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise SummaryInputError(
            f"{path}: field {field!r} must be a finite number, got {value!r}"
        )
    return value


def _per_class_metrics(
    payload: dict[str, Any], path: Path
) -> dict[str, dict[str, Any]] | None:
    """Validate the optional per-class payload using the legacy tool contract."""
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
            if support:
                raise SummaryInputError(
                    f"{path}: per_class[{class_name!r}].accuracy is null despite "
                    f"support={support}"
                )
        else:
            accuracy = _required_probability(
                {"accuracy": accuracy},
                "accuracy",
                Path(f"{path}:per_class[{class_name!r}]"),
            )
        parsed[class_name] = {"support": support, "accuracy": accuracy}
    return parsed


def load_evaluation_summary(path: Path) -> dict[str, Any]:
    """Load one evaluation summary with strict numeric and schema validation."""
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
        metric: _required_probability(payload, metric, path) for metric in METRICS
    }
    result["validation_samples"] = _required_positive_integer(
        payload, "validation_samples", path
    )
    result["per_class"] = _per_class_metrics(payload, path)
    if payload.get("hand_representation") != "rot6d":
        raise SummaryInputError(
            f"{path}: hand_representation must be 'rot6d', got "
            f"{payload.get('hand_representation')!r}"
        )
    result["hand_representation"] = "rot6d"
    seed_value = _optional_nonnegative_integer(payload, "seed", path)
    if seed_value is None:
        raise SummaryInputError(f"{path}: missing required field 'seed'")
    result["seed"] = seed_value
    if payload.get("deterministic") is not True:
        raise SummaryInputError(
            f"{path}: deterministic must be true for paired A/B runs"
        )
    result["deterministic"] = True

    for field in (
        "best_epoch",
        "train_recordings",
        "validation_recordings",
        "train_samples",
    ):
        value = _optional_nonnegative_integer(payload, field, path)
        if value is not None:
            result[field] = value
    validation_loss = _optional_finite_number(payload, "validation_loss", path)
    if validation_loss is not None:
        result["validation_loss"] = validation_loss
    for field in ("experiment", "variant"):
        if field in payload:
            value = payload[field]
            if not isinstance(value, str) or not value:
                raise SummaryInputError(
                    f"{path}: field {field!r} must be a non-empty string, "
                    f"got {value!r}"
                )
            result[field] = value
    return result


def _mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(values))


def _correct_count_check(accuracy: float, support: int) -> dict[str, Any]:
    product = accuracy * support
    nearest = int(round(product))
    residual = abs(product - nearest)
    is_near = residual <= CORRECT_COUNT_ATOL
    return {
        "support": support,
        "accuracy_times_support": product,
        "nearest_integer": nearest,
        "absolute_integer_residual": residual,
        "tolerance": CORRECT_COUNT_ATOL,
        "is_near_integer": is_near,
        "inferred_correct_count": nearest if is_near else None,
    }


def _fold_report(evaluation: dict[str, Any], source: Path) -> dict[str, Any]:
    validation_samples = evaluation["validation_samples"]
    report: dict[str, Any] = {
        "source": str(source),
        **{metric: evaluation[metric] for metric in METRICS},
        "validation_samples": validation_samples,
        "accuracy_correct_count_check": _correct_count_check(
            evaluation["validation_accuracy"], validation_samples
        ),
    }
    for field in (
        "experiment",
        "variant",
        "hand_representation",
        "seed",
        "deterministic",
        "best_epoch",
        "train_recordings",
        "validation_recordings",
        "train_samples",
        "validation_loss",
    ):
        if field in evaluation:
            report[field] = evaluation[field]
    if evaluation["per_class"] is not None:
        report["per_class"] = evaluation["per_class"]
        support_total = sum(
            row["support"] for row in evaluation["per_class"].values()
        )
        report["per_class_support_check"] = {
            "support_total": support_total,
            "matches_validation_samples": support_total == validation_samples,
        }
    return report


def build_summary(classifier_root: Path, seeds: Iterable[int]) -> dict[str, Any]:
    classifier_root = classifier_root.expanduser().resolve()
    seed_values = tuple(seeds)
    if not seed_values:
        raise SummaryInputError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_values):
        raise SummaryInputError("seeds must be non-negative integers")
    if len(set(seed_values)) != len(seed_values):
        raise SummaryInputError("seeds must be unique")

    evaluations: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        variant: {} for variant in VARIANTS
    }
    paths: dict[str, dict[int, dict[str, Path]]] = {
        variant: {} for variant in VARIANTS
    }
    for variant in VARIANTS:
        for seed in seed_values:
            evaluations[variant][seed] = {}
            paths[variant][seed] = {}
            for fold in FOLDS:
                path = evaluation_path(classifier_root, variant, seed, fold)
                paths[variant][seed][fold] = path
                try:
                    evaluation = load_evaluation_summary(path)
                    if evaluation["seed"] != seed:
                        raise SummaryInputError(
                            f"{path}: summary seed {evaluation['seed']} != path seed {seed}"
                        )
                    evaluations[variant][seed][fold] = evaluation
                except SummaryInputError as error:
                    raise SummaryInputError(
                        f"variant={variant!r}, seed={seed}, fold={fold!r}: {error}"
                    ) from error

    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        seed_reports: dict[str, Any] = {}
        for seed in seed_values:
            folds = {
                fold: _fold_report(
                    evaluations[variant][seed][fold], paths[variant][seed][fold]
                )
                for fold in FOLDS
            }
            seed_reports[str(seed)] = {
                "folds": folds,
                "two_fold_mean": {
                    metric: _mean(
                        evaluations[variant][seed][fold][metric] for fold in FOLDS
                    )
                    for metric in METRICS
                },
                "validation_samples_total": sum(
                    evaluations[variant][seed][fold]["validation_samples"]
                    for fold in FOLDS
                ),
            }
        variants[variant] = {"seeds": seed_reports}

    paired_seeds: dict[str, Any] = {}
    for seed in seed_values:
        fold_deltas: dict[str, Any] = {}
        for fold in FOLDS:
            approx = evaluations["approx"][seed][fold]
            calibrated = evaluations["calibrated"][seed][fold]
            approx_samples = approx["validation_samples"]
            calibrated_samples = calibrated["validation_samples"]
            fold_deltas[fold] = {
                **{
                    metric: calibrated[metric] - approx[metric]
                    for metric in METRICS
                },
                "validation_samples": {
                    "approx": approx_samples,
                    "calibrated": calibrated_samples,
                    "match": approx_samples == calibrated_samples,
                },
            }

        approx_total = variants["approx"]["seeds"][str(seed)][
            "validation_samples_total"
        ]
        calibrated_total = variants["calibrated"]["seeds"][str(seed)][
            "validation_samples_total"
        ]
        paired_seeds[str(seed)] = {
            "folds": fold_deltas,
            "two_fold_mean_delta": {
                metric: (
                    variants["calibrated"]["seeds"][str(seed)]["two_fold_mean"][metric]
                    - variants["approx"]["seeds"][str(seed)]["two_fold_mean"][metric]
                )
                for metric in METRICS
            },
            "validation_samples_total": {
                "approx": approx_total,
                "calibrated": calibrated_total,
                "match": approx_total == calibrated_total,
            },
        }

    across_seed: dict[str, Any] = {}
    for metric in METRICS:
        values = [
            paired_seeds[str(seed)]["two_fold_mean_delta"][metric]
            for seed in seed_values
        ]
        across_seed[metric] = {
            "seed_count": len(values),
            "mean": _mean(values),
            "sample_std": float(statistics.stdev(values)) if len(values) >= 2 else None,
            "min": min(values),
            "max": max(values),
        }

    all_sample_counts_match = all(
        paired_seeds[str(seed)]["folds"][fold]["validation_samples"]["match"]
        for seed in seed_values
        for fold in FOLDS
    )
    all_accuracy_counts_near_integer = all(
        variants[variant]["seeds"][str(seed)]["folds"][fold][
            "accuracy_correct_count_check"
        ]["is_near_integer"]
        for variant in VARIANTS
        for seed in seed_values
        for fold in FOLDS
    )

    return {
        "schema_version": 1,
        "comparison": "calibrated_minus_approx",
        "representation": "rot6d",
        "seeds": list(seed_values),
        "metrics": list(METRICS),
        "fold_mean_definition": (
            "unweighted arithmetic mean across folds 1_to_2 and 2_to_1"
        ),
        "across_seed_std_definition": "sample standard deviation (n - 1)",
        "correct_count_check_definition": (
            "validation_accuracy * validation_samples is compared with its nearest "
            f"integer using absolute tolerance {CORRECT_COUNT_ATOL:g}"
        ),
        "caveats": list(CAVEATS),
        "classifier_root": str(classifier_root),
        "variants": variants,
        "paired_calibrated_minus_approx": {
            "seeds": paired_seeds,
            "across_seed_two_fold_delta": across_seed,
        },
        "integrity_checks": {
            "all_paired_validation_sample_counts_match": all_sample_counts_match,
            "all_accuracy_correct_counts_near_integer": (
                all_accuracy_counts_near_integer
            ),
        },
    }


def _format_float(value: float | None, digits: int = 8) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_json(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "row_type",
        "variant",
        "seed",
        "fold",
        "statistic",
        *METRICS,
        "validation_samples",
        "approx_validation_samples",
        "calibrated_validation_samples",
        "validation_sample_count_match",
        "accuracy_times_validation_samples",
        "nearest_correct_count",
        "accuracy_integer_residual",
        "accuracy_near_integer",
    ]
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in report["seeds"]:
            seed_report = report["variants"][variant]["seeds"][str(seed)]
            for fold in FOLDS:
                fold_report = seed_report["folds"][fold]
                count = fold_report["accuracy_correct_count_check"]
                rows.append({
                    "row_type": "variant_fold",
                    "variant": variant,
                    "seed": seed,
                    "fold": fold,
                    **{metric: fold_report[metric] for metric in METRICS},
                    "validation_samples": fold_report["validation_samples"],
                    "accuracy_times_validation_samples": count[
                        "accuracy_times_support"
                    ],
                    "nearest_correct_count": count["nearest_integer"],
                    "accuracy_integer_residual": count[
                        "absolute_integer_residual"
                    ],
                    "accuracy_near_integer": count["is_near_integer"],
                })
            rows.append({
                "row_type": "variant_seed_two_fold_mean",
                "variant": variant,
                "seed": seed,
                "fold": "mean",
                **seed_report["two_fold_mean"],
                "validation_samples": seed_report["validation_samples_total"],
            })

    comparison = report["paired_calibrated_minus_approx"]
    for seed in report["seeds"]:
        seed_report = comparison["seeds"][str(seed)]
        for fold in FOLDS:
            fold_report = seed_report["folds"][fold]
            samples = fold_report["validation_samples"]
            rows.append({
                "row_type": "paired_fold_delta",
                "variant": "calibrated_minus_approx",
                "seed": seed,
                "fold": fold,
                **{metric: fold_report[metric] for metric in METRICS},
                "approx_validation_samples": samples["approx"],
                "calibrated_validation_samples": samples["calibrated"],
                "validation_sample_count_match": samples["match"],
            })
        samples = seed_report["validation_samples_total"]
        rows.append({
            "row_type": "paired_seed_two_fold_delta",
            "variant": "calibrated_minus_approx",
            "seed": seed,
            "fold": "mean",
            **seed_report["two_fold_mean_delta"],
            "approx_validation_samples": samples["approx"],
            "calibrated_validation_samples": samples["calibrated"],
            "validation_sample_count_match": samples["match"],
        })

    for statistic in ("mean", "sample_std", "min", "max"):
        rows.append({
            "row_type": "paired_across_seed_two_fold_delta",
            "variant": "calibrated_minus_approx",
            "seed": "all",
            "fold": "mean",
            "statistic": statistic,
            **{
                metric: comparison["across_seed_two_fold_delta"][metric][statistic]
                for metric in METRICS
            },
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
        "# 08-05 robust rot6d calibration classifier A/B",
        "",
        "All deltas below are `calibrated - approx`. The per-seed result is the "
        "unweighted mean of folds `1_to_2` and `2_to_1`; across-seed spread uses "
        "sample standard deviation (`n - 1`).",
        "",
        "## Interpretation limits",
        "",
    ]
    lines.extend(f"> - {caveat}" for caveat in report["caveats"])

    lines.extend([
        "",
        "## Variant two-fold means by seed",
        "",
        "| Variant | Seed | Accuracy | Macro F1 | Weighted F1 | Validation samples |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        for seed in report["seeds"]:
            seed_report = report["variants"][variant]["seeds"][str(seed)]
            metrics = seed_report["two_fold_mean"]
            lines.append(
                f"| {variant} | {seed} | {_percent(metrics['validation_accuracy'])} "
                f"| {_percent(metrics['validation_f1_macro'])} "
                f"| {_percent(metrics['validation_f1_weighted'])} "
                f"| {seed_report['validation_samples_total']} |"
            )

    lines.extend([
        "",
        "## Paired deltas by seed and fold",
        "",
        "| Seed | Fold | Accuracy delta | Macro F1 delta | Weighted F1 delta | "
        "Samples (approx/calibrated) | Match |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    comparison = report["paired_calibrated_minus_approx"]
    for seed in report["seeds"]:
        for fold in FOLDS:
            metrics = comparison["seeds"][str(seed)]["folds"][fold]
            samples = metrics["validation_samples"]
            lines.append(
                f"| {seed} | {fold} "
                f"| {_percent(metrics['validation_accuracy'], signed=True)} "
                f"| {_percent(metrics['validation_f1_macro'], signed=True)} "
                f"| {_percent(metrics['validation_f1_weighted'], signed=True)} "
                f"| {samples['approx']}/{samples['calibrated']} "
                f"| {'yes' if samples['match'] else '**NO**'} |"
            )

    lines.extend([
        "",
        "## Paired two-fold delta by seed",
        "",
        "| Seed | Accuracy delta | Macro F1 delta | Weighted F1 delta |",
        "|---:|---:|---:|---:|",
    ])
    for seed in report["seeds"]:
        metrics = comparison["seeds"][str(seed)]["two_fold_mean_delta"]
        lines.append(
            f"| {seed} | {_percent(metrics['validation_accuracy'], signed=True)} "
            f"| {_percent(metrics['validation_f1_macro'], signed=True)} "
            f"| {_percent(metrics['validation_f1_weighted'], signed=True)} |"
        )

    lines.extend([
        "",
        "## Across-seed paired two-fold delta",
        "",
        "| Metric | Mean | Sample std | Min | Max | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric in METRICS:
        stats = comparison["across_seed_two_fold_delta"][metric]
        lines.append(
            f"| {METRIC_TITLES[metric]} | {_percent(stats['mean'], signed=True)} "
            f"| {_percent(stats['sample_std'])} "
            f"| {_percent(stats['min'], signed=True)} "
            f"| {_percent(stats['max'], signed=True)} | {stats['seed_count']} |"
        )

    lines.extend([
        "",
        "## Accuracy correct-count audit",
        "",
        f"`accuracy × validation_samples` is considered integer-consistent when "
        f"the absolute residual is at most `{CORRECT_COUNT_ATOL:g}`.",
        "",
        "| Variant | Seed | Fold | Samples | Accuracy x samples | Nearest correct | "
        "Residual | Consistent |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for variant in VARIANTS:
        for seed in report["seeds"]:
            for fold in FOLDS:
                fold_report = report["variants"][variant]["seeds"][str(seed)][
                    "folds"
                ][fold]
                count = fold_report["accuracy_correct_count_check"]
                lines.append(
                    f"| {variant} | {seed} | {fold} | {count['support']} "
                    f"| {count['accuracy_times_support']:.8f} "
                    f"| {count['nearest_integer']} "
                    f"| {count['absolute_integer_residual']:.3g} "
                    f"| {'yes' if count['is_near_integer'] else '**NO**'} |"
                )

    checks = report["integrity_checks"]
    lines.extend([
        "",
        "## Integrity summary",
        "",
        f"- Paired validation sample counts match: "
        f"`{str(checks['all_paired_validation_sample_counts_match']).lower()}`",
        f"- All accuracy-derived correct counts are integer-consistent: "
        f"`{str(checks['all_accuracy_correct_counts_near_integer']).lower()}`",
        "",
        f"Classifier root: `{report['classifier_root']}`",
        "",
    ])
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
        description="Summarize paired multi-seed 08-05 rot6d calibration A/B runs."
    )
    parser.add_argument(
        "--classifier-root",
        type=Path,
        default=DEFAULT_CLASSIFIER_ROOT,
        help=f"Classifier output root (default: {DEFAULT_CLASSIFIER_ROOT})",
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        metavar="SEED",
        help="Paired training seeds (default: 42 43 44 45 46)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prefix or Path(args.prefix).name != args.prefix:
        parser.error("--prefix must be a non-empty filename stem, not a path")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds values must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds values must be unique")
    try:
        report = build_summary(args.classifier_root, args.seeds)
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
