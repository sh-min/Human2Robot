"""Compare the matched V-JEPA baseline with object-semantic fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")
COLORS = ("#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2")


def load_summary(path: Path) -> dict:
    value = json.loads(path.read_text())
    if list(value["per_class"]) != list(LABELS):
        raise ValueError(f"unexpected class order in {path}")
    return value


def validate_contract(baseline: dict, semantic: dict) -> None:
    for key in (
        "seed",
        "deterministic",
        "train_recordings",
        "validation_recordings",
        "train_samples",
        "validation_samples",
    ):
        if baseline[key] != semantic[key]:
            raise ValueError(f"comparison contract differs for {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_summary(args.baseline)
    semantic = load_summary(args.semantic)
    validate_contract(baseline, semantic)
    base_recall = np.asarray(
        [baseline["per_class"][name]["accuracy"] for name in LABELS]
    )
    semantic_recall = np.asarray(
        [semantic["per_class"][name]["accuracy"] for name in LABELS]
    )
    rows = [
        {
            "metric": "accuracy",
            "baseline": baseline["validation_accuracy"],
            "object_semantic": semantic["validation_accuracy"],
            "delta_percentage_points": 100
            * (semantic["validation_accuracy"] - baseline["validation_accuracy"]),
        },
        {
            "metric": "macro_f1",
            "baseline": baseline["validation_f1_macro"],
            "object_semantic": semantic["validation_f1_macro"],
            "delta_percentage_points": 100
            * (semantic["validation_f1_macro"] - baseline["validation_f1_macro"]),
        },
    ]
    for name, first, second in zip(
        LABELS, base_recall, semantic_recall, strict=True
    ):
        rows.append(
            {
                "metric": f"{name}_recall",
                "baseline": float(first),
                "object_semantic": float(second),
                "delta_percentage_points": float(100 * (second - first)),
            }
        )
    csv_path = args.output_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "comparison_contract": {
            "same_seed": baseline["seed"],
            "same_train_recordings": baseline["train_recordings"],
            "same_validation_recordings": baseline["validation_recordings"],
            "same_train_samples": baseline["train_samples"],
            "same_validation_samples": baseline["validation_samples"],
            "same_window_size": 8,
            "same_dropout": 0.4,
        },
        "baseline": baseline,
        "object_semantic": semantic,
        "deltas_percentage_points": {
            row["metric"]: row["delta_percentage_points"] for row in rows
        },
        "interpretation": (
            "Single deterministic seed: promising evidence, not a confidence interval. "
            "Run additional scene-disjoint seeds before claiming a robust gain."
        ),
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    aggregate_names = ("Accuracy", "Macro F1")
    x = np.arange(2)
    width = 0.34
    baseline_aggregate = (
        baseline["validation_accuracy"],
        baseline["validation_f1_macro"],
    )
    semantic_aggregate = (
        semantic["validation_accuracy"],
        semantic["validation_f1_macro"],
    )
    axes[0, 0].bar(
        x - width / 2, baseline_aggregate, width, label="V-JEPA attention", color="#9AA6B2"
    )
    axes[0, 0].bar(
        x + width / 2, semantic_aggregate, width, label="VLM + SAM2 + V-JEPA", color="#1976D2"
    )
    axes[0, 0].set_xticks(x, aggregate_names)
    axes[0, 0].set_ylim(0.7, 0.93)
    axes[0, 0].set_title("Held-out validation metrics")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.2)

    positions = np.arange(len(LABELS))
    axes[0, 1].bar(
        positions - width / 2,
        base_recall,
        width,
        label="V-JEPA attention",
        color="#9AA6B2",
    )
    axes[0, 1].bar(
        positions + width / 2,
        semantic_recall,
        width,
        label="VLM + SAM2 + V-JEPA",
        color=COLORS,
    )
    axes[0, 1].set_xticks(positions, LABELS, rotation=25)
    axes[0, 1].set_ylim(0.55, 1.0)
    axes[0, 1].set_title("Per-class recall")
    axes[0, 1].grid(axis="y", alpha=0.2)

    deltas = 100 * (semantic_recall - base_recall)
    axes[1, 0].barh(
        LABELS,
        deltas,
        color=["#2E8B57" if value >= 0 else "#D94A4A" for value in deltas],
    )
    axes[1, 0].axvline(0, color="#263238", linewidth=1)
    axes[1, 0].set_xlabel("Change in recall (percentage points)")
    axes[1, 0].set_title("Effect of object-semantic context")
    for y, value in enumerate(deltas):
        axes[1, 0].text(
            value + (0.15 if value >= 0 else -0.15),
            y,
            f"{value:+.2f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontweight="bold",
        )

    axes[1, 1].axis("off")
    lines = [
        "Pipeline",
        "RGB video → Grounding DINO Base (fixed object bank)",
        "→ SAM2 masks → V-JEPA 2.1 patch pooling",
        "→ 8-token temporal context → 6 action classes",
        "",
        "No label leakage",
        "Every clip receives all 7 object queries.",
        "The ground-truth action sentence is never selected as input.",
        "",
        "Matched comparison",
        "72 train views / 9 validation scenes / seed 42 / dropout 0.4",
        "Choco excluded; Color Jitter is training-only.",
        "",
        "Caution",
        "One deterministic seed is evidence for a pilot, not a robust claim.",
    ]
    axes[1, 1].text(
        0.02, 0.98, "\n".join(lines), va="top", fontsize=11.5, linespacing=1.45
    )
    figure.suptitle(
        "Kitchen skill classification: object-semantic V-JEPA pilot",
        fontsize=17,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    dashboard = args.output_dir / "comparison_dashboard.png"
    figure.savefig(dashboard, dpi=190, facecolor="white")
    figure.savefig(args.output_dir / "comparison_dashboard.pdf", facecolor="white")
    plt.close(figure)
    print(dashboard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
