"""Compare V-JEPA 2.1 spatial-mean dropout and spatial-attention runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES = (
    "Mean + D0.3",
    "Mean + D0.4",
    "Mean + D0.5",
    "Attn + D0.3",
    "Attn + CJ + D0.3",
    "Attn + CJ + D0.4",
    "Attn + CJ + D0.5",
)
COLORS = (
    "#4c78a8",
    "#72b7b2",
    "#f2cf5b",
    "#e45756",
    "#7a5195",
    "#003f5c",
    "#ffa600",
)
COLOR_JITTER_BASELINE = "Attn + CJ + D0.3"


def load_run(path: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((path / "evaluation_summary.json").read_text())
    with (path / "training_history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    return summary, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean03", required=True, type=Path)
    parser.add_argument("--mean04", required=True, type=Path)
    parser.add_argument("--mean05", required=True, type=Path)
    parser.add_argument("--attention", required=True, type=Path)
    parser.add_argument("--color_jitter", required=True, type=Path)
    parser.add_argument("--color_jitter04", required=True, type=Path)
    parser.add_argument("--color_jitter05", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    run_dirs = tuple(
        path.resolve()
        for path in (
            args.mean03,
            args.mean04,
            args.mean05,
            args.attention,
            args.color_jitter,
            args.color_jitter04,
            args.color_jitter05,
        )
    )
    loaded = tuple(load_run(path) for path in run_dirs)
    summaries = tuple(item[0] for item in loaded)
    histories = tuple(item[1] for item in loaded)
    contract_keys = ("validation_recordings", "validation_samples", "seed")
    for key in contract_keys:
        if len({summary[key] for summary in summaries}) != 1:
            raise ValueError(f"run contract differs for {key}")
    classes = list(summaries[0]["per_class"])
    if any(list(summary["per_class"]) != classes for summary in summaries):
        raise ValueError("run class order differs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        name: {
            "experiment_dir": str(path),
            "best_epoch": int(summary["best_epoch"]),
            "train_recordings": int(summary["train_recordings"]),
            "train_samples": int(summary["train_samples"]),
            "accuracy": float(summary["validation_accuracy"]),
            "macro_f1": float(summary["validation_f1_macro"]),
            "weighted_f1": float(summary["validation_f1_weighted"]),
            "validation_loss": float(summary["validation_loss"]),
            "per_class_recall": {
                label: float(values["accuracy"])
                for label, values in summary["per_class"].items()
            },
        }
        for name, path, summary in zip(NAMES, run_dirs, summaries)
    }
    baseline = metrics[NAMES[0]]
    report = {
        "schema_version": 1,
        "validation_contract": {key: summaries[0][key] for key in contract_keys},
        "runs": metrics,
        "accuracy_delta_vs_mean_dropout_03": {
            name: values["accuracy"] - baseline["accuracy"]
            for name, values in metrics.items()
        },
        "accuracy_delta_vs_attention_color_jitter_dropout_03": {
            name: values["accuracy"] - metrics[COLOR_JITTER_BASELINE]["accuracy"]
            for name, values in metrics.items()
        },
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    global_names = ("Accuracy", "Macro F1", "Weighted F1")
    global_keys = ("accuracy", "macro_f1", "weighted_f1")
    x = np.arange(len(global_names))
    width = min(0.15, 0.78 / len(NAMES))
    for index, (name, color) in enumerate(zip(NAMES, COLORS)):
        bars = axes[0, 0].bar(
            x + (index - (len(NAMES) - 1) / 2) * width,
            [metrics[name][key] for key in global_keys],
            width,
            label=name,
            color=color,
        )
        axes[0, 0].bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axes[0, 0].set_xticks(x, global_names)
    axes[0, 0].set_ylim(0, 0.85)
    axes[0, 0].set_title("Held-out classification")
    axes[0, 0].grid(axis="y", alpha=0.25)

    x = np.arange(len(classes))
    for index, (name, color) in enumerate(zip(NAMES, COLORS)):
        axes[0, 1].bar(
            x + (index - (len(NAMES) - 1) / 2) * width,
            [metrics[name]["per_class_recall"][label] for label in classes],
            width,
            label=name,
            color=color,
        )
    axes[0, 1].set_xticks(x, classes, rotation=20, ha="right")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Per-class recall")
    axes[0, 1].grid(axis="y", alpha=0.25)

    for name, color, history in zip(NAMES, COLORS, histories):
        axes[1, 0].plot(
            [int(row["epoch"]) for row in history],
            [float(row["val_acc"]) for row in history],
            label=name,
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )
    axes[1, 0].set_ylim(0.3, 0.85)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Validation accuracy")
    axes[1, 0].set_title("Validation trajectory")
    axes[1, 0].grid(alpha=0.25)

    deltas = [
        report["accuracy_delta_vs_attention_color_jitter_dropout_03"][name] * 100
        for name in NAMES
    ]
    bars = axes[1, 1].bar(NAMES, deltas, color=COLORS)
    axes[1, 1].axhline(0, color="#333", linewidth=0.8)
    axes[1, 1].bar_label(bars, fmt="%+.2f pp", padding=3)
    axes[1, 1].set_ylabel("Accuracy difference (percentage points)")
    axes[1, 1].set_title("Difference from Attention + Color Jitter + Dropout 0.3")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(axis="y", alpha=0.25)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle(
        "V-JEPA 2.1: dropout, learned spatial attention, and Color Jitter\n"
        "Same 45 source recordings and held-out split; Color Jitter augments train only",
        y=0.995,
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    dashboard = args.output_dir / "comparison_dashboard.png"
    fig.savefig(dashboard, dpi=180)
    plt.close(fig)

    links = []
    for name, run_dir in zip(NAMES, run_dirs):
        relative = os.path.relpath(run_dir / "confusion_matrix_best.png", args.output_dir)
        links.append(
            f'<li><a href="{html.escape(relative)}">{html.escape(name)} confusion matrix</a></li>'
        )
    (args.output_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>V-JEPA 2.1 pooling comparison</title>"
        "<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;}"
        "img{width:100%;height:auto}</style>"
        "<h1>V-JEPA 2.1 pooling, attention, and Color Jitter comparison</h1>"
        "<img src='comparison_dashboard.png' alt='comparison dashboard'>"
        "<ul><li><a href='comparison_summary.json'>JSON metrics</a></li>"
        + "".join(links)
        + "</ul>"
    )
    print(f"[ok] dashboard: {dashboard.resolve()}")
    print(f"[ok] report: {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
