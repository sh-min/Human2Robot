"""Create a compact performance dashboard from a skill-classifier experiment.

Usage:
    python -m skill_classifier.visualize_training_report \
        --experiment_dir output/skill_classifier/my_experiment
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_name", default="performance_dashboard.png")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    summary = json.loads((exp_dir / "evaluation_summary.json").read_text())
    with (exp_dir / "training_history.csv").open(newline="") as f:
        history = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]

    epochs = [row["epoch"] for row in history]
    best_epoch = summary["best_epoch"]
    best_row = min(history, key=lambda row: abs(row["epoch"] - best_epoch))

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.8, 1.2))
    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_class = fig.add_subplot(grid[0, 1])
    ax_acc = fig.add_subplot(grid[1, 0])
    ax_f1 = fig.add_subplot(grid[1, 1])

    ax_metrics.axis("off")
    metrics_text = (
        f"Best epoch     {best_epoch}\n"
        f"Validation accuracy   {summary['validation_accuracy']:.1%}\n"
        f"Macro F1              {summary['validation_f1_macro']:.1%}\n"
        f"Weighted F1           {summary['validation_f1_weighted']:.1%}\n\n"
        f"Train scenes / samples       "
        f"{summary['train_recordings']} / {summary['train_samples']:,}\n"
        f"Validation scenes / samples  "
        f"{summary['validation_recordings']} / {summary['validation_samples']:,}\n"
        f"Input variant          {summary['variant']}"
    )
    ax_metrics.text(
        0.04, 0.94, metrics_text, va="top", family="monospace", fontsize=15,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f2f6fa", "edgecolor": "#5b7894"},
    )
    ax_metrics.set_title("Held-out scene evaluation", fontsize=15, fontweight="bold")

    class_names = list(summary["per_class"])
    class_acc = [
        summary["per_class"][name]["accuracy"] or 0.0 for name in class_names
    ]
    colors = ["#2b8cbe" if value >= 0.8 else "#fdae6b" if value >= 0.6 else "#e34a33"
              for value in class_acc]
    bars = ax_class.barh(class_names, class_acc, color=colors)
    ax_class.set_xlim(0, 1)
    ax_class.set_xlabel("Recall / per-class accuracy")
    ax_class.set_title("Class performance", fontsize=15, fontweight="bold")
    ax_class.grid(axis="x", alpha=0.25)
    ax_class.invert_yaxis()
    for bar, value in zip(bars, class_acc):
        ax_class.text(
            min(value + 0.015, 0.94), bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}", va="center", fontweight="bold",
        )

    ax_acc.plot(epochs, [row["train_acc"] for row in history], label="Train", alpha=0.9)
    ax_acc.plot(epochs, [row["val_acc"] for row in history], label="Validation", alpha=0.9)
    ax_acc.axvline(best_epoch, color="#d62728", linestyle="--", alpha=0.7,
                   label=f"Selected epoch {best_epoch}")
    ax_acc.scatter([best_epoch], [best_row["val_acc"]], color="#d62728", zorder=3)
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Learning curve", fontsize=15, fontweight="bold")
    ax_acc.grid(alpha=0.25)
    ax_acc.legend()

    ax_f1.plot(
        epochs, [row["val_f1_macro"] for row in history],
        label="Macro F1", color="#756bb1",
    )
    ax_f1.plot(
        epochs, [row["val_f1_weighted"] for row in history],
        label="Weighted F1", color="#31a354",
    )
    ax_f1.axvline(best_epoch, color="#d62728", linestyle="--", alpha=0.7)
    ax_f1.set_ylim(0, 1.02)
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1")
    ax_f1.set_title("Validation F1", fontsize=15, fontweight="bold")
    ax_f1.grid(alpha=0.25)
    ax_f1.legend()

    fig.suptitle(
        "V-JEPA2 robot-video skill classifier — performance report",
        fontsize=20, fontweight="bold",
    )
    fig.text(
        0.5, 0.015,
        "Validation is scene-disjoint from training. High train accuracy with lower "
        "validation accuracy indicates remaining generalization gap.",
        ha="center", fontsize=11, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output_path = exp_dir / args.output_name
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
