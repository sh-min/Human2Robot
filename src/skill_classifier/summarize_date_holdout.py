"""Pair global/object results for leave-one-date-out validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    root = Path(suite["output_dir"])
    rows = []
    for experiment in suite["experiments"]:
        path = root / experiment["id"] / "evaluation_summary.json"
        summary = json.loads(path.read_text())
        rows.append({
            "experiment": experiment["id"],
            "holdout": experiment["group"].removeprefix("holdout_"),
            "branch": "object" if experiment["id"].startswith("object") else "global",
            "train_recordings": summary["train_recordings"],
            "validation_recordings": summary["validation_recordings"],
            "validation_samples": summary["validation_samples"],
            "accuracy": summary["validation_accuracy"],
            "macro_f1": summary["validation_f1_macro"],
            "best_epoch": summary["best_epoch"],
            "per_class": summary["per_class"],
            "summary_path": str(path.resolve()),
        })
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "date_holdout_results.csv").open("w", newline="") as stream:
        csv_rows = [
            {
                **{key: value for key, value in row.items() if key != "per_class"},
                **{
                    f"recall_{name}": metric["accuracy"]
                    for name, metric in row["per_class"].items()
                },
            }
            for row in rows
        ]
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (output / "date_holdout_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    dates = sorted({row["holdout"] for row in rows})
    branches = ("global", "object")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(dates))
    width = 0.36
    for axis, metric, title in zip(
        axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")
    ):
        for offset, branch in enumerate(branches):
            values = [
                next(row[metric] for row in rows if row["holdout"] == date and row["branch"] == branch)
                for date in dates
            ]
            axis.bar(x + (offset - 0.5) * width, values, width, label=branch)
        axis.set_xticks(x, dates)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_xlabel("Held-out recording date")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Validation score")
    axes[1].legend()
    fig.suptitle("Leave-one-date-out generalization (W=4, 300 epochs, seed 42)")
    fig.tight_layout()
    fig.savefig(output / "date_holdout_comparison.png", dpi=180)
    plt.close(fig)

    class_names = list(rows[0]["per_class"])
    fig, axes = plt.subplots(1, len(dates), figsize=(6 * len(dates), 6), sharey=True)
    for axis, date in zip(np.atleast_1d(axes), dates):
        global_row = next(row for row in rows if row["holdout"] == date and row["branch"] == "global")
        object_row = next(row for row in rows if row["holdout"] == date and row["branch"] == "object")
        delta = [
            100.0 * (
                object_row["per_class"][name]["accuracy"]
                - global_row["per_class"][name]["accuracy"]
            )
            for name in class_names
        ]
        color = ["#2ca02c" if value >= 0 else "#d62728" for value in delta]
        axis.barh(class_names, delta, color=color)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(f"Holdout {date}")
        axis.set_xlabel("Object − global recall (percentage points)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Per-class effect of object grounding across unseen dates")
    fig.tight_layout()
    fig.savefig(output / "date_holdout_per_class_delta.png", dpi=180)
    plt.close(fig)

    paired = []
    for date in dates:
        global_row = next(row for row in rows if row["holdout"] == date and row["branch"] == "global")
        object_row = next(row for row in rows if row["holdout"] == date and row["branch"] == "object")
        paired.append({
            "holdout": date,
            "global_accuracy": global_row["accuracy"],
            "object_accuracy": object_row["accuracy"],
            "accuracy_delta": object_row["accuracy"] - global_row["accuracy"],
            "global_macro_f1": global_row["macro_f1"],
            "object_macro_f1": object_row["macro_f1"],
            "macro_f1_delta": object_row["macro_f1"] - global_row["macro_f1"],
        })
    aggregate = {
        branch: {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows if row["branch"] == branch])),
                "std": float(np.std([row[metric] for row in rows if row["branch"] == branch])),
            }
            for metric in ("accuracy", "macro_f1")
        }
        for branch in branches
    }
    for branch in branches:
        branch_rows = [row for row in rows if row["branch"] == branch]
        total_samples = sum(row["validation_samples"] for row in branch_rows)
        aggregate[branch]["accuracy"]["sample_weighted_mean"] = float(
            sum(row["accuracy"] * row["validation_samples"] for row in branch_rows)
            / total_samples
        )
    per_class_aggregate = {}
    for name in class_names:
        per_class_aggregate[name] = {}
        for branch in branches:
            branch_rows = [row for row in rows if row["branch"] == branch]
            support = sum(row["per_class"][name]["support"] for row in branch_rows)
            correct = sum(
                row["per_class"][name]["support"]
                * row["per_class"][name]["accuracy"]
                for row in branch_rows
            )
            per_class_aggregate[name][branch] = float(correct / support)
        per_class_aggregate[name]["delta"] = (
            per_class_aggregate[name]["object"]
            - per_class_aggregate[name]["global"]
        )
    (output / "date_holdout_paired.json").write_text(
        json.dumps({
            "paired": paired,
            "aggregate": aggregate,
            "sample_weighted_per_class": per_class_aggregate,
        }, indent=2) + "\n"
    )
    readme = [
        "# Leave-one-date-out comparison",
        "",
        "Each date is held out in full; its original recordings are never used for training.",
        "All runs use W=4, 300 epochs, seed 42 and select the best validation checkpoint.",
        "",
        "| Held out | Global accuracy | Object accuracy | Delta | Global macro-F1 | Object macro-F1 | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        readme.append(
            f"| {row['holdout']} | {row['global_accuracy']:.3f} | {row['object_accuracy']:.3f} | "
            f"{row['accuracy_delta']:+.3f} | {row['global_macro_f1']:.3f} | "
            f"{row['object_macro_f1']:.3f} | {row['macro_f1_delta']:+.3f} |"
        )
    readme.extend([
        "",
        "## Mean over held-out dates",
        "",
        f"- Global: accuracy {aggregate['global']['accuracy']['mean']:.3f}, macro-F1 {aggregate['global']['macro_f1']['mean']:.3f}",
        f"- Object: accuracy {aggregate['object']['accuracy']['mean']:.3f}, macro-F1 {aggregate['object']['macro_f1']['mean']:.3f}",
        f"- Sample-weighted accuracy: global {aggregate['global']['accuracy']['sample_weighted_mean']:.3f}, object {aggregate['object']['accuracy']['sample_weighted_mean']:.3f}",
        "",
        "## Sample-weighted recall by class",
        "",
        "| Class | Global | Object | Delta |",
        "|---|---:|---:|---:|",
    ])
    for name in class_names:
        metric = per_class_aggregate[name]
        readme.append(
            f"| {name} | {metric['global']:.3f} | {metric['object']:.3f} | {metric['delta']:+.3f} |"
        )
    readme.extend([
        "",
        "The per-class delta plot is important: an average gain can hide regressions in individual actions.",
    ])
    (output / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps(rows, indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
