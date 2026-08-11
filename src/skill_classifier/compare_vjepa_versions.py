"""Build a compact V-JEPA 2 versus 2.1 classification/clustering report."""

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


VERSION_NAMES = ("V-JEPA 2 ViT-L/256", "V-JEPA 2.1 ViT-L/384")
GLOBAL_METRICS = (
    ("validation_accuracy", "Accuracy"),
    ("validation_f1_macro", "Macro F1"),
    ("validation_f1_weighted", "Weighted F1"),
)
CLUSTER_METRICS = (
    ("label_silhouette", "Label silhouette"),
    ("kmeans_silhouette", "K-Means silhouette"),
    ("adjusted_rand_index", "ARI"),
    ("normalized_mutual_information", "NMI"),
    ("cluster_purity", "Purity"),
)


def load_reports(experiment_dir: Path) -> tuple[dict, dict]:
    evaluation_path = experiment_dir / "evaluation_summary.json"
    clustering_path = experiment_dir / "clustering" / "clustering_summary.json"
    if not evaluation_path.is_file() or not clustering_path.is_file():
        raise FileNotFoundError(
            f"missing evaluation/clustering report under {experiment_dir}"
        )
    return (
        json.loads(evaluation_path.read_text()),
        json.loads(clustering_path.read_text()),
    )


def grouped_bars(ax, labels, first, second, title):
    x = np.arange(len(labels))
    width = 0.36
    bars = (
        ax.bar(x - width / 2, first, width, label=VERSION_NAMES[0]),
        ax.bar(x + width / 2, second, width, label=VERSION_NAMES[1]),
    )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for group in bars:
        ax.bar_label(group, fmt="%.3f", padding=2, fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vjepa2_dir", required=True, type=Path)
    parser.add_argument("--vjepa21_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    experiment_dirs = (args.vjepa2_dir.resolve(), args.vjepa21_dir.resolve())
    reports = tuple(load_reports(path) for path in experiment_dirs)
    evaluations = tuple(report[0] for report in reports)
    clusterings = tuple(report[1] for report in reports)

    contract = (
        "train_recordings",
        "validation_recordings",
        "train_samples",
        "validation_samples",
    )
    for key in contract:
        if evaluations[0][key] != evaluations[1][key]:
            raise ValueError(f"comparison contract differs for {key}")
    class_names = list(evaluations[0]["per_class"])
    if class_names != list(evaluations[1]["per_class"]):
        raise ValueError("comparison class order differs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "comparison_contract": {
            key: evaluations[0][key] for key in contract
        },
        "versions": {},
        "delta_vjepa21_minus_vjepa2": {},
    }
    for name, directory, evaluation, clustering in zip(
        VERSION_NAMES, experiment_dirs, evaluations, clusterings
    ):
        summary["versions"][name] = {
            "experiment_dir": str(directory),
            "best_epoch": evaluation["best_epoch"],
            "classification": {
                label: float(evaluation[key])
                for key, label in GLOBAL_METRICS
            },
            "per_class_recall": {
                label: float(values["accuracy"])
                for label, values in evaluation["per_class"].items()
            },
            "raw_clustering": {
                label: float(clustering["raw_metrics"][key])
                for key, label in CLUSTER_METRICS
            },
            "adapted_clustering": {
                label: float(clustering["adapted_metrics"][key])
                for key, label in CLUSTER_METRICS
            },
        }
    first = summary["versions"][VERSION_NAMES[0]]
    second = summary["versions"][VERSION_NAMES[1]]
    for section in (
        "classification",
        "per_class_recall",
        "raw_clustering",
        "adapted_clustering",
    ):
        summary["delta_vjepa21_minus_vjepa2"][section] = {
            key: second[section][key] - first[section][key]
            for key in first[section]
        }

    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    csv_path = args.output_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("section", "metric", *VERSION_NAMES, "delta_2.1_minus_2"))
        for section in (
            "classification",
            "per_class_recall",
            "raw_clustering",
            "adapted_clustering",
        ):
            for metric, value in first[section].items():
                writer.writerow(
                    (section, metric, value, second[section][metric], second[section][metric] - value)
                )

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    global_labels = [label for _, label in GLOBAL_METRICS]
    grouped_bars(
        axes[0, 0],
        global_labels,
        [first["classification"][label] for label in global_labels],
        [second["classification"][label] for label in global_labels],
        "Held-out classification (same 9 recordings)",
    )
    grouped_bars(
        axes[0, 1],
        class_names,
        [first["per_class_recall"][label] for label in class_names],
        [second["per_class_recall"][label] for label in class_names],
        "Per-class recall",
    )
    cluster_labels = [label for _, label in CLUSTER_METRICS]
    grouped_bars(
        axes[1, 0],
        cluster_labels,
        [first["raw_clustering"][label] for label in cluster_labels],
        [second["raw_clustering"][label] for label in cluster_labels],
        "Frozen backbone embeddings",
    )
    grouped_bars(
        axes[1, 1],
        cluster_labels,
        [first["adapted_clustering"][label] for label in cluster_labels],
        [second["adapted_clustering"][label] for label in cluster_labels],
        "Classifier-adapted embeddings",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
    )
    fig.suptitle(
        "Robot-overlay skill features: V-JEPA 2 vs V-JEPA 2.1\n"
        "45 recordings, identical split / labels / 4 FPS sampling / MLP",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    dashboard_path = args.output_dir / "comparison_dashboard.png"
    fig.savefig(dashboard_path, dpi=180)
    plt.close(fig)

    report_links = []
    for name, directory in zip(VERSION_NAMES, experiment_dirs):
        clustering_index = directory / "clustering" / "index.html"
        relative = os.path.relpath(clustering_index, args.output_dir)
        report_links.append(
            f'<li><a href="{html.escape(relative)}">{html.escape(name)} clustering report</a></li>'
        )
    index_path = args.output_dir / "index.html"
    index_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>V-JEPA comparison</title>"
        "<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto;}"
        "img{width:100%;height:auto;}code{background:#eee;padding:2px 5px}</style>"
        "<h1>V-JEPA 2 vs V-JEPA 2.1</h1>"
        "<p>Same 45 robot-overlay recordings, recording-level split, labels, "
        "4 FPS sampling, 8-token window, MLP, and seed 42.</p>"
        "<img src='comparison_dashboard.png' alt='comparison dashboard'>"
        "<h2>Artifacts</h2><ul>"
        "<li><a href='comparison_summary.json'>JSON summary</a></li>"
        "<li><a href='comparison_metrics.csv'>CSV metrics</a></li>"
        + "".join(report_links)
        + "</ul>"
    )
    print(f"[ok] dashboard: {dashboard_path.resolve()}")
    print(f"[ok] report:    {index_path.resolve()}")


if __name__ == "__main__":
    main()
