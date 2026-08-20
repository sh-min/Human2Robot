"""Summarize full-budget seed robustness for global and object models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


EXISTING = (
    (
        "object_w8", 42,
        "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco/seed_42/evaluation_summary.json",
    ),
    (
        "global_w8", 42,
        "output/skill_classifier/kitchen_0724_0728_human_vjepa21_spatial_attention_color_jitter_no_choco/dropout_04_seed_42/evaluation_summary.json",
    ),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    root = Path(suite["output_dir"])
    records = []
    for group, seed, summary_path in EXISTING:
        summary = json.loads(Path(summary_path).read_text())
        records.append((group, seed, summary, str(Path(summary_path).resolve())))
    for experiment in suite["experiments"]:
        summary_path = root / experiment["id"] / "evaluation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        records.append((experiment["group"], summary["seed"], summary, str(summary_path.resolve())))

    grouped = defaultdict(list)
    per_seed = []
    for group, seed, summary, path in records:
        row = {
            "group": group,
            "seed": seed,
            "accuracy": summary["validation_accuracy"],
            "macro_f1": summary["validation_f1_macro"],
            "weighted_f1": summary["validation_f1_weighted"],
            "best_epoch": summary["best_epoch"],
            "summary_path": path,
        }
        per_seed.append(row)
        grouped[group].append(row)
    aggregates = []
    for group in sorted(grouped):
        rows = grouped[group]
        aggregates.append({
            "group": group,
            "n": len(rows),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in rows], ddof=1)),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
            "macro_f1_std": float(np.std([row["macro_f1"] for row in rows], ddof=1)),
        })

    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_seed", per_seed), ("aggregate", aggregates)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / f"{name}.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )

    groups = [row["group"] for row in aggregates]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.36
    ax.bar(
        x - width / 2,
        [row["accuracy_mean"] for row in aggregates],
        width,
        yerr=[row["accuracy_std"] for row in aggregates],
        capsize=5,
        label="Accuracy",
    )
    ax.bar(
        x + width / 2,
        [row["macro_f1_mean"] for row in aggregates],
        width,
        yerr=[row["macro_f1_std"] for row in aggregates],
        capsize=5,
        label="Macro-F1",
    )
    ax.set_xticks(x, groups)
    ax.set_ylim(0.65, 1.0)
    ax.set_ylabel("Mean over 3 deterministic seeds")
    ax.set_title("300-epoch robustness: global vs object context, W=4 vs W=8")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "multiseed_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(aggregates, indent=2))
    print(output.resolve())


if __name__ == "__main__":
    main()
