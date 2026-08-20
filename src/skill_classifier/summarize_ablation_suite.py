"""Aggregate an ablation manifest into presentation-ready tables and plots."""

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


REFERENCE_RUNS = (
    {
        "id": "global_baseline_300e",
        "group": "reference",
        "hypothesis": "Existing global spatial-attention V-JEPA 2.1 baseline.",
        "summary": "output/skill_classifier/kitchen_0724_0728_human_vjepa21_spatial_attention_color_jitter_no_choco/dropout_04_seed_42/evaluation_summary.json",
        "history": "output/skill_classifier/kitchen_0724_0728_human_vjepa21_spatial_attention_color_jitter_no_choco/dropout_04_seed_42/training_history.csv",
    },
    {
        "id": "full_object_semantics_300e",
        "group": "reference",
        "hypothesis": "Full Grounding-DINO + SAM2 object-semantic fusion.",
        "summary": "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco/seed_42/evaluation_summary.json",
        "history": "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco/seed_42/training_history.csv",
    },
)


def read_best_train_accuracy(history_path: Path, best_epoch: int):
    if not history_path.is_file():
        return None
    with history_path.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["epoch"]) == int(best_epoch):
                return float(row["train_acc"])
    return None


def load_row(exp_id, group, hypothesis, summary_path, history_path):
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    train_acc = read_best_train_accuracy(history_path, summary["best_epoch"])
    row = {
        "experiment": exp_id,
        "group": group,
        "epochs": 300 if exp_id.endswith("300e") else 150,
        "seed": summary["seed"],
        "best_epoch": summary["best_epoch"],
        "accuracy": summary["validation_accuracy"],
        "macro_f1": summary["validation_f1_macro"],
        "weighted_f1": summary["validation_f1_weighted"],
        "train_accuracy_at_best": train_acc,
        "generalization_gap": (
            train_acc - summary["validation_accuracy"]
            if train_acc is not None else None
        ),
        "parameters": summary.get("trainable_parameters"),
        "hypothesis": hypothesis,
        "summary_path": str(summary_path.resolve()),
        "confusion_matrix": str(
            (summary_path.parent / "confusion_matrix_best.png").resolve()
        ),
    }
    row["per_class_recall"] = {
        name: value.get("recall", value.get("accuracy"))
        for name, value in summary["per_class"].items()
    }
    return row


def save_csv(rows, path):
    fields = [key for key in rows[0] if key != "per_class_recall"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def plot_dashboard(rows, path):
    ordered = sorted(rows, key=lambda row: row["macro_f1"])
    names = [row["experiment"] for row in ordered]
    y = np.arange(len(ordered))
    fig = plt.figure(figsize=(21, max(13, 0.52 * len(ordered) + 7)))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 1.0), hspace=0.33, wspace=0.22)

    ax = fig.add_subplot(grid[0, 0])
    ax.barh(y - 0.18, [row["accuracy"] for row in ordered], height=0.34, label="Accuracy")
    ax.barh(y + 0.18, [row["macro_f1"] for row in ordered], height=0.34, label="Macro-F1")
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_title("Validation metrics (best checkpoint)")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()

    ax = fig.add_subplot(grid[0, 1])
    colors = ["#d62728" if row["group"] == "reference" else "#1f77b4" for row in ordered]
    ax.scatter(
        [row["accuracy"] for row in ordered],
        [row["macro_f1"] for row in ordered],
        c=colors,
        s=55,
    )
    for row in ordered:
        ax.annotate(row["experiment"], (row["accuracy"], row["macro_f1"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Balanced performance vs overall performance")
    ax.grid(alpha=0.2)

    labels = list(next(iter(rows))["per_class_recall"])
    heat_order = sorted(rows, key=lambda row: row["macro_f1"], reverse=True)
    heat = np.asarray([
        [row["per_class_recall"].get(label, np.nan) for label in labels]
        for row in heat_order
    ])
    ax = fig.add_subplot(grid[1, :])
    image = ax.imshow(heat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(heat_order)), [row["experiment"] for row in heat_order], fontsize=8)
    ax.set_title("Per-class recall")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if heat[i, j] > 0.62 else "black")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    fig.suptitle("V-JEPA 2.1 object-semantic ablation study", fontsize=18)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(rows, path):
    ranked = sorted(rows, key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    lines = [
        "# V-JEPA 2.1 객체·언어 조건화 비교 실험",
        "",
        "동일 validation recording split을 사용했으며 탐색 실험은 150 epoch, 기준 모델은 300 epoch입니다. 최종 선택은 탐색 1회 결과가 아니라 300 epoch 다중 seed 재검증으로 확정해야 합니다.",
        "",
        "| 순위 | 실험 | 그룹 | Accuracy | Macro-F1 | 최고 epoch |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {row['experiment']} | {row['group']} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['best_epoch']} |"
        )
    lines.extend(["", "## 실험 의도", ""])
    for row in rows:
        lines.append(f"- `{row['experiment']}`: {row['hypothesis']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    output_dir = Path(suite["output_dir"])
    rows = []
    missing = []
    for reference in REFERENCE_RUNS:
        row = load_row(
            reference["id"], reference["group"], reference["hypothesis"],
            Path(reference["summary"]), Path(reference["history"]),
        )
        if row:
            rows.append(row)
    for experiment in suite["experiments"]:
        exp_dir = output_dir / experiment["id"]
        row = load_row(
            experiment["id"], experiment["group"], experiment["hypothesis"],
            exp_dir / "evaluation_summary.json", exp_dir / "training_history.csv",
        )
        if row is None:
            missing.append(experiment["id"])
        else:
            rows.append(row)
    if missing and not args.allow_partial:
        raise RuntimeError(f"missing experiments: {missing}")
    report_dir = output_dir / "summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, report_dir / "ablation_results.csv")
    (report_dir / "ablation_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_dashboard(rows, report_dir / "ablation_dashboard.png")
    write_report(rows, report_dir / "README.md")
    print(f"Summarized {len(rows)} experiments; missing={missing}")
    print(report_dir.resolve())


if __name__ == "__main__":
    main()
