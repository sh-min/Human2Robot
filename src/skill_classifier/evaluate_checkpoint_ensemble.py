"""Evaluate a fixed 50:50 ensemble and an explicitly exploratory weight sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader

from skill_classifier.compare_model_diagnostics import (
    _brier,
    _build_model,
    _dataset,
    _ece,
    _evaluate,
)


def _load_probabilities(path, batch_size, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    dataset = _dataset(saved_args)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _build_model(checkpoint, dataset).to(device).eval()
    object_model = saved_args["model"] in (
        "object_mask_attention_mlp", "object_text_prototype_mlp"
    )
    probabilities, labels, _ = _evaluate(model, loader, device, object_model)
    return probabilities, labels, list(saved_args["active_labels"])


def _metrics(probabilities, labels, classes):
    # Float32 convex combinations can miss an exact unit row sum by a few ULPs.
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    ece, _ = _ece(probabilities, labels)
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(classes))),
        "brier": _brier(probabilities, labels, classes),
        "ece": ece,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--object-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    global_probability, global_labels, global_names = _load_probabilities(
        args.global_checkpoint, args.batch_size, device
    )
    object_probability, object_labels, object_names = _load_probabilities(
        args.object_checkpoint, args.batch_size, device
    )
    if global_names != object_names or not np.array_equal(global_labels, object_labels):
        raise ValueError("ensemble checkpoints do not share the same validation samples")

    rows = []
    for alpha in np.linspace(0.0, 1.0, 21):
        probability = (1.0 - alpha) * global_probability + alpha * object_probability
        rows.append({"object_weight": float(alpha), **_metrics(probability, global_labels, len(global_names))})
    fixed = next(row for row in rows if abs(row["object_weight"] - 0.5) < 1.0e-8)
    best_accuracy = max(rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    best_macro = max(rows, key=lambda row: (row["macro_f1"], row["accuracy"]))
    payload = {
        "global_checkpoint": str(args.global_checkpoint.resolve()),
        "object_checkpoint": str(args.object_checkpoint.resolve()),
        "validation_samples": len(global_labels),
        "global_only": rows[0],
        "object_only": rows[-1],
        "fixed_50_50": fixed,
        "exploratory_best_accuracy": best_accuracy,
        "exploratory_best_macro_f1": best_macro,
        "warning": "Sweep optima are selected on the evaluation split and are not unbiased deployment estimates.",
    }
    (args.output_dir / "ensemble_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "ensemble_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metrics, title in (
        (axes[0], ("accuracy", "macro_f1"), "Classification quality"),
        (axes[1], ("nll", "ece"), "Calibration / confidence quality (lower is better)"),
    ):
        for metric in metrics:
            axis.plot([row["object_weight"] for row in rows],
                      [row[metric] for row in rows], marker="o", label=metric)
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1, label="fixed 50:50")
        axis.set_xlabel("Object-model probability weight")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Global/object probability ensemble sweep (exploratory)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "ensemble_sweep.png", dpi=180)
    plt.close(fig)
    print(json.dumps(payload, indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
