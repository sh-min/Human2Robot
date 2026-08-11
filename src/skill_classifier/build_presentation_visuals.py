"""Build a presentation-ready visual report for a spatial-attention classifier."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, FancyBboxPatch
from torch.utils.data import DataLoader

from data_preprocess.preprocess import VJEPA2_EVAL_CROP, preprocess_rgb_frame
from skill_classifier.models import build_model
from skill_classifier.skill_dataset import SkillWindowDataset, load_recordings


NAVY = "#15324A"
BLUE = "#1976D2"
CYAN = "#20A4B8"
ORANGE = "#F59E0B"
RED = "#D94A4A"
GREEN = "#2E8B57"
PURPLE = "#7A5195"
LIGHT = "#F4F7FA"
CLASS_COLORS = ("#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_rgb(source: Path, frame_index: int) -> np.ndarray:
    if source.is_dir():
        paths = sorted(
            path for path in source.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        bgr = cv2.imread(str(paths[frame_index]), cv2.IMREAD_COLOR)
    else:
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"cannot decode {source} frame {frame_index}")
    if bgr is None:
        raise RuntimeError(f"cannot read {source} frame {frame_index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frame_for_sample(dataset, bundles, sample_index: int) -> tuple[np.ndarray, dict]:
    rec_index, token_index, _ = dataset.samples[sample_index]
    bundle = bundles[rec_index]
    frame_index = int(bundle["token_center_frame_indices"][token_index])
    source = Path(bundle["input_provenance"]["rgb"]["path"])
    rgb = preprocess_rgb_frame(load_rgb(source, frame_index), 384, VJEPA2_EVAL_CROP)
    return rgb, {
        "recording": bundle["recording"],
        "token_index": int(token_index),
        "frame_index": frame_index,
        "source": str(source),
    }


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, count: int) -> np.ndarray:
    matrix = np.zeros((count, count), dtype=int)
    for target, prediction in zip(labels, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def class_metrics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diagonal = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1, support


def calibration_stats(probabilities: np.ndarray, labels: np.ndarray) -> tuple[list[dict], float, float]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    bins = []
    expected_calibration_error = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (
            confidence <= upper if index == len(edges) - 2 else confidence < upper
        )
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else 0.0
        mean_confidence = float(confidence[mask].mean()) if count else (lower + upper) / 2
        expected_calibration_error += count / len(labels) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    one_hot = np.eye(probabilities.shape[1])[labels]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return bins, float(expected_calibration_error), brier


def add_metric_card(ax, xy, width, height, title, value, subtitle, color):
    card = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#D7E0E8",
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.text(xy[0] + 0.03, xy[1] + height - 0.08, title, transform=ax.transAxes,
            fontsize=11, color="#607080", va="top")
    ax.text(xy[0] + 0.03, xy[1] + height * 0.48, value, transform=ax.transAxes,
            fontsize=24, color=color, va="center", fontweight="bold")
    ax.text(xy[0] + 0.03, xy[1] + 0.045, subtitle, transform=ax.transAxes,
            fontsize=9.5, color="#607080", va="bottom")


def executive_summary(output_dir, summary, names, recall, config):
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.48, 0.52), width_ratios=(0.52, 0.48))
    cards = fig.add_subplot(grid[0, :])
    bars = fig.add_subplot(grid[1, 0])
    pipeline = fig.add_subplot(grid[1, 1])
    cards.axis("off")
    card_values = (
        ("Validation Accuracy", f"{summary['validation_accuracy']:.1%}", "Held-out robot-overlay frames", BLUE),
        ("Macro F1", f"{summary['validation_f1_macro']:.1%}", "Equal weight across 6 skills", PURPLE),
        ("Weighted F1", f"{summary['validation_f1_weighted']:.1%}", "Weighted by class support", GREEN),
        ("Selected Epoch", str(summary["best_epoch"]), "Best validation checkpoint", ORANGE),
    )
    for index, values in enumerate(card_values):
        add_metric_card(cards, (0.015 + index * 0.247, 0.11), 0.225, 0.70, *values)

    order = np.argsort(recall)
    colors = [GREEN if recall[index] >= 0.8 else ORANGE if recall[index] >= 0.6 else RED for index in order]
    bars.barh(np.asarray(names)[order], recall[order], color=colors)
    bars.set_xlim(0, 1.02)
    bars.set_xlabel("Recall")
    bars.set_title("Per-class recall", loc="left", fontsize=15, fontweight="bold")
    bars.grid(axis="x", alpha=0.2)
    for y, value in enumerate(recall[order]):
        bars.text(min(value + 0.018, 0.93), y, f"{value:.1%}", va="center", fontweight="bold")

    pipeline.axis("off")
    pipeline.set_title("Model and evaluation contract", loc="left", fontsize=15, fontweight="bold")
    blocks = [
        ("Robot-overlay\nvideo", BLUE),
        ("Frozen V-JEPA 2.1\nViT-L / 384", CYAN),
        ("Learned spatial\nattention", PURPLE),
        ("8-token context\n+ MLP, D0.4", ORANGE),
        ("6 skill classes", GREEN),
    ]
    y = 0.62
    for index, (label, color) in enumerate(blocks):
        x = 0.015 + index * 0.195
        block = FancyBboxPatch(
            (x, y - 0.105),
            0.17,
            0.21,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            linewidth=0,
            facecolor=color,
            transform=pipeline.transAxes,
        )
        pipeline.add_patch(block)
        pipeline.text(
            x + 0.085,
            y,
            label,
            transform=pipeline.transAxes,
            ha="center",
            va="center",
            fontsize=8.6,
            color="white",
            fontweight="bold",
        )
        if index < len(blocks) - 1:
            pipeline.annotate("", xy=(x + 0.192, y), xytext=(x + 0.171, y),
                              xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#657786"})
    contract = (
        f"Train: {summary['train_recordings']} views / {summary['train_samples']:,} samples   |   "
        f"Validation: {summary['validation_recordings']} scenes / {summary['validation_samples']:,} samples\n"
        "Color Jitter: train only   |   Choco excluded   |   Scene-disjoint validation   |   "
        f"Seed {summary['seed']}"
    )
    pipeline.text(0.02, 0.22, contract, transform=pipeline.transAxes, fontsize=10.5,
                  color=NAVY, va="top", linespacing=1.6,
                  bbox={"boxstyle": "round,pad=0.7", "facecolor": LIGHT, "edgecolor": "#D7E0E8"})
    fig.suptitle("V-JEPA 2.1 Skill Classifier — Best Model Summary", fontsize=22,
                 fontweight="bold", color=NAVY, y=0.98)
    fig.text(0.5, 0.02, "Spatial Attention + Color Jitter + Dropout 0.4", ha="center",
             fontsize=13, color="#526878")
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.94))
    save_figure(fig, output_dir, "01_executive_summary")


def plot_confusion(output_dir, matrix, names):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8))
    for ax, data, title, is_normalized in (
        (axes[0], matrix, "Prediction counts", False),
        (axes[1], normalized, "Row-normalized recall", True),
    ):
        image = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if is_normalized else None)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted class", fontweight="bold")
        ax.set_ylabel("Ground-truth class", fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold")
        threshold = 0.5 if is_normalized else matrix.max() / 2
        for row in range(len(names)):
            for column in range(len(names)):
                value = data[row, column]
                text = f"{value:.0%}" if is_normalized else str(int(value))
                ax.text(column, row, text, ha="center", va="center", fontsize=11,
                        color="white" if value > threshold else NAVY,
                        fontweight="bold" if row == column else "normal")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Held-out Confusion Matrix — Best D0.4 Checkpoint", fontsize=21,
                 fontweight="bold", color=NAVY)
    fig.text(0.5, 0.012, "Diagonal cells are correct predictions; off-diagonal cells reveal class confusion.",
             ha="center", fontsize=11, color="#526878")
    fig.tight_layout(rect=(0.02, 0.085, 0.98, 0.93))
    save_figure(fig, output_dir, "02_confusion_matrix")


def plot_class_metrics(output_dir, names, precision, recall, f1, support):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": (0.72, 0.28)})
    x = np.arange(len(names))
    width = 0.24
    axes[0].bar(x - width, precision, width, label="Precision", color=BLUE)
    axes[0].bar(x, recall, width, label="Recall", color=ORANGE)
    axes[0].bar(x + width, f1, width, label="F1", color=PURPLE)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision, recall, and F1 by skill", fontsize=15, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=3, loc="lower right")
    for offset, values in ((-width, precision), (0, recall), (width, f1)):
        for index, value in enumerate(values):
            axes[0].text(index + offset, value + 0.018, f"{value:.2f}", ha="center", fontsize=8)

    bars = axes[1].barh(names, support, color=CLASS_COLORS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Validation samples")
    axes[1].set_title("Class support", fontsize=15, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, support):
        axes[1].text(value + 1, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    fig.suptitle("Class-wise Performance and Validation Balance", fontsize=21,
                 fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    save_figure(fig, output_dir, "03_class_metrics")


def read_history(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def plot_learning(output_dir, history, best_epoch):
    epochs = np.asarray([row["epoch"] for row in history])
    best = int(np.argmin(np.abs(epochs - best_epoch)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    panels = (
        ("Accuracy", "train_acc", "val_acc", "Train", "Validation"),
        ("Loss", "train_loss", "val_loss", "Train", "Validation"),
        ("Validation F1", "val_f1_macro", "val_f1_weighted", "Macro F1", "Weighted F1"),
    )
    for ax, (title, first, second, first_name, second_name) in zip(axes, panels):
        first_values = np.asarray([row[first] for row in history])
        second_values = np.asarray([row[second] for row in history])
        ax.plot(epochs, first_values, label=first_name, color=BLUE, linewidth=1.8)
        ax.plot(epochs, second_values, label=second_name, color=ORANGE, linewidth=1.8)
        ax.axvline(best_epoch, color=RED, linestyle="--", linewidth=1.3, label=f"Best epoch {best_epoch}")
        ax.scatter([best_epoch], [second_values[best]], color=RED, s=45, zorder=4)
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylim(0.25, 1.02)
    axes[2].set_ylim(0.25, 1.02)
    fig.suptitle("Training Dynamics — Attention + Color Jitter + Dropout 0.4",
                 fontsize=20, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
    save_figure(fig, output_dir, "04_learning_curves")


def plot_attention_representatives(output_dir, selected, names):
    fig, axes = plt.subplots(2, len(names), figsize=(18, 6.6))
    for column, item in enumerate(selected):
        axes[0, column].imshow(item["rgb"])
        axes[0, column].set_title(
            f"{item['true_name']}\n{item['recording']} · frame {item['frame_index']}",
            fontsize=9.5, fontweight="bold",
        )
        axes[0, column].axis("off")
        axes[1, column].imshow(item["rgb"])
        axes[1, column].imshow(item["heatmap"], cmap="turbo", alpha=0.58,
                               extent=(0, 384, 384, 0), interpolation="bilinear")
        axes[1, column].set_title(
            f"p={item['confidence']:.2f} · entropy={item['entropy']:.3f}", fontsize=9
        )
        axes[1, column].axis("off")
    fig.suptitle("What the Model Attends To — Correct High-confidence Examples",
                 fontsize=19, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.02, "Top: model input crop   |   Bottom: learned spatial attention for the current token",
             ha="center", fontsize=10.5, color="#526878")
    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.91))
    save_figure(fig, output_dir, "05_attention_representatives")


def plot_attention_statistics(output_dir, attention, labels, names):
    side = round(math.sqrt(attention.shape[1]))
    entropy = -np.sum(attention * np.log(attention + 1e-12), axis=1) / math.log(attention.shape[1])
    top_count = max(1, math.ceil(attention.shape[1] * 0.1))
    top_mass = np.sort(attention, axis=1)[:, -top_count:].sum(axis=1)
    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, len(names), height_ratios=(0.62, 0.38))
    images = []
    for index, name in enumerate(names):
        ax = fig.add_subplot(grid[0, index])
        mean_map = attention[labels == index].mean(axis=0).reshape(side, side)
        images.append(mean_map)
        ax.imshow(mean_map, cmap="turbo", interpolation="bilinear")
        ax.set_title(f"{name}\nmean attention", fontsize=10, fontweight="bold")
        ax.axis("off")
    entropy_ax = fig.add_subplot(grid[1, :3])
    mass_ax = fig.add_subplot(grid[1, 3:])
    positions = np.arange(len(names))
    entropy_means = np.asarray([entropy[labels == index].mean() for index in positions])
    entropy_stds = np.asarray([entropy[labels == index].std() for index in positions])
    mass_means = np.asarray([top_mass[labels == index].mean() for index in positions])
    mass_stds = np.asarray([top_mass[labels == index].std() for index in positions])
    entropy_ax.bar(positions, entropy_means, yerr=entropy_stds, color=CLASS_COLORS, capsize=3)
    entropy_ax.set_xticks(positions, names)
    entropy_ax.set_ylim(max(0, entropy_means.min() - 0.08), 1.01)
    entropy_ax.set_ylabel("Normalized entropy")
    entropy_ax.set_title("Attention spread (lower = more concentrated)", fontweight="bold")
    entropy_ax.grid(axis="y", alpha=0.2)
    mass_ax.bar(positions, mass_means, yerr=mass_stds, color=CLASS_COLORS, capsize=3)
    mass_ax.axhline(0.1, linestyle="--", color="#526878", label="Uniform baseline")
    mass_ax.set_xticks(positions, names)
    mass_ax.set_ylabel("Mass in top 10% patches")
    mass_ax.set_title("Attention concentration", fontweight="bold")
    mass_ax.grid(axis="y", alpha=0.2)
    mass_ax.legend(fontsize=9)
    fig.suptitle("Attention Distribution Across Validation Skills", fontsize=20,
                 fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    save_figure(fig, output_dir, "06_attention_statistics")
    return entropy, top_mass


def plot_confidence(output_dir, probabilities, labels, names):
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    bins, ece, brier = calibration_stats(probabilities, labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    centers = np.asarray([(item["lower"] + item["upper"]) / 2 for item in bins])
    accuracy = np.asarray([item["accuracy"] for item in bins])
    counts = np.asarray([item["count"] for item in bins])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777", label="Perfect calibration")
    axes[0].plot(centers[counts > 0], accuracy[counts > 0], marker="o", color=BLUE,
                 linewidth=2, label="Model")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Mean confidence bin")
    axes[0].set_ylabel("Observed accuracy")
    axes[0].set_title(f"Reliability · ECE {ece:.3f}", fontweight="bold")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=9)

    edges = np.linspace(0, 1, 16)
    axes[1].hist(confidence[correct], bins=edges, alpha=0.75, color=GREEN,
                 label=f"Correct ({correct.sum()})")
    axes[1].hist(confidence[~correct], bins=edges, alpha=0.75, color=RED,
                 label=f"Incorrect ({(~correct).sum()})")
    axes[1].set_xlabel("Prediction confidence")
    axes[1].set_ylabel("Samples")
    axes[1].set_title(f"Confidence distribution · Brier {brier:.3f}", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)

    class_confidence = [confidence[labels == index].mean() for index in range(len(names))]
    class_accuracy = [(predictions[labels == index] == index).mean() for index in range(len(names))]
    x = np.arange(len(names))
    width = 0.36
    axes[2].bar(x - width / 2, class_accuracy, width, color=ORANGE, label="Recall")
    axes[2].bar(x + width / 2, class_confidence, width, color=BLUE, label="Mean confidence")
    axes[2].set_xticks(x, names, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Confidence versus recall", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(fontsize=9)
    fig.suptitle("Prediction Confidence and Calibration", fontsize=20,
                 fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
    save_figure(fig, output_dir, "07_confidence_calibration")
    return bins, ece, brier


def plot_error_gallery(output_dir, selected_errors):
    columns = 4
    rows = math.ceil(len(selected_errors) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, selected_errors):
        ax.imshow(item["rgb"])
        ax.imshow(item["heatmap"], cmap="turbo", alpha=0.48,
                  extent=(0, 384, 384, 0), interpolation="bilinear")
        ax.set_title(
            f"GT {item['true_name']}  →  Pred {item['pred_name']}\n"
            f"confidence {item['confidence']:.2f} · {item['recording']} · frame {item['frame_index']}",
            fontsize=10, color=RED, fontweight="bold",
        )
        ax.axis("off")
    fig.suptitle("High-confidence Failure Cases with Attention", fontsize=20,
                 fontweight="bold", color=NAVY)
    fig.text(0.5, 0.02, "These examples are diagnostic cases, not representative successes.",
             ha="center", fontsize=10.5, color="#526878")
    fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.93))
    save_figure(fig, output_dir, "08_error_gallery")


def plot_timelines(output_dir, dataset, bundles, labels, predictions, names):
    fig, axes = plt.subplots(len(bundles), 1, figsize=(16, 11), sharex=False)
    cmap = ListedColormap(CLASS_COLORS)
    for rec_index, (ax, bundle) in enumerate(zip(axes, bundles)):
        indices = [index for index, sample in enumerate(dataset.samples) if sample[0] == rec_index]
        true = labels[indices]
        pred = predictions[indices]
        values = np.stack((true, pred))
        ax.imshow(values, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, aspect="auto",
                  interpolation="nearest")
        mismatches = np.flatnonzero(true != pred)
        ax.scatter(mismatches, np.full(len(mismatches), 1), marker="x", s=35,
                   color="black", linewidth=1.3)
        accuracy = float((true == pred).mean())
        ax.set_yticks((0, 1), ("GT", "Pred"))
        ax.set_title(f"{bundle['recording']}  ·  {len(indices)} labeled tokens  ·  accuracy {accuracy:.1%}",
                     loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Labeled token order (2 Hz feature rate)", fontsize=9)
    legend = [Patch(facecolor=color, label=name) for color, name in zip(CLASS_COLORS, names)]
    fig.legend(handles=legend, loc="upper center", ncol=len(names), bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Temporal Prediction Timeline by Held-out Recording", fontsize=20,
                 fontweight="bold", color=NAVY, y=0.99)
    fig.text(0.5, 0.015, "Black × markers denote incorrect predictions.", ha="center",
             fontsize=10.5, color="#526878")
    fig.tight_layout(rect=(0.03, 0.035, 0.98, 0.92))
    save_figure(fig, output_dir, "09_temporal_predictions")


def plot_key_findings(output_dir, matrix, names, recall, summary):
    errors = int(matrix.sum() - np.trace(matrix))
    transition_index = names.index("Trans")
    transition_errors = int(
        matrix[transition_index].sum()
        + matrix[:, transition_index].sum()
        - 2 * matrix[transition_index, transition_index]
    )
    strongest = int(np.argmax(recall))
    weakest = int(np.argmin(recall))
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor(LIGHT)
    ax.axis("off")
    findings = (
        (
            "Headline result",
            f"{summary['validation_accuracy']:.1%}",
            f"Macro F1 {summary['validation_f1_macro']:.1%}\nBest checkpoint: epoch {summary['best_epoch']}",
            BLUE,
        ),
        (
            "Strongest skill",
            names[strongest],
            f"Recall {recall[strongest]:.1%}\nReliable on the held-out split",
            GREEN,
        ),
        (
            "Primary weakness",
            names[weakest],
            f"Recall {recall[weakest]:.1%}\nNeeds more diverse examples",
            RED,
        ),
        (
            "Boundary-related errors",
            f"{transition_errors}/{errors}",
            f"{transition_errors / errors:.0%} of errors involve Trans\nTemporal boundary modeling is the next target",
            ORANGE,
        ),
    )
    for index, (heading, value, detail, color) in enumerate(findings):
        column = index % 2
        row = index // 2
        x = 0.06 + column * 0.48
        y = 0.53 - row * 0.39
        card = FancyBboxPatch(
            (x, y), 0.40, 0.30,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2, edgecolor="#D7E0E8", facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(card)
        ax.add_patch(FancyBboxPatch(
            (x, y), 0.012, 0.30,
            boxstyle="round,pad=0,rounding_size=0.006",
            linewidth=0, facecolor=color, transform=ax.transAxes,
        ))
        ax.text(x + 0.035, y + 0.25, heading, transform=ax.transAxes,
                fontsize=13, color="#607080", va="top", fontweight="bold")
        ax.text(x + 0.035, y + 0.155, value, transform=ax.transAxes,
                fontsize=28, color=color, va="center", fontweight="bold")
        ax.text(x + 0.035, y + 0.04, detail, transform=ax.transAxes,
                fontsize=11, color=NAVY, va="bottom", linespacing=1.45)
    fig.suptitle("Key Findings and Next Step", fontsize=24, fontweight="bold", color=NAVY, y=0.95)
    fig.text(0.5, 0.045,
             "Best current setting: V-JEPA 2.1 + Spatial Attention + train-only Color Jitter + Dropout 0.4",
             ha="center", fontsize=12, color="#526878")
    save_figure(fig, output_dir, "10_key_findings")


def write_gallery(output_dir: Path, figures: list[tuple[str, str]]) -> None:
    cards = "".join(
        f"<figure><a href='{html.escape(name)}.png'><img src='{html.escape(name)}.png'></a>"
        f"<figcaption>{html.escape(caption)} · "
        f"<a href='{html.escape(name)}.pdf'>PDF</a></figcaption></figure>"
        for name, caption in figures
    )
    document = f"""<!doctype html><meta charset='utf-8'>
<title>V-JEPA 2.1 D0.4 Presentation Visuals</title>
<style>
body{{font:16px system-ui,sans-serif;background:#eef2f6;color:#15324a;max-width:1500px;margin:30px auto;padding:0 20px}}
h1{{font-size:30px}} .meta{{background:white;padding:18px;border-radius:10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:20px;margin-top:20px}}
figure{{margin:0;background:white;padding:12px;border-radius:10px;box-shadow:0 2px 10px #0001}}
img{{width:100%;height:auto}} figcaption{{padding:10px 4px;font-weight:600}} a{{color:#1976d2}}
</style>
<h1>Best model presentation visuals</h1>
<div class='meta'>V-JEPA 2.1 · Spatial Attention · train-only Color Jitter · Dropout 0.4 · seed 42<br>
<a href='presentation_metrics.json'>Metrics JSON</a> · <a href='predictions.csv'>Per-sample predictions</a> ·
<a href='README.md'>Slide guide</a> · <a href='PRESENTATION_GUIDE_KO.md'>한국어 발표 가이드</a>
 · <a href='attention_videos/index.html'>전체 Attention 영상</a>
</div><div class='grid'>{cards}</div>"""
    (output_dir / "index.html").write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    args.experiment_dir = args.experiment_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((args.experiment_dir / "config.yaml").read_text())
    summary = json.loads((args.experiment_dir / "evaluation_summary.json").read_text())
    checkpoint = torch.load(
        args.experiment_dir / "best_spatial_attention_mlp.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_args = checkpoint["args"]
    names = list(model_args["active_labels"])
    if float(model_args["dropout"]) != 0.4:
        raise ValueError("presentation report expects the selected Dropout 0.4 checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        "spatial_attention_mlp",
        vjepa_dim=int(model_args["vjepa_dim"]),
        hand_dim=int(model_args["hand_dim"]),
        window_size=int(model_args["window_size"]),
        num_classes=len(names),
        hidden_dims=tuple(model_args["hidden_dims"]),
        dropout=float(model_args["dropout"]),
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    bundles = load_recordings(config["val_data_root"], config["val_recording_glob"])
    dataset = SkillWindowDataset(
        bundles,
        window_size=int(config["window_size"]),
        variant=config["variant"],
        vjepa_diff=bool(config.get("vjepa_diff", False)),
        hand_representation=config["hand_representation"],
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    all_probabilities, all_labels, all_attention = [], [], []
    with torch.inference_mode():
        for dense, hand, labels in loader:
            dense = dense.to(device)
            hand = hand.to(device)
            logits = model(dense, hand)
            _, weights = model.representation(dense)
            all_probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_attention.append(weights[:, -1].cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels).astype(int)
    attention = np.concatenate(all_attention)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    matrix = confusion_matrix(labels, predictions, len(names))
    precision, recall, f1, support = class_metrics(matrix)

    executive_summary(args.output_dir, summary, names, recall, config)
    plot_confusion(args.output_dir, matrix, names)
    plot_class_metrics(args.output_dir, names, precision, recall, f1, support)
    plot_learning(args.output_dir, read_history(args.experiment_dir / "training_history.csv"), summary["best_epoch"])

    representative = []
    for class_index, name in enumerate(names):
        candidates = np.flatnonzero((labels == class_index) & (predictions == class_index))
        if not len(candidates):
            candidates = np.flatnonzero(labels == class_index)
        sample_index = int(candidates[np.argmax(probabilities[candidates, class_index])])
        rgb, metadata = frame_for_sample(dataset, bundles, sample_index)
        weights = attention[sample_index]
        entropy = -float(np.sum(weights * np.log(weights + 1e-12))) / math.log(len(weights))
        representative.append(
            {
                **metadata,
                "sample_index": sample_index,
                "true_name": name,
                "pred_name": names[predictions[sample_index]],
                "confidence": float(confidence[sample_index]),
                "entropy": entropy,
                "rgb": rgb,
                "heatmap": weights.reshape(24, 24),
            }
        )
    plot_attention_representatives(args.output_dir, representative, names)
    entropy, top_mass = plot_attention_statistics(args.output_dir, attention, labels, names)
    calibration_bins, ece, brier = plot_confidence(args.output_dir, probabilities, labels, names)

    error_indices = np.flatnonzero(predictions != labels)
    total_errors = int(len(error_indices))
    error_indices = error_indices[np.argsort(confidence[error_indices])[::-1]][:8]
    selected_errors = []
    for sample_index in error_indices:
        rgb, metadata = frame_for_sample(dataset, bundles, int(sample_index))
        selected_errors.append(
            {
                **metadata,
                "sample_index": int(sample_index),
                "true_name": names[labels[sample_index]],
                "pred_name": names[predictions[sample_index]],
                "confidence": float(confidence[sample_index]),
                "rgb": rgb,
                "heatmap": attention[sample_index].reshape(24, 24),
            }
        )
    plot_error_gallery(args.output_dir, selected_errors)
    plot_timelines(args.output_dir, dataset, bundles, labels, predictions, names)
    plot_key_findings(args.output_dir, matrix, names, recall, summary)

    prediction_rows = []
    for sample_index, (target, prediction, probability) in enumerate(zip(labels, predictions, confidence)):
        rec_index, token_index, _ = dataset.samples[sample_index]
        bundle = bundles[rec_index]
        prediction_rows.append(
            {
                "sample_index": sample_index,
                "recording": bundle["recording"],
                "token_index": token_index,
                "source_frame": int(bundle["token_center_frame_indices"][token_index]),
                "ground_truth": names[target],
                "prediction": names[prediction],
                "correct": int(target == prediction),
                "confidence": float(probability),
                "attention_entropy": float(entropy[sample_index]),
                "attention_top10_mass": float(top_mass[sample_index]),
            }
        )
    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    metrics = {
        "experiment_dir": str(args.experiment_dir),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dropout": float(model_args["dropout"]),
        "summary": summary,
        "class_names": names,
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "calibration": {"ece_10_bin": ece, "multiclass_brier": brier, "bins": calibration_bins},
        "errors": total_errors,
    }
    (args.output_dir / "presentation_metrics.json").write_text(json.dumps(metrics, indent=2))

    figures = [
        ("01_executive_summary", "Executive summary"),
        ("02_confusion_matrix", "Confusion matrix"),
        ("03_class_metrics", "Class-wise metrics"),
        ("04_learning_curves", "Training dynamics"),
        ("05_attention_representatives", "Representative attention heatmaps"),
        ("06_attention_statistics", "Attention distribution statistics"),
        ("07_confidence_calibration", "Confidence and calibration"),
        ("08_error_gallery", "High-confidence error analysis"),
        ("09_temporal_predictions", "Temporal prediction timelines"),
        ("10_key_findings", "Key findings and next step"),
    ]
    write_gallery(args.output_dir, figures)
    readme = "# Presentation visual pack\n\n"
    readme += "Selected model: V-JEPA 2.1 spatial attention + train-only Color Jitter + Dropout 0.4.\n\n"
    readme += "Suggested slide order:\n\n"
    readme += "1. `01_executive_summary` — model recipe and headline result\n"
    readme += "2. `02_confusion_matrix` and `03_class_metrics` — quantitative evaluation\n"
    readme += "3. `05_attention_representatives` — qualitative explanation\n"
    readme += "4. `08_error_gallery` and `09_temporal_predictions` — limitations and future work\n"
    readme += "5. `10_key_findings` — conclusion slide\n\n"
    readme += "PNG files are slide-ready; PDF files are vector versions for publication.\n"
    (args.output_dir / "README.md").write_text(readme)
    guide_ko = """# 발표용 시각화 가이드\n\n
## 추천 구성\n\n
1. `01_executive_summary`: 최종 모델 구조와 Accuracy 80.6%, Macro F1 73.2%를 설명합니다.\n
2. `02_confusion_matrix`: Cup과 Sweep은 잘 구별하지만 Milk, Lock, Snack은 서로 혼동되는 사례가 남아 있음을 설명합니다.\n
3. `03_class_metrics`: 클래스별 데이터 수가 불균형하므로 Accuracy와 함께 Macro F1을 제시해야 한다고 설명합니다.\n
4. `04_learning_curves`: 255 epoch의 최고 체크포인트를 선택했으며 train-validation 간 일반화 차이가 남아 있음을 설명합니다.\n
5. `05_attention_representatives`: 빨강/노랑 영역이 상대적으로 높은 attention입니다. 로봇손과 조작 물체 주변을 함께 참고하는 정성 결과입니다.\n
6. `06_attention_statistics`: attention은 한 점에만 몰리지 않지만 상위 10% 패치에 약 24~27%의 질량을 배분합니다. 점선은 균일 attention 기준 10%입니다.\n
7. `07_confidence_calibration`: ECE 0.073이며, 높은 confidence 구간은 대체로 정확하지만 고신뢰 오분류도 남아 있습니다.\n
8. `08_error_gallery`: 실패 사례를 제시합니다. 전체 48개 오류 중 32개(67%)가 Trans와 다른 클래스 사이에서 발생합니다.\n
9. `09_temporal_predictions`: 오류가 영상 전체에 균일하게 생기기보다 동작 전환 구간에 집중되는지를 녹화본별로 보여줍니다.\n
10. `10_key_findings`: 결론 및 다음 개선 방향으로 temporal boundary modeling을 제안합니다.\n\n
## 주의사항\n\n
- 현재 결과는 seed 42 단일 실행입니다. Dropout 0.4가 0.3보다 검증 샘플 1개를 더 맞힌 수준이므로 통계적으로 확정된 차이라고 표현하면 안 됩니다.\n
- Attention heatmap은 모델의 내부 가중치를 시각화한 것으로, 인과적인 설명이나 정확한 객체 segmentation으로 해석하면 안 됩니다.\n
- PNG는 슬라이드 삽입용, PDF는 확대 가능한 벡터 버전입니다.\n"""
    (args.output_dir / "PRESENTATION_GUIDE_KO.md").write_text(guide_ko)
    source_attention = args.experiment_dir / "attention" / "representative_attention.png"
    if source_attention.is_file():
        shutil.copy2(source_attention, args.output_dir / "reference_attention_original.png")
    print(f"[ok] presentation gallery: {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
