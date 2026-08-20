"""Render an ICRA-style double-column figure for the exact Global W4 model."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/skill_classifier/kitchen_0813_model_comparison/figures"
FRAME_ROOT = OUT / "source_frames"
RESULT_ROOT = (
    ROOT / "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco"
    / "multiseed_300e"
)

INK = "#20252d"
GRAY = "#6d7480"
LIGHT = "#edf0f4"
BLUE = "#3569b7"
BLUE_BG = "#edf3fb"
GREEN = "#31966b"
GREEN_BG = "#edf8f3"
ORANGE = "#d98127"
ORANGE_BG = "#fcf3e8"
PURPLE = "#7952a6"
RED = "#c94c59"
COLORS = (BLUE, ORANGE, GREEN, RED, PURPLE, "#747f8d")


def add_box(ax, xy, width, height, text, edge=INK, face="white", fontsize=7.0,
            weight="semibold", detail=None):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.9, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height * (0.58 if detail else 0.5), text,
            ha="center", va="center", fontsize=fontsize, fontweight=weight,
            color=edge, linespacing=1.05)
    if detail:
        ax.text(xy[0] + width / 2, xy[1] + height * 0.20, detail,
                ha="center", va="center", fontsize=5.5, color=GRAY)
    return patch


def add_arrow(ax, start, end, color=INK, dashed=False, lw=0.9):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8, linewidth=lw,
        color=color, linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)


def crop(path: Path, aspect=1.18):
    image = Image.open(path).convert("RGB")
    current = image.width / image.height
    if current > aspect:
        width = int(image.height * aspect)
        left = (image.width - width) // 2
        image = image.crop((left, 0, left + width, image.height))
    else:
        height = int(image.width / aspect)
        top = (image.height - height) // 2
        image = image.crop((0, top, image.width, top + height))
    return image


def add_frame(ax, path, extent):
    ax.imshow(crop(path), extent=extent, aspect="auto", zorder=2)
    ax.add_patch(Rectangle((extent[0], extent[2]), extent[1] - extent[0],
                           extent[3] - extent[2], fill=False, edgecolor=INK,
                           linewidth=0.6, zorder=3))


def main():
    summaries = [json.loads(path.read_text()) for path in
                 sorted(RESULT_ROOT.glob("global_w4_seed*/evaluation_summary.json"))]
    if len(summaries) != 3:
        raise ValueError(f"expected 3 Global W4 results, got {len(summaries)}")
    labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")
    accuracy_values = [float(item["validation_accuracy"]) for item in summaries]
    accuracy = statistics.mean(accuracy_values)
    accuracy_std = statistics.stdev(accuracy_values)
    recalls = [statistics.mean(float(item["per_class"][label]["accuracy"])
                               for item in summaries) for label in labels]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.16, 3.05), dpi=300, facecolor="white")
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # (a) Model architecture. No figure title; caption belongs in the paper.
    ax.text(0.004, 0.966, "(a)  Global W4 training architecture", fontsize=8.2,
            fontweight="bold", ha="left", va="top", color=INK)
    ax.plot([0.004, 0.996], [0.925, 0.925], color=LIGHT, linewidth=0.8)

    frame_paths = (
        FRAME_ROOT / "frame_01_cup.png",
        FRAME_ROOT / "frame_02_red_snack.png",
        FRAME_ROOT / "frame_03_wipe.png",
    )
    for i, path in enumerate(frame_paths):
        add_frame(ax, path, (0.012 + i * 0.047, 0.055 + i * 0.047, 0.688, 0.855))
    ax.text(0.079, 0.875, "labeled clips", ha="center", va="bottom",
            fontsize=6.4, fontweight="semibold", color=INK)
    for i, label in enumerate(labels):
        x = 0.008 + (i % 3) * 0.047
        y = 0.64 - (i // 3) * 0.045
        ax.text(x + 0.020, y, label, ha="center", va="center", fontsize=4.8,
                color=COLORS[i], fontweight="semibold")

    add_arrow(ax, (0.154, 0.765), (0.180, 0.765))
    add_box(ax, (0.186, 0.680), 0.115, 0.170, "Frozen\nV-JEPA2.1",
            BLUE, BLUE_BG, detail="ViT-L/16 · 384 px")
    ax.text(0.2435, 0.655, "frozen", ha="center", va="top", fontsize=5.2,
            color=BLUE, fontweight="semibold")

    add_arrow(ax, (0.307, 0.765), (0.332, 0.765))
    # Four dense token grids.
    for token in range(4):
        x0 = 0.339 + token * 0.026
        for row in range(3):
            for col in range(3):
                ax.add_patch(Rectangle((x0 + col * 0.006, 0.748 + row * 0.017),
                                       0.0045, 0.012, facecolor=BLUE_BG,
                                       edgecolor=BLUE, linewidth=0.35))
        ax.text(x0 + 0.006, 0.728, rf"$t_{token + 1}$", ha="center", va="top",
                fontsize=4.8, color=GRAY)
    ax.text(0.382, 0.845, "dense tokens", ha="center", va="bottom",
            fontsize=6.2, fontweight="semibold", color=INK)
    ax.text(0.382, 0.690, r"$\mathbf{X}\in\mathbb{R}^{4\times S\times1024}$",
            ha="center", va="center", fontsize=5.4, color=GRAY)

    add_arrow(ax, (0.440, 0.765), (0.463, 0.765))
    add_box(ax, (0.469, 0.680), 0.120, 0.170, "Spatial\nattention",
            BLUE, BLUE_BG, detail=r"softmax over $S$")
    ax.text(0.529, 0.655, "trainable", ha="center", va="top", fontsize=5.2,
            color=GREEN, fontweight="semibold")

    add_arrow(ax, (0.595, 0.765), (0.618, 0.765))
    add_box(ax, (0.624, 0.680), 0.096, 0.170, "Temporal\nmean",
            GREEN, GREEN_BG, detail=r"$\frac{1}{4}\sum_{t=1}^{4}$")

    add_arrow(ax, (0.726, 0.765), (0.749, 0.765))
    add_box(ax, (0.755, 0.680), 0.143, 0.170, "MLP",
            ORANGE, ORANGE_BG, detail="1024 → 256 → 128 → 6")
    ax.text(0.8265, 0.655, "ReLU · dropout 0.4", ha="center", va="top",
            fontsize=5.1, color=GRAY)

    add_arrow(ax, (0.904, 0.765), (0.927, 0.765))
    add_box(ax, (0.933, 0.680), 0.060, 0.170, "skill\nlogits", PURPLE,
            "#f5f1fa", fontsize=6.5, detail="6 classes")

    # Ground-truth supervision attaches at the output loss.
    ax.text(0.079, 0.565, "primitive label", ha="center", va="center",
            fontsize=5.5, color=RED, fontweight="semibold")
    ax.plot([0.079, 0.963], [0.545, 0.545], color=RED, linewidth=0.65)
    add_arrow(ax, (0.963, 0.545), (0.963, 0.672), RED, lw=0.65)
    ax.text(0.804, 0.555, r"$\mathcal{L}_{CE}$  (label smoothing 0.1)",
            ha="center", va="bottom", fontsize=5.2, color=RED)

    # Divider between architecture and downstream use.
    ax.plot([0.004, 0.996], [0.495, 0.495], color="#cfd4dc", linewidth=0.8)

    # (b) Long-horizon inference.
    ax.text(0.004, 0.465, "(b)  Long-horizon inference", fontsize=8.2,
            fontweight="bold", ha="left", va="top", color=INK)
    for i, path in enumerate(frame_paths):
        add_frame(ax, path, (0.012 + i * 0.062, 0.067 + i * 0.062, 0.120, 0.355))
    ax.text(0.195, 0.235, "…", fontsize=11, fontweight="bold", color=INK,
            ha="center", va="center")
    ax.text(0.104, 0.372, "untrimmed video", ha="center", va="bottom",
            fontsize=6.4, fontweight="semibold", color=INK)

    add_arrow(ax, (0.210, 0.235), (0.252, 0.235))
    add_box(ax, (0.260, 0.155), 0.205, 0.160,
            r"shared $f_{\theta}$", BLUE, BLUE_BG, fontsize=8.0,
            detail="V-JEPA2.1 → attention → mean → MLP")
    add_arrow(ax, (0.825, 0.635), (0.365, 0.322), GREEN, dashed=True, lw=0.75)
    ax.text(0.540, 0.475, "shared weights", fontsize=5.2, color=GREEN,
            fontweight="semibold", rotation=-17, ha="center")

    add_arrow(ax, (0.473, 0.235), (0.515, 0.270))
    ax.text(0.635, 0.408, "frame-wise labels", ha="center", va="bottom",
            fontsize=6.4, fontweight="semibold", color=INK)
    starts = [0.520, 0.565, 0.620, 0.660, 0.718, 0.790]
    widths = [0.045, 0.055, 0.040, 0.058, 0.072, 0.050]
    for i, (start, width) in enumerate(zip(starts, widths)):
        ax.add_patch(Rectangle((start, 0.245), width, 0.055,
                               facecolor=COLORS[i], edgecolor="none"))
    ax.plot([0.520, 0.840], [0.220, 0.220], color=INK, linewidth=0.7)
    add_arrow(ax, (0.825, 0.220), (0.840, 0.220), lw=0.7)
    ax.text(0.680, 0.188, "time", ha="center", va="top", fontsize=5.2, color=GRAY)
    ax.text(0.680, 0.145, "automatic primitive-skill segments", ha="center",
            va="center", fontsize=5.4, color=GRAY)

    # (c) Quantitative result: compact and explicitly separate from (b).
    ax.plot([0.862, 0.862], [0.085, 0.450], color=LIGHT, linewidth=0.8)
    ax.text(0.875, 0.465, "(c)  Validation", fontsize=8.2, fontweight="bold",
            ha="left", va="top", color=INK)
    ax.text(0.930, 0.350, f"{accuracy * 100:.2f}", ha="center", va="center",
            fontsize=18, fontweight="bold", color=GREEN)
    ax.text(0.977, 0.353, "%", ha="left", va="center", fontsize=7,
            fontweight="bold", color=GREEN)
    ax.text(0.930, 0.295, f"± {accuracy_std * 100:.2f}", ha="center", va="center",
            fontsize=6.5, color=GREEN)
    ax.text(0.930, 0.247, "accuracy", ha="center", va="center", fontsize=6.0,
            fontweight="semibold", color=INK)
    ax.text(0.930, 0.205, "fixed split · 3 seeds", ha="center", va="center",
            fontsize=5.2, color=GRAY)
    # Tiny per-class recall glyphs, useful without turning the figure into a report.
    for i, value in enumerate(recalls):
        x = 0.881 + i * 0.017
        ax.add_patch(Rectangle((x, 0.095), 0.010, 0.075 * value,
                               facecolor=COLORS[i], edgecolor="none"))
    ax.text(0.930, 0.075, "per-class recall", ha="center", va="top",
            fontsize=4.8, color=GRAY)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "global_w4_icra_figure.png"
    pdf = OUT / "global_w4_icra_figure.pdf"
    svg = OUT / "global_w4_icra_figure.svg"
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
