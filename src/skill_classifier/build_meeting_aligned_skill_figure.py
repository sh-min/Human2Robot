"""Meeting-aligned automatic skill annotation figure in the original visual mood."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

from skill_classifier.build_primitive_skill_pipeline_figure import (
    BLUE,
    BLUE_SOFT,
    BORDER,
    GREEN,
    GREEN_SOFT,
    LABEL_COLORS,
    MUTED,
    NAVY,
    ORANGE,
    ORANGE_SOFT,
    PURPLE,
    PURPLE_SOFT,
    RED,
    TEXT,
    arrow,
    centered,
    font,
    rounded,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/skill_classifier/kitchen_0813_model_comparison/figures"
RESULT_ROOT = (
    ROOT / "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco"
    / "multiseed_300e"
)
RAW = ROOT / "data/robot_overlay/kitchen_0724_0728/0724/IMG_5019/rgb_hawor/extracted_images"
FRAME_SPECS = (("0094.jpg", "Cup"), ("0154.jpg", "Lock"),
               ("0229.jpg", "Snack"), ("0304.jpg", "Sweep"))


def paper_card(draw, box, outline=(27, 48, 84), fill=(255, 255, 255), radius=20, width=3):
    rounded(draw, box, fill=fill, outline=outline, radius=radius, width=width)


def load_landscape(path: Path, size: tuple[int, int]):
    image = Image.open(path).convert("RGB")
    if image.height > image.width:
        image = image.rotate(90, expand=True)
    target_ratio = size[0] / size[1]
    ratio = image.width / image.height
    if ratio > target_ratio:
        width = int(image.height * target_ratio)
        left = (image.width - width) // 2
        image = image.crop((left, 0, left + width, image.height))
    else:
        height = int(image.width / target_ratio)
        top = (image.height - height) // 2
        image = image.crop((0, top, image.width, top + height))
    return image.resize(size, Image.Resampling.LANCZOS)


def pill(draw, box, text, color, text_font):
    draw.rounded_rectangle(box, radius=10, fill=(255, 255, 255), outline=color, width=2)
    centered(draw, box, text, text_font, color)


def main():
    summaries = [json.loads(path.read_text()) for path in
                 sorted(RESULT_ROOT.glob("global_w4_seed*/evaluation_summary.json"))]
    if len(summaries) != 3:
        raise ValueError(f"expected three Global W4 summaries, found {len(summaries)}")
    labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")
    accuracy_values = [float(item["validation_accuracy"]) for item in summaries]
    accuracy = statistics.mean(accuracy_values)
    accuracy_std = statistics.stdev(accuracy_values)
    recalls = tuple(
        statistics.mean(float(item["per_class"][label]["accuracy"]) for item in summaries)
        for label in labels
    )

    canvas = Image.new("RGB", (2400, 1200), (249, 251, 254))
    draw = ImageDraw.Draw(canvas)
    title = font(54, True)
    subtitle = font(21)
    section = font(16, True)
    card_title = font(25, True)
    body = font(18)
    small = font(15)
    tiny = font(12)

    draw.text((65, 38), "Automatic Primitive Skill Annotation from Long-Horizon Video",
              font=title, fill=NAVY)
    draw.text((68, 105),
              "Primitive-labelled demonstrations train a V-JEPA2.1 skill classifier for frame-wise long-horizon segmentation.",
              font=subtitle, fill=MUTED)

    # Left: meeting-requested distinction between supervision and long-horizon input.
    train_box = (55, 190, 565, 535)
    infer_box = (55, 655, 565, 1030)
    paper_card(draw, train_box)
    paper_card(draw, infer_box)
    rounded(draw, (78, 210, 245, 250), fill=BLUE_SOFT, outline=None, radius=19, width=0)
    centered(draw, (78, 210, 245, 250), "TRAINING INPUT", section, BLUE)
    draw.text((80, 275), "Primitive-labelled Clips", font=card_title, fill=NAVY)
    draw.text((80, 311), "Human demonstrations with six skill labels", font=body, fill=MUTED)
    for index, (name, _) in enumerate(FRAME_SPECS[:3]):
        frame = load_landscape(RAW / name, (142, 106))
        x = 80 + index * 156
        canvas.paste(frame, (x, 350))
        draw.rectangle((x, 350, x + 142, 456), outline=NAVY, width=2)
    for index, label in enumerate(labels):
        x = 80 + (index % 3) * 156
        y = 477 + (index // 3) * 31
        pill(draw, (x, y, x + 142, y + 24), label, LABEL_COLORS[index], tiny)

    rounded(draw, (78, 675, 255, 715), fill=ORANGE_SOFT, outline=None, radius=19, width=0)
    centered(draw, (78, 675, 255, 715), "INFERENCE INPUT", section, ORANGE)
    draw.text((80, 739), "Long-Horizon Video", font=card_title, fill=NAVY)
    draw.text((80, 775), "Untrimmed sequence containing multiple skills", font=body, fill=MUTED)
    for index, (name, label) in enumerate(FRAME_SPECS):
        frame = load_landscape(RAW / name, (108, 154))
        x = 80 + index * 116
        canvas.paste(frame, (x, 820))
        draw.rectangle((x, 820, x + 108, 974), outline=NAVY, width=2)
        draw.rectangle((x, 950, x + 108, 974), fill=LABEL_COLORS[labels.index(label)])
        centered(draw, (x, 950, x + 108, 974), label, tiny, (255, 255, 255))
    draw.text((535, 880), "…", font=font(26, True), fill=NAVY)

    # Central shared model: one coherent card instead of many tall boxes.
    model_box = (700, 225, 1440, 1000)
    paper_card(draw, model_box, outline=BLUE, fill=(253, 254, 255), radius=24)
    rounded(draw, (730, 250, 914, 292), fill=BLUE_SOFT, outline=None, radius=20, width=0)
    centered(draw, (730, 250, 914, 292), "SHARED MODEL", section, BLUE)
    draw.text((730, 320), "Global W4 Skill Classifier", font=font(30, True), fill=NAVY)
    draw.text((730, 366), "RGB-only model · no object, text, or hand branch", font=body, fill=MUTED)

    # V-JEPA tokenization.
    sub_boxes = [(738, 430, 970, 670), (1015, 430, 1247, 670),
                 (738, 735, 970, 925), (1015, 735, 1400, 925)]
    sub_styles = ((BLUE, BLUE_SOFT), (BLUE, BLUE_SOFT),
                  (GREEN, GREEN_SOFT), (ORANGE, ORANGE_SOFT))
    for box, (outline, fill) in zip(sub_boxes, sub_styles):
        rounded(draw, box, fill=fill, outline=outline, radius=16, width=2)

    centered(draw, (750, 448, 958, 500), "Frozen V-JEPA2.1", font(19, True), BLUE)
    centered(draw, (750, 500, 958, 535), "ViT-L/16 · 384 px", small, MUTED)
    for token in range(4):
        x0 = 765 + token * 45
        for row in range(3):
            for col in range(3):
                x, y = x0 + col * 8, 565 + row * 10
                draw.rectangle((x, y, x + 5, y + 6), fill=(255, 255, 255), outline=BLUE, width=1)
    centered(draw, (750, 610, 958, 654), "[B, 4, S, 1024]", tiny, TEXT)

    centered(draw, (1027, 448, 1235, 500), "Spatial Attention", font(19, True), BLUE)
    centered(draw, (1027, 500, 1235, 535), "learned patch weights", small, MUTED)
    for token in range(4):
        x0 = 1042 + token * 45
        for row in range(3):
            for col in range(3):
                strength = (row * 3 + col + token * 2) % 9
                mix = strength / 14
                color = tuple(int(BLUE_SOFT[i] * (1 - mix) + BLUE[i] * mix) for i in range(3))
                x, y = x0 + col * 8, 565 + row * 10
                draw.rectangle((x, y, x + 5, y + 6), fill=color, outline=BLUE, width=1)
    centered(draw, (1027, 610, 1235, 654), "softmax over S", tiny, TEXT)
    arrow(draw, (976, 550), (1009, 550), color=NAVY, width=4)

    centered(draw, (750, 757, 958, 810), "Temporal Mean", font(19, True), GREEN)
    centered(draw, (750, 812, 958, 872), "mean over 4 tokens", small, MUTED)
    centered(draw, (750, 866, 958, 910), "1/4 Σ", font(22, True), GREEN)

    centered(draw, (1027, 757, 1388, 810), "MLP Classification Head", font(19, True), ORANGE)
    centered(draw, (1027, 815, 1388, 865), "1024 → 256 → 128 → 6", font(18, True), ORANGE)
    centered(draw, (1027, 866, 1388, 908), "ReLU · Dropout 0.4", tiny, MUTED)
    arrow(draw, (970, 830), (1009, 830), color=NAVY, width=4)
    arrow(draw, (1130, 676), (1130, 726), color=NAVY, width=4)

    # Both flows enter the same architecture; labels only supervise training.
    arrow(draw, (577, 360), (690, 490), color=BLUE, width=6)
    draw.line((577, 840, 595, 840, 595, 570), fill=ORANGE, width=6)
    arrow(draw, (595, 570), (620, 570), color=ORANGE, width=6)
    rounded(draw, (820, 947, 1215, 982), fill=(255, 241, 243), outline=RED,
            radius=14, width=2)
    centered(draw, (820, 947, 1215, 982),
             "Primitive labels → cross-entropy (training only)", tiny, RED)

    # Right: actual result values requested in the meeting, then automatic segments.
    results_box = (1540, 205, 1945, 1005)
    annotation_box = (2020, 300, 2355, 900)
    paper_card(draw, results_box)
    paper_card(draw, annotation_box)
    arrow(draw, (1452, 605), (1528, 605), color=NAVY, width=7)
    arrow(draw, (1957, 605), (2008, 605), color=NAVY, width=7)

    draw.text((1570, 235), "Primitive Skill Prediction", font=card_title, fill=NAVY)
    draw.text((1570, 274), "Fixed-split validation · 3-seed mean", font=small, fill=MUTED)
    centered(draw, (1570, 315, 1915, 400), f"{accuracy:.2%}", font(42, True), GREEN)
    centered(draw, (1570, 392, 1915, 430), f"Accuracy  ± {accuracy_std:.2%}", small, GREEN)
    draw.line((1570, 450, 1915, 450), fill=BORDER, width=2)
    draw.text((1570, 475), "Per-skill recall", font=font(17, True), fill=NAVY)
    for index, (label, value) in enumerate(zip(labels, recalls)):
        y = 525 + index * 65
        draw.text((1572, y), label, font=small, fill=TEXT)
        draw.rounded_rectangle((1648, y + 1, 1855, y + 23), radius=8, fill=(231, 236, 243))
        draw.rounded_rectangle((1648, y + 1, 1648 + int(207 * value), y + 23),
                               radius=8, fill=LABEL_COLORS[index])
        draw.text((1865, y), f"{value:.1%}", font=small, fill=MUTED)

    draw.text((2050, 330), "Automatic Annotation", font=card_title, fill=NAVY)
    draw.text((2050, 370), "Frame predictions → primitive segments", font=small, fill=MUTED)
    x, y = 2055, 500
    widths = (43, 50, 36, 58, 78, 52)
    cursor = x
    for index, width in enumerate(widths):
        draw.rectangle((cursor, y, cursor + width, y + 66), fill=LABEL_COLORS[index])
        cursor += width
    draw.line((x, y + 95, cursor, y + 95), fill=NAVY, width=3)
    draw.polygon(((cursor, y + 95), (cursor - 14, y + 87), (cursor - 14, y + 103)), fill=NAVY)
    centered(draw, (2050, 610, 2325, 650), "Time", small, MUTED)
    for index, label in enumerate(labels):
        col, row = index % 2, index // 2
        x0, y0 = 2055 + col * 150, 705 + row * 43
        draw.ellipse((x0, y0 + 3, x0 + 16, y0 + 19), fill=LABEL_COLORS[index])
        draw.text((x0 + 25, y0), label, font=small, fill=TEXT)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "meeting_aligned_automatic_skill_annotation.png"
    pdf = OUT / "meeting_aligned_automatic_skill_annotation.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)
    print(png)


if __name__ == "__main__":
    main()
