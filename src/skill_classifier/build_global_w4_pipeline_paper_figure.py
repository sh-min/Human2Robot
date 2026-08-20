"""Paper figure in the visual language of the original pipeline illustration."""

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
    SURFACE,
    TEXT,
    arrow,
    centered,
    crop_frame,
    font,
    rounded,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/skill_classifier/kitchen_0813_model_comparison/figures"
FRAME_ROOT = OUTPUT / "source_frames"
RESULT_ROOT = (
    ROOT / "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco"
    / "multiseed_300e"
)


def label_pill(draw, box, text, color, text_font):
    draw.rounded_rectangle(box, radius=10, fill=(255, 255, 255), outline=color, width=2)
    centered(draw, box, text, text_font, color)


def main():
    summaries = [json.loads(path.read_text()) for path in
                 sorted(RESULT_ROOT.glob("global_w4_seed*/evaluation_summary.json"))]
    if len(summaries) != 3:
        raise ValueError(f"expected three Global W4 summaries, found {len(summaries)}")
    accuracy_values = [float(item["validation_accuracy"]) for item in summaries]
    accuracy = statistics.mean(accuracy_values)
    accuracy_std = statistics.stdev(accuracy_values)
    labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")

    canvas = Image.new("RGB", (2400, 1240), (250, 252, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = font(53, True)
    subtitle_font = font(22)
    panel_font = font(20, True)
    card_font = font(26, True)
    body_font = font(18)
    small_font = font(15)
    tiny_font = font(13)

    draw.text((65, 38), "Primitive Skill Training and Long-Horizon Prediction",
              font=title_font, fill=NAVY)
    draw.text((68, 108),
              "Frozen V-JEPA2.1 dense features → learned spatial attention → temporal mean → MLP classifier",
              font=subtitle_font, fill=MUTED)

    # Training panel: the exact Global W4 path.
    rounded(draw, (45, 170, 2355, 695), fill=SURFACE, outline=(175, 194, 224), radius=28)
    rounded(draw, (70, 191, 250, 235), fill=BLUE_SOFT, outline=None, radius=21, width=0)
    centered(draw, (70, 191, 250, 235), "TRAINING", panel_font, BLUE)

    boxes = [(72, 270, 455, 615), (535, 270, 850, 615),
             (930, 270, 1245, 615), (1325, 270, 1790, 615),
             (1870, 270, 2328, 615)]
    for box in boxes:
        rounded(draw, box, fill=(255, 255, 255), outline=BORDER, radius=20)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2] + 12, 440), (right[0] - 12, 440), color=NAVY, width=6)

    # Primitive-labelled real clips.
    draw.text((98, 292), "Primitive-labelled Clips", font=card_font, fill=NAVY)
    draw.text((98, 331), "Human video + skill label", font=body_font, fill=MUTED)
    paths = (FRAME_ROOT / "frame_01_cup.png", FRAME_ROOT / "frame_02_red_snack.png",
             FRAME_ROOT / "frame_03_wipe.png")
    for index, path in enumerate(paths):
        image = crop_frame(path, (100, 135))
        x = 98 + index * 111
        canvas.paste(image, (x, 372))
        draw.rectangle((x, 372, x + 100, 507), outline=NAVY, width=2)
    for index, label in enumerate(labels):
        col, row = index % 3, index // 3
        x, y = 98 + col * 111, 528 + row * 35
        label_pill(draw, (x, y, x + 100, y + 27), label, LABEL_COLORS[index], tiny_font)

    # Frozen V-JEPA.
    draw.text((570, 292), "Frozen V-JEPA2.1", font=card_font, fill=BLUE)
    draw.text((570, 331), "ViT-L/16 · 384 px", font=body_font, fill=MUTED)
    for group in range(4):
        x0 = 573 + group * 62
        for row in range(4):
            for col in range(4):
                x, y = x0 + col * 12, 405 + row * 14
                draw.rectangle((x, y, x + 9, y + 10), fill=BLUE_SOFT, outline=BLUE, width=1)
        draw.text((x0 + 14, 469), f"t{group + 1}", font=tiny_font, fill=MUTED)
    centered(draw, (555, 510, 830, 580), "Dense patch tokens\n[B, 4, S, 1024]", body_font, TEXT)

    # Learned attention.
    draw.text((966, 292), "Spatial Attention", font=card_font, fill=BLUE)
    draw.text((966, 331), "Learned patch weighting", font=body_font, fill=MUTED)
    for token in range(4):
        x0 = 972 + token * 62
        for row in range(4):
            for col in range(4):
                strength = (row * 4 + col + token * 2) % 16
                mix = strength / 22
                shade = tuple(int(BLUE_SOFT[i] * (1 - mix) + BLUE[i] * mix) for i in range(3))
                x, y = x0 + col * 12, 405 + row * 14
                draw.rectangle((x, y, x + 9, y + 10), fill=shade, outline=BLUE, width=1)
    centered(draw, (952, 492, 1223, 580), "softmax over patches\nweighted sum per token", body_font, TEXT)

    # Exact temporal and MLP head.
    draw.text((1360, 292), "Temporal Mean + MLP", font=card_font, fill=GREEN)
    draw.text((1360, 331), "Exact Global W4 head", font=body_font, fill=MUTED)
    layers = [
        ((1355, 397, 1450, 463), "Mean\n4 tokens", BLUE_SOFT, BLUE),
        ((1480, 397, 1575, 463), "Linear\n1024→256", GREEN_SOFT, GREEN),
        ((1605, 397, 1700, 463), "Linear\n256→128", ORANGE_SOFT, ORANGE),
        ((1730, 397, 1765, 463), "6", PURPLE_SOFT, PURPLE),
    ]
    for box, text_value, fill_value, color_value in layers:
        rounded(draw, box, fill=fill_value, outline=color_value, radius=11, width=2)
        centered(draw, box, text_value, font(13, True), color_value)
    for left, right in zip(layers, layers[1:]):
        arrow(draw, (left[0][2] + 3, 430), (right[0][0] - 3, 430), color=GREEN, width=4)
    centered(draw, (1355, 500, 1765, 575),
             "ReLU + Dropout 0.4\nCross-entropy · label smoothing 0.1",
             small_font, MUTED)

    # Six-class output.
    draw.text((1904, 292), "Primitive Skill Prediction", font=card_font, fill=NAVY)
    draw.text((1904, 331), "Illustrative six-class softmax output", font=body_font, fill=MUTED)
    example = (0.04, 0.04, 0.03, 0.78, 0.07, 0.04)
    for index, (label, probability) in enumerate(zip(labels, example)):
        y = 388 + index * 34
        draw.text((1905, y), label, font=small_font, fill=TEXT)
        draw.rounded_rectangle((1985, y + 2, 2235, y + 21), radius=8, fill=(229, 235, 243))
        draw.rounded_rectangle((1985, y + 2, 1985 + max(4, int(250 * probability)), y + 21),
                               radius=8, fill=LABEL_COLORS[index])
        draw.text((2250, y), f"{probability:.2f}", font=small_font, fill=MUTED)

    # Supervision line is visibly separate from the video feature path.
    draw.line((260, 648, 2100, 648), fill=RED, width=3)
    arrow(draw, (2100, 648), (2100, 620), color=RED, width=3)
    draw.text((82, 635), "Primitive GT label", font=small_font, fill=RED)
    draw.text((1640, 655), "supervision only during training", font=tiny_font, fill=RED)

    # Inference panel in the same visual language.
    rounded(draw, (45, 735, 2355, 1185), fill=(255, 253, 249), outline=(229, 204, 166), radius=28)
    rounded(draw, (70, 756, 255, 800), fill=ORANGE_SOFT, outline=None, radius=21, width=0)
    centered(draw, (70, 756, 255, 800), "INFERENCE", panel_font, ORANGE)

    draw.text((90, 830), "Long-Horizon Video", font=card_font, fill=NAVY)
    draw.text((90, 868), "Untrimmed sequence", font=body_font, fill=MUTED)
    for index, path in enumerate(paths):
        image = crop_frame(path, (145, 205))
        x = 90 + index * 157
        canvas.paste(image, (x, 920))
        draw.rectangle((x, 920, x + 145, 1125), outline=NAVY, width=2)
    draw.text((548, 1000), "…", font=font(32, True), fill=NAVY)

    arrow(draw, (590, 1018), (710, 1018), color=ORANGE, width=6)
    rounded(draw, (735, 875, 1240, 1128), fill=BLUE_SOFT, outline=BLUE, radius=20)
    centered(draw, (755, 888, 1220, 1058),
             "Frozen V-JEPA2.1\n→ Spatial attention\n→ 4-token mean → MLP",
             font(24, True), BLUE)
    centered(draw, (755, 1060, 1220, 1115), "Shared trained Global W4 model · RGB only",
             small_font, MUTED)
    arrow(draw, (1580, 615), (980, 860), color=GREEN, width=4, dashed=True)
    draw.text((1265, 760), "shared weights", font=small_font, fill=GREEN)

    arrow(draw, (1265, 1018), (1375, 1018), color=ORANGE, width=6)
    draw.text((1405, 845), "Automatic Skill Segments", font=card_font, fill=NAVY)
    draw.text((1405, 883), "Frame-wise predictions over time", font=body_font, fill=MUTED)
    x, y = 1410, 970
    widths = (70, 82, 62, 90, 145, 90)
    cursor = x
    for index, width in enumerate(widths):
        draw.rectangle((cursor, y, cursor + width, y + 58), fill=LABEL_COLORS[index])
        cursor += width
    draw.line((x, y + 82, x + sum(widths), y + 82), fill=NAVY, width=3)
    draw.polygon(((x + sum(widths), y + 82), (x + sum(widths) - 15, y + 74),
                  (x + sum(widths) - 15, y + 90)), fill=NAVY)
    draw.text((1650, y + 96), "Time", font=small_font, fill=MUTED)

    rounded(draw, (2040, 865, 2305, 1128), fill=GREEN_SOFT, outline=GREEN, radius=20)
    centered(draw, (2060, 890, 2285, 935), "FIXED-SPLIT VALIDATION", tiny_font, GREEN)
    centered(draw, (2060, 940, 2285, 1020), f"{accuracy:.2%}", font(38, True), GREEN)
    centered(draw, (2060, 1015, 2285, 1055), f"± {accuracy_std:.2%}", body_font, GREEN)
    centered(draw, (2060, 1060, 2285, 1105), "3-seed mean", small_font, MUTED)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / "global_w4_pipeline_paper_figure.png"
    pdf = OUTPUT / "global_w4_pipeline_paper_figure.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)
    print(png)


if __name__ == "__main__":
    main()
