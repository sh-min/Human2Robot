"""Rebuild Global W4 content in the original annotation-pipeline visual mood."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

from skill_classifier.build_primitive_skill_pipeline_figure import (
    BLUE,
    BLUE_SOFT,
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
    crop_frame,
    font,
    rounded,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/skill_classifier/kitchen_0813_model_comparison/figures"
FRAMES = OUT / "source_frames"
RESULTS = (
    ROOT / "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco"
    / "multiseed_300e"
)
CARD_BORDER = (22, 42, 78)
CARD_FILL = (253, 254, 255)
BG = (252, 253, 255)


def card(draw, box):
    rounded(draw, box, fill=CARD_FILL, outline=CARD_BORDER, radius=18, width=3)


def chip(draw, box, label, color, text_font):
    draw.rounded_rectangle(box, radius=9, fill=(255, 255, 255), outline=color, width=2)
    centered(draw, box, label, text_font, color)


def main():
    summaries = [json.loads(path.read_text()) for path in
                 sorted(RESULTS.glob("global_w4_seed*/evaluation_summary.json"))]
    if len(summaries) != 3:
        raise ValueError(f"expected three Global W4 results, found {len(summaries)}")
    values = [float(item["validation_accuracy"]) for item in summaries]
    accuracy = statistics.mean(values)
    accuracy_std = statistics.stdev(values)
    f1 = statistics.mean(float(item["validation_f1_macro"]) for item in summaries)
    f1_std = statistics.stdev(float(item["validation_f1_macro"]) for item in summaries)
    labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")

    canvas = Image.new("RGB", (2400, 1350), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = font(62, True)
    module_title = font(26, True)
    compact_title = font(21, True)
    body = font(19)
    small = font(15)
    tiny = font(13)

    title = "V-JEPA2.1 Primitive Skill Annotation Pipeline"
    title_w = draw.textbbox((0, 0), title, font=title_font)[2]
    draw.text(((2400 - title_w) / 2, 35), title, font=title_font, fill=NAVY)

    # Main row follows the original figure: independent white cards connected by
    # dark navy arrows, with real frames and compact technical illustrations.
    y1, y2 = 230, 845
    boxes = [
        (25, y1, 405, y2),
        (485, y1, 805, y2),
        (885, y1, 1205, y2),
        (1285, y1, 1645, y2),
        (1725, y1, 2010, y2),
        (2090, y1, 2375, y2),
    ]
    for box in boxes:
        card(draw, box)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2] + 10, 535), (right[0] - 10, 535), color=NAVY, width=7)

    # 1. Primitive-labelled training clips.
    centered(draw, (45, 255, 385, 315), "Primitive-labelled Clips", module_title, NAVY)
    centered(draw, (45, 310, 385, 350), "Human demonstration + skill label", body, MUTED)
    paths = (FRAMES / "frame_01_cup.png", FRAMES / "frame_02_red_snack.png",
             FRAMES / "frame_03_wipe.png")
    for index, path in enumerate(paths):
        thumb = crop_frame(path, (104, 220))
        x = 46 + index * 116
        canvas.paste(thumb, (x, 382))
        draw.rectangle((x, 382, x + 104, 602), outline=NAVY, width=2)
    draw.text((367, 475), "…", font=font(28, True), fill=NAVY)
    for index, label in enumerate(labels):
        col, row = index % 3, index // 3
        x, y = 45 + col * 116, 640 + row * 42
        chip(draw, (x, y, x + 104, y + 31), label, LABEL_COLORS[index], tiny)
    centered(draw, (45, 742, 385, 812), "Primitive skill supervision", small, RED)

    # 2. Frozen V-JEPA2.1 encoder.
    centered(draw, (505, 255, 785, 315), "A. Frozen V-JEPA2.1", module_title, BLUE)
    centered(draw, (505, 310, 785, 350), "ViT-L/16 · 384 px", body, MUTED)
    for group in range(4):
        x0 = 530 + group * 63
        for row in range(5):
            for col in range(5):
                x, y = x0 + col * 10, 420 + row * 12
                draw.rectangle((x, y, x + 7, y + 8), fill=BLUE_SOFT, outline=BLUE, width=1)
        draw.text((x0 + 16, 488), f"t{group + 1}", font=tiny, fill=MUTED)
    centered(draw, (510, 545, 780, 635), "Dense patch tokens\n[B, 4, S, 1024]", body, TEXT)
    centered(draw, (510, 690, 780, 765), "Encoder weights frozen", small, BLUE)

    # 3. Spatial attention.
    centered(draw, (905, 255, 1185, 315), "B. Spatial Attention", module_title, BLUE)
    centered(draw, (905, 310, 1185, 350), "Learned patch weighting", body, MUTED)
    for token in range(4):
        x0 = 930 + token * 63
        for row in range(5):
            for col in range(5):
                strength = (row * 5 + col + token * 4) % 25
                mix = strength / 32
                shade = tuple(int(BLUE_SOFT[i] * (1 - mix) + BLUE[i] * mix) for i in range(3))
                x, y = x0 + col * 10, 420 + row * 12
                draw.rectangle((x, y, x + 7, y + 8), fill=shade, outline=BLUE, width=1)
    centered(draw, (910, 540, 1180, 650),
             "softmax over S patches\nweighted spatial pooling", body, TEXT)
    centered(draw, (910, 690, 1180, 765), "Attention query trainable", small, GREEN)

    # 4. Exact classifier head.
    centered(draw, (1305, 255, 1625, 315), "C. Temporal Mean + MLP", module_title, GREEN)
    centered(draw, (1305, 310, 1625, 350), "Exact Global W4 classification head", body, MUTED)
    layers = [
        ((1315, 420, 1400, 505), "Mean\n4 tokens", BLUE_SOFT, BLUE),
        ((1430, 420, 1515, 505), "Linear\n1024→256", GREEN_SOFT, GREEN),
        ((1545, 420, 1605, 505), "Linear\n256→128", ORANGE_SOFT, ORANGE),
        ((1618, 420, 1632, 505), "6", PURPLE_SOFT, PURPLE),
    ]
    for layer_box, text_value, fill_value, outline_value in layers:
        rounded(draw, layer_box, fill=fill_value, outline=outline_value, radius=10, width=2)
        centered(draw, layer_box, text_value, font(11, True), outline_value)
    for left, right in zip(layers, layers[1:]):
        arrow(draw, (left[0][2] + 2, 462), (right[0][0] - 2, 462), color=GREEN, width=3)
    centered(draw, (1310, 555, 1620, 635), "ReLU + Dropout 0.4", body, TEXT)
    centered(draw, (1310, 650, 1620, 745),
             "Cross-entropy\nlabel smoothing 0.1", small, RED)

    # 5. Frame-wise prediction card.
    centered(draw, (1743, 255, 1992, 315), "Frame-wise Prediction", compact_title, NAVY)
    centered(draw, (1743, 310, 1992, 350), "6 primitive-skill probabilities", body, MUTED)
    example = (0.04, 0.04, 0.03, 0.78, 0.07, 0.04)
    for index, (label, probability) in enumerate(zip(labels, example)):
        y = 400 + index * 53
        draw.text((1748, y), label, font=small, fill=TEXT)
        draw.rounded_rectangle((1818, y + 1, 1952, y + 22), radius=8, fill=(230, 235, 242))
        draw.rounded_rectangle((1818, y + 1, 1818 + max(4, int(134 * probability)), y + 22),
                               radius=8, fill=LABEL_COLORS[index])
        draw.text((1962, y), f"{probability:.2f}", font=tiny, fill=MUTED)
    centered(draw, (1745, 740, 1990, 810), "Illustrative softmax output", tiny, MUTED)

    # 6. Automatic annotation card.
    centered(draw, (2108, 255, 2357, 315), "Automatic Annotation", compact_title, NAVY)
    centered(draw, (2108, 310, 2357, 350), "Frame labels → skill segments", body, MUTED)
    x, y = 2112, 475
    widths = (36, 42, 30, 48, 62, 42)
    cursor = x
    for index, width in enumerate(widths):
        draw.rectangle((cursor, y, cursor + width, y + 62), fill=LABEL_COLORS[index])
        cursor += width
    draw.line((x, y + 90, x + sum(widths), y + 90), fill=NAVY, width=3)
    draw.polygon(((x + sum(widths), y + 90), (x + sum(widths) - 13, y + 82),
                  (x + sum(widths) - 13, y + 98)), fill=NAVY)
    centered(draw, (2110, 590, 2355, 640), "Time", small, MUTED)
    centered(draw, (2110, 690, 2355, 790), "Long-horizon\nprimitive skill sequence", body, TEXT)

    # Original-style dashed evaluation strip.
    eval_box = (35, 945, 2365, 1305)
    draw.rounded_rectangle(eval_box, radius=22, fill=(255, 255, 255),
                           outline=NAVY, width=3)
    # Dashed overlay on top of the solid outline reproduces the reference mood.
    for x in range(58, 2342, 24):
        draw.line((x, 945, min(x + 12, 2342), 945), fill=NAVY, width=4)
        draw.line((x, 1305, min(x + 12, 2342), 1305), fill=NAVY, width=4)
    for y in range(968, 1282, 24):
        draw.line((35, y, 35, min(y + 12, 1282)), fill=NAVY, width=4)
        draw.line((2365, y, 2365, min(y + 12, 1282)), fill=NAVY, width=4)

    draw.text((78, 980), "Long-Horizon Inference", font=module_title, fill=NAVY)
    for index, path in enumerate(paths):
        thumb = crop_frame(path, (115, 175))
        x0 = 78 + index * 126
        canvas.paste(thumb, (x0, 1050))
        draw.rectangle((x0, 1050, x0 + 115, 1225), outline=NAVY, width=2)
    draw.text((454, 1120), "…", font=font(26, True), fill=NAVY)

    arrow(draw, (500, 1135), (650, 1135), color=NAVY, width=7)
    rounded(draw, (685, 1010, 1120, 1255), fill=BLUE_SOFT, outline=BLUE, radius=18)
    centered(draw, (705, 1025, 1100, 1180),
             "Shared trained model\nV-JEPA2.1 → attention\n→ 4-token mean → MLP",
             font(22, True), BLUE)
    centered(draw, (705, 1185, 1100, 1238), "RGB only · no object/text/hand branch", tiny, MUTED)

    arrow(draw, (1155, 1135), (1310, 1135), color=NAVY, width=7)
    draw.text((1345, 990), "Predicted Primitive Segments", font=module_title, fill=NAVY)
    cursor, track_y = 1348, 1100
    for index, width in enumerate((70, 82, 62, 90, 142, 90)):
        draw.rectangle((cursor, track_y, cursor + width, track_y + 56), fill=LABEL_COLORS[index])
        cursor += width
    draw.line((1348, track_y + 82, cursor, track_y + 82), fill=NAVY, width=3)
    draw.polygon(((cursor, track_y + 82), (cursor - 14, track_y + 74),
                  (cursor - 14, track_y + 90)), fill=NAVY)

    draw.line((1970, 990, 1970, 1265), fill=(198, 207, 221), width=3)
    centered(draw, (2000, 980, 2330, 1030), "FIXED-SPLIT VALIDATION", small, GREEN)
    centered(draw, (2000, 1040, 2330, 1130), f"Accuracy  {accuracy:.2%} ± {accuracy_std:.2%}",
             font(25, True), GREEN)
    centered(draw, (2000, 1135, 2330, 1195), f"Macro-F1  {f1:.2%} ± {f1_std:.2%}",
             font(19, True), BLUE)
    centered(draw, (2000, 1200, 2330, 1250), "3-seed mean · 247 validation samples", tiny, MUTED)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "global_w4_original_mood_figure.png"
    pdf = OUT / "global_w4_original_mood_figure.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, "PDF", resolution=300.0)
    print(png)


if __name__ == "__main__":
    main()
