"""Build a presentation figure for primitive-skill training and prediction."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/skill_classifier/kitchen_0813_model_comparison/figures"
FRAME_ROOT = OUTPUT / "source_frames"
MULTISEED_ROOT = (
    ROOT / "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco"
    / "multiseed_300e"
)

NAVY = (22, 43, 77)
TEXT = (42, 58, 82)
MUTED = (96, 113, 138)
BORDER = (205, 216, 230)
SURFACE = (255, 255, 255)
BACKGROUND = (246, 249, 253)
BLUE = (49, 105, 230)
BLUE_SOFT = (232, 239, 255)
GREEN = (35, 165, 112)
GREEN_SOFT = (228, 247, 239)
ORANGE = (239, 145, 29)
ORANGE_SOFT = (255, 244, 226)
PURPLE = (133, 85, 205)
PURPLE_SOFT = (242, 235, 252)
RED = (222, 76, 91)
GRAY = (108, 120, 138)
LABEL_COLORS = (BLUE, ORANGE, GREEN, RED, PURPLE, GRAY)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if bold else
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    ]
    return ImageFont.truetype(str(next(path for path in candidates if path.is_file())), size)


def rounded(draw: ImageDraw.ImageDraw, box, fill=SURFACE, outline=BORDER, radius=24, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start, end, color=BLUE, width=8, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        length = max(abs(x2 - x1), abs(y2 - y1))
        for offset in range(0, length, 24):
            t1 = offset / max(length, 1)
            t2 = min(offset + 13, length) / max(length, 1)
            draw.line((x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1,
                       x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2),
                      fill=color, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        points = [(x2, y2), (x2 - direction * 24, y2 - 16),
                  (x2 - direction * 24, y2 + 16)]
    else:
        direction = 1 if y2 > y1 else -1
        points = [(x2, y2), (x2 - 16, y2 - direction * 24),
                  (x2 + 16, y2 - direction * 24)]
    draw.polygon(points, fill=color)


def centered(draw, box, text, text_font, fill=TEXT):
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=8, align="center")
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.multiline_text(((left + right - width) / 2, (top + bottom - height) / 2),
                        text, font=text_font, fill=fill, spacing=8, align="center")


def crop_frame(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    target_ratio = size[0] / size[1]
    ratio = image.width / image.height
    if ratio > target_ratio:
        crop_w = int(image.height * target_ratio)
        left = (image.width - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, image.height))
    else:
        crop_h = int(image.width / target_ratio)
        top = (image.height - crop_h) // 2
        image = image.crop((0, top, image.width, top + crop_h))
    return image.resize(size, Image.Resampling.LANCZOS)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = [
        json.loads(path.read_text())
        for path in sorted(MULTISEED_ROOT.glob("global_w4_seed*/evaluation_summary.json"))
    ]
    if len(summaries) != 3:
        raise ValueError(f"expected three Global W4 seed summaries, found {len(summaries)}")
    accuracies = [float(summary["validation_accuracy"]) for summary in summaries]
    accuracy = statistics.mean(accuracies)
    accuracy_std = statistics.stdev(accuracies)
    metric_labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")
    per_class_accuracy = tuple(
        statistics.mean(float(summary["per_class"][label]["accuracy"]) for summary in summaries)
        for label in metric_labels
    )

    canvas = Image.new("RGB", (2400, 1350), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title = font(58, True)
    subtitle = font(25)
    section = font(23, True)
    card_title = font(28, True)
    body = font(21)
    small = font(18)
    small_bold = font(18, True)

    draw.text((70, 42), "V-JEPA2.1 Spatial-Attention Skill Classifier",
              font=title, fill=NAVY)
    draw.text((73, 116),
              "Exact 94.74% model: frozen dense features, learned patch attention, temporal mean, and MLP.",
              font=subtitle, fill=MUTED)

    # Training lane.
    rounded(draw, (55, 180, 2345, 660), fill=(251, 253, 255), outline=(189, 207, 234), radius=30)
    rounded(draw, (78, 202, 288, 248), fill=BLUE_SOFT, outline=None, radius=23, width=0)
    centered(draw, (78, 202, 288, 248), "TRAINING FLOW", section, BLUE)

    train_boxes = [(85, 275, 550, 610), (650, 275, 1050, 610),
                   (1150, 275, 1575, 610), (1675, 275, 2315, 610)]
    for box in train_boxes:
        rounded(draw, box)
    for left, right in zip(train_boxes, train_boxes[1:]):
        arrow(draw, (left[2] + 16, 442), (right[0] - 16, 442))

    draw.text((115, 300), "Primitive-labelled Clips", font=card_title, fill=NAVY)
    draw.text((115, 343), "Human-labelled training videos", font=body, fill=MUTED)
    labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")
    for index, label in enumerate(labels):
        column, row = index % 2, index // 2
        x, y = 115 + column * 210, 395 + row * 61
        color = LABEL_COLORS[index]
        draw.rounded_rectangle((x, y, x + 190, y + 42), radius=13,
                               fill=tuple(int(255 - (255 - c) * 0.13) for c in color),
                               outline=color, width=2)
        centered(draw, (x, y, x + 190, y + 42), label, small_bold, color)

    draw.text((685, 300), "Frozen V-JEPA2.1", font=card_title, fill=BLUE)
    draw.text((685, 343), "ViT-L/16 · 384 px encoder", font=body, fill=MUTED)
    for group in range(3):
        origin_x = 705 + group * 103
        for row in range(4):
            for column in range(4):
                x, y = origin_x + column * 17 + row * 3, 412 + row * 18
                draw.rectangle((x, y, x + 13, y + 13), fill=BLUE_SOFT,
                               outline=BLUE, width=1)
    centered(draw, (675, 510, 1025, 580), "Dense patch tokens\n[B, 4, S, 1024]", small, TEXT)

    draw.text((1185, 300), "Learned Spatial Attention", font=card_title, fill=NAVY)
    draw.text((1185, 343), "Content attention over patches", font=body, fill=MUTED)
    for token in range(4):
        x0 = 1190 + token * 83
        for row in range(3):
            for column in range(3):
                strength = (row * 3 + column + token) % 9
                shade = tuple(int(BLUE_SOFT[i] * (1 - strength / 28) + BLUE[i] * strength / 28)
                              for i in range(3))
                draw.rectangle((x0 + column * 17, 408 + row * 17,
                                x0 + column * 17 + 13, 408 + row * 17 + 13),
                               fill=shade, outline=BLUE, width=1)
        draw.text((x0 + 9, 475), f"t{token + 1}", font=font(14), fill=MUTED)
    centered(draw, (1175, 510, 1550, 585),
             "Softmax patch weights\n→ weighted spatial pooling", small, TEXT)

    draw.text((1710, 300), "Temporal Mean + MLP", font=card_title, fill=GREEN)
    draw.text((1710, 343), "Exact trained classification head", font=body, fill=MUTED)
    layer_boxes = [
        ((1710, 402, 1820, 454), "Mean\n4 tokens", BLUE_SOFT, BLUE),
        ((1850, 402, 1960, 454), "Linear\n1024→256", GREEN_SOFT, GREEN),
        ((1990, 402, 2100, 454), "Linear\n256→128", ORANGE_SOFT, ORANGE),
        ((2130, 402, 2240, 454), "Linear\n128→6", PURPLE_SOFT, PURPLE),
    ]
    for box, text_value, fill_value, color_value in layer_boxes:
        rounded(draw, box, fill=fill_value, outline=color_value, radius=13, width=2)
        centered(draw, box, text_value, font(14, True), color_value)
    for left, right in zip(layer_boxes, layer_boxes[1:]):
        arrow(draw, (left[0][2] + 4, 428), (right[0][0] - 4, 428), color=GREEN, width=4)
    draw.text((1710, 480), "ReLU + Dropout 0.4 after hidden layers", font=small, fill=MUTED)
    for index, label in enumerate(labels):
        column, row = index % 3, index // 3
        x, y = 1710 + column * 170, 526 + row * 34
        draw.ellipse((x, y + 4, x + 16, y + 20), fill=LABEL_COLORS[index])
        draw.text((x + 25, y), label, font=small, fill=TEXT)
    draw.text((1710, 590), "Cross-entropy · label smoothing 0.1 · balanced sampling",
              font=font(14), fill=MUTED)

    # Inference lane.
    rounded(draw, (55, 710, 2345, 1285), fill=(255, 253, 249), outline=(235, 210, 172), radius=30)
    rounded(draw, (78, 732, 300, 778), fill=ORANGE_SOFT, outline=None, radius=23, width=0)
    centered(draw, (78, 732, 300, 778), "INFERENCE FLOW", section, ORANGE)

    infer_boxes = [(85, 825, 670, 1225), (770, 825, 1175, 1225),
                   (1275, 825, 1710, 1225), (1810, 825, 2315, 1225)]
    for box in infer_boxes:
        rounded(draw, box)
    for left, right in zip(infer_boxes, infer_boxes[1:]):
        arrow(draw, (left[2] + 16, 1025), (right[0] - 16, 1025), color=ORANGE)

    draw.text((120, 850), "Long-Horizon Video", font=card_title, fill=NAVY)
    draw.text((120, 893), "Untrimmed sequence with multiple skills", font=body, fill=MUTED)
    frame_paths = [FRAME_ROOT / "frame_01_cup.png", FRAME_ROOT / "frame_02_red_snack.png",
                   FRAME_ROOT / "frame_03_wipe.png"]
    thumbs = [crop_frame(path, (164, 205)) for path in frame_paths]
    for index, thumb in enumerate(thumbs):
        x = 116 + index * 174
        canvas.paste(thumb, (x, 958))
        draw.rectangle((x, 958, x + 164, 1163), outline=NAVY, width=3)
    draw.text((622, 1040), "…", font=font(34, True), fill=NAVY)

    draw.text((805, 850), "Trained Global Model", font=card_title, fill=NAVY)
    draw.text((805, 893), "Apply the trained model", font=body, fill=MUTED)
    rounded(draw, (825, 970, 1120, 1080), fill=BLUE_SOFT, outline=BLUE, radius=18)
    centered(draw, (825, 970, 1120, 1080),
             "V-JEPA2.1 → attention\n→ 4-token mean → MLP", card_title, BLUE)
    draw.text((812, 1122), "Global RGB only · no object, text, or hand branch",
              font=font(15), fill=MUTED)
    arrow(draw, (2155, 610), (1000, 805), color=GREEN, width=6, dashed=True)

    draw.text((1310, 850), "Validation Results", font=card_title, fill=NAVY)
    draw.text((1310, 893), "Per-class recall · 3-seed mean", font=body, fill=MUTED)
    for index, probability in enumerate(per_class_accuracy):
        y = 952 + index * 42
        draw.text((1310, y), labels[index], font=small, fill=TEXT)
        draw.rounded_rectangle((1428, y + 3, 1658, y + 24), radius=10, fill=(231, 236, 243))
        draw.rounded_rectangle((1428, y + 3, 1428 + max(5, int(230 * probability)), y + 24),
                               radius=10, fill=LABEL_COLORS[index])
        draw.text((1668, y), f"{probability:.2f}", font=small, fill=MUTED)

    draw.text((1845, 850), "Long-Horizon Prediction", font=card_title, fill=NAVY)
    draw.text((1845, 893), "Frame labels → primitive skill segments", font=body, fill=MUTED)
    track_x, track_y, track_w = 1850, 975, 430
    widths = (58, 75, 55, 72, 106, 64)
    cursor = track_x
    for index, width in enumerate(widths):
        draw.rectangle((cursor, track_y, cursor + width, track_y + 54), fill=LABEL_COLORS[index])
        cursor += width
    draw.line((track_x, track_y + 72, track_x + track_w, track_y + 72), fill=NAVY, width=3)
    draw.polygon(((track_x + track_w, track_y + 72), (track_x + track_w - 16, track_y + 64),
                  (track_x + track_w - 16, track_y + 80)), fill=NAVY)
    draw.text((track_x + 180, track_y + 84), "Time", font=small, fill=MUTED)
    rounded(draw, (1850, 1100, 2280, 1198), fill=GREEN_SOFT, outline=GREEN, radius=18)
    draw.text((1875, 1114), "FIXED-SPLIT VALIDATION", font=font(15, True), fill=GREEN)
    draw.text((1875, 1162), "3-seed mean", font=font(14), fill=MUTED)
    value = f"{accuracy:.2%}"
    value_font = font(35, True)
    value_width = draw.textbbox((0, 0), value, font=value_font)[2]
    draw.text((2255 - value_width, 1110), value, font=value_font, fill=GREEN)
    std_text = f"± {accuracy_std:.2%}"
    std_width = draw.textbbox((0, 0), std_text, font=font(14))[2]
    draw.text((2255 - std_width, 1162), std_text, font=font(14), fill=GREEN)

    output = OUTPUT / "primitive_skill_long_horizon_pipeline.png"
    canvas.save(output, quality=95)
    print(output)


if __name__ == "__main__":
    main()
