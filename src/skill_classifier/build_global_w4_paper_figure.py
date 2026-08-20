"""Create a compact paper-style figure for the exact Global W4 model."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/skill_classifier/kitchen_0813_model_comparison/figures"
FRAMES = OUT / "source_frames"
RESULTS = (
    ROOT / "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco"
    / "multiseed_300e"
)

INK = (25, 32, 44)
GRAY = (94, 105, 120)
LIGHT = (236, 239, 244)
BLUE = (46, 101, 190)
BLUE_BG = (234, 241, 251)
GREEN = (40, 145, 100)
GREEN_BG = (233, 247, 240)
ORANGE = (220, 132, 38)
ORANGE_BG = (252, 242, 229)
PURPLE = (126, 83, 178)
RED = (205, 72, 84)
CLASS_COLORS = (BLUE, ORANGE, GREEN, RED, PURPLE, (104, 116, 132))


def get_font(size: int, bold: bool = False):
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size)


def center_text(draw, box, text, fnt, fill=INK, spacing=5):
    x1, y1, x2, y2 = box
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.multiline_text(((x1 + x2 - w) / 2, (y1 + y2 - h) / 2), text,
                        font=fnt, fill=fill, spacing=spacing, align="center")


def module(draw, box, title, detail="", fill=(255, 255, 255), outline=INK):
    draw.rounded_rectangle(box, radius=12, fill=fill, outline=outline, width=2)
    x1, y1, x2, y2 = box
    # Reserve separate title, operation, and metadata zones so architecture
    # symbols placed by the caller never collide with their module names.
    center_text(draw, (x1 + 10, y1 + 12, x2 - 10, y1 + 78),
                title, get_font(18, True), outline)
    if detail:
        center_text(draw, (x1 + 8, y2 - 34, x2 - 8, y2 - 5), detail, get_font(13), GRAY)


def arrow(draw, start, end, color=INK, width=3, dashed=False):
    x1, y1 = start
    x2, y2 = end
    if dashed:
        length = max(abs(x2 - x1), abs(y2 - y1))
        for offset in range(0, length, 18):
            t1, t2 = offset / length, min(offset + 9, length) / length
            draw.line((x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1,
                       x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2),
                      fill=color, width=width)
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 > x1 else -1
        head = ((x2, y2), (x2 - sign * 15, y2 - 9), (x2 - sign * 15, y2 + 9))
    else:
        sign = 1 if y2 > y1 else -1
        head = ((x2, y2), (x2 - 9, y2 - sign * 15), (x2 + 9, y2 - sign * 15))
    draw.polygon(head, fill=color)


def crop(path: Path, size: tuple[int, int]):
    image = Image.open(path).convert("RGB")
    ratio = size[0] / size[1]
    if image.width / image.height > ratio:
        width = int(image.height * ratio)
        left = (image.width - width) // 2
        image = image.crop((left, 0, left + width, image.height))
    else:
        height = int(image.width / ratio)
        top = (image.height - height) // 2
        image = image.crop((0, top, image.width, top + height))
    return image.resize(size, Image.Resampling.LANCZOS)


def panel_label(draw, x, y, letter, title):
    draw.text((x, y), f"({letter})", font=get_font(23, True), fill=INK)
    draw.text((x + 53, y), title, font=get_font(23, True), fill=INK)


def main():
    summaries = [json.loads(path.read_text()) for path in
                 sorted(RESULTS.glob("global_w4_seed*/evaluation_summary.json"))]
    if len(summaries) != 3:
        raise ValueError(f"expected three seed summaries, found {len(summaries)}")
    labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")
    accuracies = [float(item["validation_accuracy"]) for item in summaries]
    recalls = [statistics.mean(float(item["per_class"][label]["accuracy"])
                               for item in summaries) for label in labels]
    accuracy = statistics.mean(accuracies)
    accuracy_std = statistics.stdev(accuracies)

    image = Image.new("RGB", (2600, 1180), "white")
    draw = ImageDraw.Draw(image)

    # (a) Exact training architecture.
    panel_label(draw, 55, 32, "a", "Supervised primitive-skill training")
    draw.line((55, 75, 2545, 75), fill=LIGHT, width=2)
    y_mid = 285

    # Primitive-labelled clips.
    draw.rounded_rectangle((70, 120, 395, 450), radius=12, fill=(250, 250, 251),
                           outline=INK, width=2)
    draw.text((92, 138), "Primitive-labelled clips", font=get_font(19, True), fill=INK)
    source_paths = (FRAMES / "frame_01_cup.png", FRAMES / "frame_02_red_snack.png",
                    FRAMES / "frame_03_wipe.png")
    for index, path in enumerate(source_paths):
        thumb = crop(path, (87, 135))
        x = 92 + index * 95
        image.paste(thumb, (x, 184))
        draw.rectangle((x, 184, x + 87, 319), outline=INK, width=1)
    for index, label in enumerate(labels):
        col, row = index % 3, index // 3
        x, y = 91 + col * 96, 346 + row * 38
        draw.rounded_rectangle((x, y, x + 85, y + 27), radius=7,
                               fill=(255, 255, 255), outline=CLASS_COLORS[index], width=2)
        center_text(draw, (x, y, x + 85, y + 27), label, get_font(12, True), CLASS_COLORS[index])

    arrow(draw, (408, y_mid), (465, y_mid))
    module(draw, (480, 185, 725, 385), "Frozen V-JEPA2.1", "ViT-L/16 · 384 px",
           BLUE_BG, BLUE)
    # Dense token motif.
    for group in range(4):
        gx = 760 + group * 67
        for row in range(3):
            for col in range(3):
                x, y = gx + col * 13, 235 + row * 13
                draw.rectangle((x, y, x + 9, y + 9), fill=BLUE_BG, outline=BLUE, width=1)
        draw.text((gx + 7, 285), f"t{group + 1}", font=get_font(11), fill=GRAY)
    center_text(draw, (744, 315, 1032, 370), "dense patch tokens\n[B, 4, S, 1024]",
                get_font(14), GRAY)
    arrow(draw, (735, y_mid), (758, y_mid))

    arrow(draw, (1037, y_mid), (1090, y_mid))
    module(draw, (1105, 185, 1375, 385), "Learned spatial\nattention", "softmax over S patches",
           BLUE_BG, BLUE)
    # Weighted sum symbol.
    draw.text((1215, 266), "Σ", font=get_font(38, True), fill=BLUE)

    arrow(draw, (1388, y_mid), (1443, y_mid))
    module(draw, (1458, 185, 1655, 385), "Temporal mean", "mean over 4 tokens",
           GREEN_BG, GREEN)
    draw.text((1532, 264), "1/4 Σ", font=get_font(28, True), fill=GREEN)

    arrow(draw, (1668, y_mid), (1723, y_mid))
    module(draw, (1738, 155, 2180, 415), "MLP classifier", "ReLU + dropout 0.4",
           ORANGE_BG, ORANGE)
    layers = ((1770, "1024"), (1885, "256"), (1997, "128"), (2100, "6"))
    for index, (x, label) in enumerate(layers):
        height = (140, 105, 78, 52)[index]
        top = 277 - height // 2
        draw.rounded_rectangle((x, top, x + 54, top + height), radius=8,
                               fill="white", outline=ORANGE, width=2)
        center_text(draw, (x, top, x + 54, top + height), label, get_font(13, True), ORANGE)
        if index < len(layers) - 1:
            arrow(draw, (x + 57, 277), (layers[index + 1][0] - 4, 277), ORANGE, 2)

    arrow(draw, (2193, y_mid), (2248, y_mid))
    module(draw, (2263, 155, 2535, 415), "6 skill logits", "softmax at inference",
           (246, 243, 251), PURPLE)
    for index, label in enumerate(labels):
        y = 218 + index * 28
        draw.ellipse((2292, y + 3, 2305, y + 16), fill=CLASS_COLORS[index])
        draw.text((2317, y), label, font=get_font(13), fill=INK)

    # Supervision is attached to the classifier output, not to V-JEPA.
    draw.rounded_rectangle((108, 475, 357, 528), radius=9, fill=(255, 249, 249),
                           outline=RED, width=2)
    center_text(draw, (108, 475, 357, 528), "Ground-truth primitive label",
                get_font(15, True), RED)
    draw.line((357, 501, 2410, 501), fill=RED, width=2)
    arrow(draw, (2410, 501), (2410, 424), RED, 2)
    draw.text((2040, 508), "cross-entropy · label smoothing 0.1",
              font=get_font(13), fill=RED)
    draw.text((480, 420), "encoder frozen", font=get_font(13, True), fill=BLUE)
    draw.text((1105, 420), "trainable", font=get_font(13, True), fill=GREEN)
    draw.text((1738, 420), "trainable", font=get_font(13, True), fill=GREEN)

    # Separator.
    draw.line((55, 565, 2545, 565), fill=(205, 210, 218), width=2)

    # (b) Long-horizon inference.
    panel_label(draw, 55, 595, "b", "Long-horizon inference")
    draw.rounded_rectangle((75, 655, 555, 995), radius=12, fill=(250, 250, 251),
                           outline=INK, width=2)
    draw.text((98, 675), "Untrimmed video", font=get_font(19, True), fill=INK)
    for index, path in enumerate(source_paths):
        thumb = crop(path, (133, 205))
        x = 98 + index * 142
        image.paste(thumb, (x, 730))
        draw.rectangle((x, 730, x + 133, 935), outline=INK, width=1)
    draw.text((510, 815), "…", font=get_font(30, True), fill=INK)

    arrow(draw, (570, 825), (650, 825))
    module(draw, (668, 715, 1130, 935),
           "Frozen V-JEPA2.1  →  spatial attention\n→  4-token mean  →  MLP",
           "trained Global W4 model · RGB only", BLUE_BG, BLUE)
    # Shared-weight connection.
    arrow(draw, (1955, 445), (900, 700), GREEN, 3, dashed=True)
    draw.text((1220, 560), "shared trained weights", font=get_font(13, True), fill=GREEN)

    arrow(draw, (1145, 825), (1225, 825))
    draw.rounded_rectangle((1242, 690, 1795, 960), radius=12, fill=(250, 250, 251),
                           outline=INK, width=2)
    draw.text((1265, 712), "Frame-wise predictions", font=get_font(19, True), fill=INK)
    track_x, track_y, track_w = 1270, 795, 495
    segment_widths = (65, 78, 55, 82, 135, 80)
    cursor = track_x
    for index, width in enumerate(segment_widths):
        draw.rectangle((cursor, track_y, cursor + width, track_y + 55), fill=CLASS_COLORS[index])
        cursor += width
    draw.line((track_x, track_y + 77, track_x + track_w, track_y + 77), fill=INK, width=2)
    draw.polygon(((track_x + track_w, track_y + 77),
                  (track_x + track_w - 13, track_y + 70),
                  (track_x + track_w - 13, track_y + 84)), fill=INK)
    draw.text((1495, track_y + 91), "time", font=get_font(13), fill=GRAY)
    draw.text((1268, 912), "automatic primitive-skill segments", font=get_font(14), fill=GRAY)

    # (c) Exact reported validation results.
    panel_label(draw, 1850, 595, "c", "Validation results")
    draw.rounded_rectangle((1855, 655, 2535, 995), radius=12, fill="white",
                           outline=INK, width=2)
    draw.text((1880, 680), "Per-class recall (3-seed mean)", font=get_font(17, True), fill=INK)
    for index, (label, value) in enumerate(zip(labels, recalls)):
        y = 730 + index * 37
        draw.text((1880, y), label, font=get_font(14), fill=INK)
        draw.rounded_rectangle((1970, y + 2, 2280, y + 21), radius=8, fill=LIGHT)
        draw.rounded_rectangle((1970, y + 2, 1970 + int(310 * value), y + 21),
                               radius=8, fill=CLASS_COLORS[index])
        draw.text((2293, y), f"{value:.1%}", font=get_font(14), fill=GRAY)
    draw.line((2372, 704, 2372, 938), fill=LIGHT, width=2)
    draw.text((2400, 745), "Accuracy", font=get_font(15, True), fill=GREEN)
    center_text(draw, (2385, 775, 2518, 855), f"{accuracy:.2%}", get_font(31, True), GREEN)
    center_text(draw, (2385, 850, 2518, 890), f"± {accuracy_std:.2%}", get_font(15), GREEN)
    center_text(draw, (2385, 900, 2518, 950), "fixed split\n3 seeds", get_font(12), GRAY)

    draw.text((58, 1080),
              "Global W4 uses no object, language, or hand-pose branch. Reported accuracy is from the fixed validation split, not the long-horizon illustration.",
              font=get_font(15), fill=GRAY)

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "global_w4_paper_architecture.png"
    image.save(output, dpi=(300, 300))
    image.save(OUT / "global_w4_paper_architecture.pdf", "PDF", resolution=300.0)
    print(output)


if __name__ == "__main__":
    main()
