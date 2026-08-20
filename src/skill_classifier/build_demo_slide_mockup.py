"""Compose a 16:9 presentation slide with a demo-video placeholder."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

from skill_classifier.build_primitive_skill_pipeline_figure import (
    BLUE,
    GREEN,
    LABEL_COLORS,
    MUTED,
    NAVY,
    TEXT,
    centered,
    font,
    rounded,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE = (
    ROOT / "output/skill_classifier/kitchen_0813_model_comparison/figures"
    / "automatic_action_annotation_pipeline_meeting_final.png"
)
RESULT_ROOT = (
    ROOT / "output/skill_classifier/kitchen_0724_0728_human_vjepa21_vlm_sam_no_choco"
    / "multiseed_300e"
)
OUTPUT = ROOT / "output/skill_classifier/kitchen_0813_model_comparison/slides"


def main():
    summaries = [json.loads(path.read_text()) for path in
                 sorted(RESULT_ROOT.glob("global_w4_seed*/evaluation_summary.json"))]
    labels = ("Cup", "Lock", "Milk", "Snack", "Sweep", "Trans")
    accuracy_values = [float(item["validation_accuracy"]) for item in summaries]
    accuracy = statistics.mean(accuracy_values)
    accuracy_std = statistics.stdev(accuracy_values)
    recalls = tuple(
        statistics.mean(float(item["per_class"][label]["accuracy"]) for item in summaries)
        for label in labels
    )

    canvas = Image.new("RGB", (1920, 1080), (247, 249, 253))
    draw = ImageDraw.Draw(canvas)
    title_font = font(50, True)
    subtitle_font = font(21)
    kicker_font = font(14, True)
    card_title = font(23, True)
    body_font = font(16)
    metric_font = font(46, True)
    small_font = font(13)

    # Header.
    draw.rounded_rectangle((55, 42, 205, 78), radius=18, fill=(231, 238, 253))
    centered(draw, (55, 42, 205, 78), "CONTRIBUTION 1", kicker_font, BLUE)
    draw.text((55, 98), "Automatic Primitive Skill Annotation",
              font=title_font, fill=NAVY)
    draw.text((57, 158),
              "Learning primitive skills from human demonstrations and segmenting long-horizon video automatically",
              font=subtitle_font, fill=MUTED)

    # Left method-overview area.
    left = (45, 220, 1180, 1015)
    rounded(draw, left, fill=(255, 255, 255), outline=(208, 218, 232), radius=22, width=2)
    draw.text((72, 246), "METHOD OVERVIEW", font=kicker_font, fill=BLUE)
    draw.text((72, 277), "Primitive supervision → shared model → temporal annotation",
              font=card_title, fill=NAVY)

    source = Image.open(FIGURE).convert("RGB")
    # Remove the original bottom metric strip; metrics receive their own slide card.
    method = source.crop((0, 0, source.width, 770))
    target_w = 1075
    target_h = int(method.height * target_w / method.width)
    method = method.resize((target_w, target_h), Image.Resampling.LANCZOS)
    canvas.paste(method, (72, 340))

    # Right demo-video card; exact 16:9 content well.
    video_card = (1215, 220, 1875, 680)
    rounded(draw, video_card, fill=(255, 255, 255), outline=(208, 218, 232), radius=22, width=2)
    draw.text((1242, 246), "QUALITATIVE DEMO", font=kicker_font, fill=BLUE)
    draw.text((1242, 277), "Long-horizon video annotation", font=card_title, fill=NAVY)
    video_box = (1242, 322, 1848, 663)
    # Keep the visible video well at the same 16:9 ratio as the rendered demo.
    draw.rounded_rectangle(video_box, radius=16, fill=(20, 34, 59),
                           outline=(20, 34, 59), width=2)
    cx = (video_box[0] + video_box[2]) // 2
    cy = (video_box[1] + video_box[3]) // 2
    draw.ellipse((cx - 43, cy - 43, cx + 43, cy + 43),
                 fill=(255, 255, 255), outline=(255, 255, 255), width=2)
    draw.polygon(((cx - 12, cy - 22), (cx - 12, cy + 22), (cx + 27, cy)), fill=BLUE)
    centered(draw, (video_box[0], cy + 58, video_box[2], cy + 90),
             "INSERT DEMO VIDEO", kicker_font, (255, 255, 255))
    centered(draw, (video_box[0], cy + 88, video_box[2], cy + 116),
             "16:9 · replace this rectangle with MP4", small_font, (174, 190, 215))

    # Right quantitative-results card.
    results_card = (1215, 700, 1875, 1015)
    rounded(draw, results_card, fill=(255, 255, 255), outline=(208, 218, 232), radius=22, width=2)
    draw.text((1242, 726), "QUANTITATIVE RESULT", font=kicker_font, fill=GREEN)
    draw.text((1242, 757), "Fixed-split validation · 3-seed mean", font=body_font, fill=MUTED)

    draw.text((1242, 800), f"{accuracy:.2%}", font=metric_font, fill=GREEN)
    draw.text((1450, 820), f"± {accuracy_std:.2%}", font=font(20, True), fill=GREEN)
    draw.text((1244, 863), "Accuracy", font=small_font, fill=MUTED)
    draw.line((1535, 792, 1535, 968), fill=(224, 229, 237), width=2)

    # Compact 3×2 metric grid, requested explicitly in the meeting.
    for index, (label, value) in enumerate(zip(labels, recalls)):
        col, row = index % 3, index // 3
        x = 1570 + col * 92
        y = 802 + row * 83
        draw.ellipse((x, y + 5, x + 14, y + 19), fill=LABEL_COLORS[index])
        draw.text((x + 21, y), label, font=small_font, fill=TEXT)
        draw.text((x, y + 29), f"{value:.1%}", font=font(17, True), fill=LABEL_COLORS[index])

    # Bottom accent / presentation footer.
    draw.rounded_rectangle((55, 1042, 1865, 1048), radius=3, fill=BLUE)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "automatic_skill_annotation_demo_slide.png"
    canvas.save(output, dpi=(150, 150))
    print(output)


if __name__ == "__main__":
    main()
