"""Auto-annotate validation videos and compare the result with existing GT.

This consumes persisted aligned V-JEPA feature bundles, never the GT as a
model input.  GT is loaded only after prediction for evaluation/visualization.
Both raw token predictions and an explicitly offline 3-token probability
smoother are exported in the annotation-tool JSON schema.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, f1_score

from skill_classifier.action_semantics import load_action_semantics
from skill_classifier.compare_model_diagnostics import _build_model, _dataset
from skill_classifier.infer_long_horizon import (
    load_object_context_sidecar,
    run_classifier_aligned,
)


PALETTE = (
    (56, 118, 255),
    (245, 158, 11),
    (45, 180, 120),
    (225, 77, 89),
    (145, 91, 220),
    (105, 115, 130),
)

SHORT_LABELS = {
    "HangCup": "Cup",
    "StackContainers": "Stack",
    "PlaceLightGreenSnackBoxInTrashBin": "Green",
    "PlaceRedSnackBoxInTrashBin": "Red",
    "WipeFloorWithSponge": "Wipe",
    "Transition": "Trans",
}


@dataclass
class EpisodeResult:
    name: str
    source_video: Path
    frames: int
    fps: float
    gt: np.ndarray
    raw: np.ndarray
    smooth: np.ndarray
    confidence: np.ndarray
    raw_metrics: dict
    smooth_metrics: dict


def _font(size, bold=False):
    candidates = [
        Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if bold else
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    return ImageFont.truetype(str(next(path for path in candidates if path.is_file())), size)


def _expand(root, recording_glob):
    return sorted({
        path
        for pattern in recording_glob.split(",") if pattern.strip()
        for path in root.glob(f"{pattern.strip()}/features.pt")
    })


def _segments(values, labels):
    if not len(values):
        return []
    output, start = [], 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            output.append({
                "start_frame": start,
                "end_frame": index - 1,
                "label": labels[int(values[start])],
            })
            start = index
    return output


def _load_gt(path, frames, labels):
    payload = json.loads(path.read_text())
    result = np.full(frames, -1, dtype=np.int64)
    for segment in payload["segments"]:
        if segment["label"] not in labels:
            continue
        start = max(0, int(segment["start_frame"]))
        end = min(frames - 1, int(segment["end_frame"]))
        result[start:end + 1] = labels.index(segment["label"])
    return result, payload


def _moving_average(probabilities, width=3):
    if width <= 1:
        return probabilities.copy()
    radius = width // 2
    padded = np.pad(probabilities, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([padded[index:index + width].mean(axis=0)
                     for index in range(len(probabilities))])


def _compress(values):
    return [segment["label"] for segment in _segments(values, [str(i) for i in range(64)])]


def _edit_score(prediction, target):
    pred, gt = _compress(prediction), _compress(target)
    table = np.zeros((len(pred) + 1, len(gt) + 1), dtype=np.int32)
    table[:, 0] = np.arange(len(pred) + 1)
    table[0, :] = np.arange(len(gt) + 1)
    for i in range(1, len(pred) + 1):
        for j in range(1, len(gt) + 1):
            table[i, j] = min(
                table[i - 1, j] + 1,
                table[i, j - 1] + 1,
                table[i - 1, j - 1] + (pred[i - 1] != gt[j - 1]),
            )
    return 100.0 * (1.0 - table[-1, -1] / max(len(pred), len(gt), 1))


def _segment_tuples(values):
    output = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            output.append((int(values[start]), start, index))  # end exclusive
            start = index
    return output


def _segment_f1(prediction, target, threshold):
    predicted, truth = _segment_tuples(prediction), _segment_tuples(target)
    used = np.zeros(len(truth), dtype=bool)
    tp = fp = 0
    for label, start, end in predicted:
        candidates = []
        for index, (gt_label, gt_start, gt_end) in enumerate(truth):
            intersection = max(0, min(end, gt_end) - max(start, gt_start))
            union = max(end, gt_end) - min(start, gt_start)
            candidates.append(intersection / union if label == gt_label and union else 0.0)
        best = int(np.argmax(candidates)) if candidates else -1
        if best >= 0 and candidates[best] >= threshold and not used[best]:
            tp += 1
            used[best] = True
        else:
            fp += 1
    fn = int((~used).sum())
    return 100.0 * (2 * tp) / max(2 * tp + fp + fn, 1)


def _metrics(prediction, target, labels):
    valid = target >= 0
    pred, gt = prediction[valid], target[valid]
    return {
        "frame_accuracy": float((pred == gt).mean()),
        "macro_f1": float(f1_score(gt, pred, labels=np.arange(len(labels)),
                                   average="macro", zero_division=0)),
        "edit_score": _edit_score(pred, gt),
        "segment_f1_10": _segment_f1(pred, gt, 0.10),
        "segment_f1_25": _segment_f1(pred, gt, 0.25),
        "segment_f1_50": _segment_f1(pred, gt, 0.50),
        "predicted_segments": len(_segment_tuples(pred)),
        "gt_segments": len(_segment_tuples(gt)),
        "evaluated_frames": int(valid.sum()),
        "per_class_recall": {
            name: float((pred[gt == index] == index).mean()) if bool((gt == index).any()) else None
            for index, name in enumerate(labels)
        },
    }


def _tolerant_frame_correct(prediction, target, radius):
    """Accept a prediction if that label occurs in the GT ±radius window.

    Away from a labelled transition this is identical to exact frame accuracy.
    Near a transition it accepts either adjacent GT label without modifying the
    prediction or selecting a preferred boundary after seeing the prediction.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if radius < 0:
        raise ValueError("tolerance radius must be non-negative")
    correct = prediction == target
    valid = target >= 0
    if radius == 0:
        return correct & valid
    for offset in range(1, radius + 1):
        correct[offset:] |= valid[offset:] & valid[:-offset] & (
            prediction[offset:] == target[:-offset]
        )
        correct[:-offset] |= valid[:-offset] & valid[offset:] & (
            prediction[:-offset] == target[offset:]
        )
    return correct & valid


def _boundary_ambiguous_mask(target, radius):
    """Return valid frames whose GT ±radius window contains another label."""
    target = np.asarray(target)
    valid = target >= 0
    ambiguous = np.zeros(target.shape, dtype=bool)
    for offset in range(1, radius + 1):
        changed = valid[offset:] & valid[:-offset] & (target[offset:] != target[:-offset])
        ambiguous[offset:] |= changed
        ambiguous[:-offset] |= changed
    return ambiguous & valid


def _tolerance_metrics(prediction, target, radius):
    valid = target >= 0
    tolerant = _tolerant_frame_correct(prediction, target, radius)
    ambiguous = _boundary_ambiguous_mask(target, radius)
    core = valid & ~ambiguous
    return {
        "radius_frames": int(radius),
        "tolerant_accuracy": float(tolerant[valid].mean()),
        "tolerant_correct_frames": int(tolerant.sum()),
        "evaluated_frames": int(valid.sum()),
        "ambiguous_boundary_frames": int(ambiguous.sum()),
        "core_frames": int(core.sum()),
        "core_coverage": float(core.sum() / max(valid.sum(), 1)),
        "core_accuracy": float((prediction[core] == target[core]).mean()) if core.any() else None,
        "core_correct_frames": int((prediction[core] == target[core]).sum()),
    }


def _save_annotation(path, name, frames, fps, values, labels, checkpoint, smoothing):
    path.write_text(json.dumps({
        "episode": name,
        "num_frames": frames,
        "fps": fps,
        "segments": _segments(values, labels),
        "generated_by": {
            "type": "vjepa2_skill_classifier",
            "checkpoint": str(checkpoint.resolve()),
            "smoothing": smoothing,
            "gt_used_for_prediction": False,
        },
    }, indent=2) + "\n")


def _video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return frames, fps, width, height


def _write_video(path, source, result, probabilities, labels, descriptions):
    capture = cv2.VideoCapture(str(source))
    source_frames, fps, source_w, source_h = _video_metadata(source)
    target_w, target_h = 960, 540
    size = (1440, 810)
    video_x, video_y = 24, 88
    panel_x, panel_y, panel_w, panel_h = 1008, 88, 408, 540
    timeline_x, timeline_y = 24, 652
    timeline_w, timeline_h = 1392, 134
    temporary = path.with_name(path.stem + "_tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    title_font = _font(22, True)
    main_font = _font(30, True)
    label_font = _font(17, True)
    small_font = _font(15)
    tiny_font = _font(13)

    def short_name(index):
        if index < 0:
            return "UNLABELED"
        return SHORT_LABELS.get(labels[index], labels[index])

    def draw_track(draw, values, y, x0, width, height):
        for segment in _segments(values, labels):
            start = x0 + round(width * segment["start_frame"] / max(len(values), 1))
            end = x0 + round(width * (segment["end_frame"] + 1) / max(len(values), 1))
            index = labels.index(segment["label"])
            draw.rectangle((start, y, max(start + 1, end), y + height),
                           fill=PALETTE[index % len(PALETTE)])

    frame_index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if frame_index >= len(result.gt):
            break
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Bright presentation theme: keep the camera feed visually dominant while
        # making the surrounding dashboard readable on white slides and screens.
        canvas = Image.new("RGB", size, (246, 248, 252))
        canvas.paste(Image.fromarray(rgb), (video_x, video_y))
        draw = ImageDraw.Draw(canvas)
        gt_index, raw_index, smooth_index = (
            int(result.gt[frame_index]), int(result.raw[frame_index]), int(result.smooth[frame_index])
        )
        correct = raw_index == gt_index
        status = "MATCH" if correct else "MISMATCH"
        status_color = (210, 246, 229) if correct else (255, 226, 230)
        status_text_color = (20, 122, 78) if correct else (190, 51, 68)

        # Header: compact metadata only, leaving the source video unobstructed.
        draw.text((24, 20), "AUTOMATIC ACTION ANNOTATION", font=title_font,
                  fill=(27, 42, 65))
        elapsed = frame_index / max(fps, 1e-6)
        duration = len(result.gt) / max(fps, 1e-6)
        meta = (f"{result.name}    {elapsed:05.1f}s / {duration:05.1f}s    "
                f"FRAME {frame_index + 1:,} / {len(result.gt):,}")
        meta_width = draw.textbbox((0, 0), meta, font=small_font)[2]
        draw.text((size[0] - meta_width - 24, 24), meta, font=small_font,
                  fill=(91, 108, 132))

        # Framed source video.
        draw.rounded_rectangle((video_x - 2, video_y - 2,
                               video_x + target_w + 2, video_y + target_h + 2),
                               radius=10, outline=(203, 213, 226), width=2)

        # Right-side prediction dashboard.
        draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
                               radius=16, fill=(255, 255, 255), outline=(216, 224, 235), width=2)
        draw.text((panel_x + 22, panel_y + 20), "LIVE PREDICTION", font=small_font,
                  fill=(93, 111, 138))
        draw.text((panel_x + 22, panel_y + 48), short_name(raw_index),
                  font=main_font, fill=PALETTE[raw_index % len(PALETTE)])
        confidence = float(probabilities[frame_index, raw_index])
        draw.text((panel_x + 22, panel_y + 91), f"Confidence  {confidence:.1%}",
                  font=label_font, fill=(42, 57, 79))
        chip_box = draw.textbbox((0, 0), status, font=tiny_font)
        chip_w = chip_box[2] - chip_box[0] + 24
        draw.rounded_rectangle((panel_x + panel_w - chip_w - 20, panel_y + 20,
                                panel_x + panel_w - 20, panel_y + 48),
                               radius=14, fill=status_color)
        draw.text((panel_x + panel_w - chip_w - 8, panel_y + 25), status,
                  font=tiny_font, fill=status_text_color)

        draw.rounded_rectangle((panel_x + 20, panel_y + 130,
                                panel_x + panel_w - 20, panel_y + 184),
                               radius=10, fill=(242, 246, 251))
        draw.text((panel_x + 34, panel_y + 143), "GT", font=small_font,
                  fill=(96, 113, 138))
        draw.text((panel_x + 92, panel_y + 140), short_name(gt_index),
                  font=label_font, fill=(154, 103, 0))
        draw.line((panel_x + 20, panel_y + 205, panel_x + panel_w - 20, panel_y + 205),
                  fill=(226, 232, 241), width=2)
        draw.text((panel_x + 22, panel_y + 222), "CLASS PROBABILITIES", font=small_font,
                  fill=(93, 111, 138))

        probability = probabilities[frame_index]
        for index, label in enumerate(labels):
            y = panel_y + 259 + index * 42
            name_color = (154, 103, 0) if index == gt_index else (50, 65, 88)
            draw.text((panel_x + 22, y), short_name(index), font=small_font,
                      fill=name_color)
            bar_x, bar_y, bar_w = panel_x + 103, y + 2, 220
            draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + 18),
                                   radius=9, fill=(229, 235, 243))
            color = PALETTE[index % len(PALETTE)]
            fill_w = max(2, int(bar_w * float(probability[index])))
            draw.rounded_rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + 18),
                                   radius=9, fill=color)
            draw.text((panel_x + 337, y), f"{probability[index]:.2f}",
                      font=small_font, fill=(91, 108, 132))

        # Full-sequence GT/AUTO tracks reveal how frame labels become segments.
        draw.rounded_rectangle((timeline_x, timeline_y,
                                timeline_x + timeline_w, timeline_y + timeline_h),
                               radius=16, fill=(255, 255, 255), outline=(216, 224, 235), width=2)
        draw.text((timeline_x + 20, timeline_y + 13), "SEQUENCE TIMELINE",
                  font=small_font, fill=(93, 111, 138))
        track_x = timeline_x + 104
        track_w = timeline_w - 128
        gt_y, auto_y, track_h = timeline_y + 51, timeline_y + 88, 22
        draw.text((timeline_x + 24, gt_y + 2), "GT", font=small_font,
                  fill=(154, 103, 0))
        draw.text((timeline_x + 24, auto_y + 2), "AUTO", font=small_font,
                  fill=(42, 57, 79))
        draw.rounded_rectangle((track_x, gt_y, track_x + track_w, gt_y + track_h),
                               radius=6, fill=(229, 235, 243))
        draw.rounded_rectangle((track_x, auto_y, track_x + track_w, auto_y + track_h),
                               radius=6, fill=(229, 235, 243))
        draw_track(draw, result.gt, gt_y, track_x, track_w, track_h)
        draw_track(draw, result.raw, auto_y, track_x, track_w, track_h)
        cursor_x = track_x + round(track_w * frame_index / max(len(result.gt) - 1, 1))
        draw.line((cursor_x, gt_y - 6, cursor_x, auto_y + track_h + 6),
                  fill=(25, 39, 61), width=3)
        draw.ellipse((cursor_x - 5, gt_y - 12, cursor_x + 5, gt_y - 2),
                     fill=(25, 39, 61))
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        frame_index += 1
    capture.release()
    writer.release()
    if frame_index != source_frames or frame_index != len(result.gt):
        raise ValueError(f"frame mismatch for {result.name}: wrote={frame_index}, video={source_frames}, expected={len(result.gt)}")
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary),
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(path),
    ], check=True)
    temporary.unlink()


def _concatenate_videos(paths, output, fps=30.0):
    """Concatenate rendered videos without inheriting mixed source time bases."""
    if not paths:
        return 0
    first = cv2.VideoCapture(str(paths[0]))
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()
    temporary = output.with_name(output.stem + "_tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    written = 0
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open rendered video: {path}")
        if (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != width
            or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height
        ):
            raise ValueError(f"rendered video dimensions differ: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        capture.release()
    writer.release()
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary),
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(output),
    ], check=True)
    temporary.unlink()
    return written


def _plot_confusion(path, gt, pred, labels):
    matrix = confusion_matrix(gt, pred, labels=np.arange(len(labels)), normalize="true")
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                      color="white" if matrix[row, column] > 0.5 else "black")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Automatic annotation")
    axis.set_ylabel("Ground truth")
    axis.set_title("Validation long sequences — normalized confusion matrix")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_tolerance_sweep(path, rows):
    radii = [row["radius_frames"] for row in rows]
    tolerant = [100.0 * row["tolerant_accuracy"] for row in rows]
    core = [100.0 * row["core_accuracy"] for row in rows]
    coverage = [100.0 * row["core_coverage"] for row in rows]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(radii, tolerant, marker="o", linewidth=2, label="Boundary-tolerant accuracy")
    axis.plot(radii, core, marker="s", linewidth=2, label="Core-only accuracy")
    axis.plot(radii, coverage, marker="^", linestyle="--", label="Core-frame coverage")
    axis.set_xlabel("GT boundary tolerance radius (source frames)")
    axis.set_ylabel("Percent")
    axis.set_ylim(0, 101)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("Automatic annotation sensitivity to uncertain GT boundaries")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--recording-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--tolerance-frames", default="0,2,4,8,15,30",
        help="Comma-separated GT boundary tolerance radii in source frames",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tolerance_radii = sorted({int(value) for value in args.tolerance_frames.split(",")})
    if not tolerance_radii or tolerance_radii[0] < 0:
        raise ValueError("--tolerance-frames must contain non-negative integers")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    labels = list(saved_args["active_labels"])
    semantics = load_action_semantics(Path("src/skill_classifier/config/kitchen_action_semantics.yaml"))
    descriptions = {label: semantics["actions"][label]["ko"] for label in labels}
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = _dataset(saved_args)
    model = _build_model(checkpoint, dataset).to(device).eval()

    results, combined_gt, combined_raw, combined_smooth = [], [], [], []
    video_paths = []
    for features_path in _expand(args.feature_root, args.recording_glob):
        name = features_path.parent.name
        bundle = torch.load(features_path, map_location="cpu", weights_only=False)
        feature_key = {
            "vjepa_orig_dense": "vjepa_orig_dense",
            "vjepa_orig": "vjepa_orig",
            "masked_vjepa_orig": "vjepa_orig_masked",
            "vjepa_robot": "vjepa_robot",
        }.get(saved_args["variant"])
        if feature_key is None:
            raise ValueError(f"unsupported V-JEPA validation variant: {saved_args['variant']}")
        vjepa = bundle[feature_key]
        if saved_args["model"] in (
            "object_mask_attention_mlp",
            "object_text_prototype_mlp",
        ):
            context = load_object_context_sidecar(
                features_path, bundle, saved_args
            )
        else:
            context = torch.zeros(len(vjepa), 0)
        token_prediction, token_probability = run_classifier_aligned(
            model, vjepa, context, int(saved_args["window_size"]), device
        )
        mapping = np.asarray(bundle["frame_to_token"], dtype=np.int64)
        source = (features_path.parent / "rgb").resolve()
        frames, fps, _, _ = _video_metadata(source)
        if len(mapping) != frames:
            raise ValueError(f"bundle/video frame mismatch for {name}: {len(mapping)} != {frames}")
        gt, _ = _load_gt((features_path.parent / "gt_labels.json").resolve(), frames, labels)
        raw = token_prediction[mapping]
        smoothed_token_probability = _moving_average(token_probability, width=3)
        smooth = smoothed_token_probability.argmax(axis=1)[mapping]
        raw_probability_per_frame = token_probability[mapping]
        probability_per_frame = smoothed_token_probability[mapping]
        valid = gt >= 0
        result = EpisodeResult(
            name=name, source_video=source, frames=frames, fps=fps, gt=gt,
            raw=raw, smooth=smooth, confidence=probability_per_frame.max(axis=1),
            raw_metrics=_metrics(raw, gt, labels),
            smooth_metrics=_metrics(smooth, gt, labels),
        )
        results.append(result)
        episode_dir = args.output_dir / name
        episode_dir.mkdir(parents=True, exist_ok=True)
        _save_annotation(episode_dir / "auto_annotations_raw.json", name, frames, fps,
                         raw, labels, args.checkpoint, "none")
        _save_annotation(episode_dir / "auto_annotations_smooth3.json", name, frames, fps,
                         smooth, labels, args.checkpoint, "offline centered 3-token probability mean")
        (episode_dir / "metrics.json").write_text(json.dumps({
            "episode": name,
            "source_video": str(source),
            "raw": result.raw_metrics,
            "smooth3": result.smooth_metrics,
        }, indent=2) + "\n")
        np.savez_compressed(episode_dir / "frame_predictions.npz", gt=gt, raw=raw,
                            smooth3=smooth, probabilities=probability_per_frame)
        if not args.skip_video:
            video_path = episode_dir / "gt_vs_auto_annotation.mp4"
            _write_video(video_path, source, result, raw_probability_per_frame, labels, descriptions)
            video_paths.append(video_path)
        combined_gt.append(gt[valid])
        combined_raw.append(raw[valid])
        combined_smooth.append(smooth[valid])

    gt = np.concatenate(combined_gt)
    raw = np.concatenate(combined_raw)
    smooth = np.concatenate(combined_smooth)
    aggregate = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": len(results),
        "total_frames": int(sum(result.frames for result in results)),
        "raw": _metrics(raw, gt, labels),
        "smooth3": _metrics(smooth, gt, labels),
        "note": "GT was used only for evaluation; smooth3 is an offline centered filter.",
        "per_episode": [{
            "episode": result.name,
            "frames": result.frames,
            "duration_seconds": result.frames / result.fps,
            "raw_accuracy": result.raw_metrics["frame_accuracy"],
            "smooth3_accuracy": result.smooth_metrics["frame_accuracy"],
            "raw_segments": result.raw_metrics["predicted_segments"],
            "smooth3_segments": result.smooth_metrics["predicted_segments"],
            "gt_segments": result.smooth_metrics["gt_segments"],
        } for result in results],
    }
    tolerance_rows = []
    for radius in tolerance_radii:
        raw_parts = [_tolerance_metrics(result.raw, result.gt, radius) for result in results]
        smooth_parts = [_tolerance_metrics(result.smooth, result.gt, radius) for result in results]
        evaluated = sum(part["evaluated_frames"] for part in raw_parts)
        ambiguous = sum(part["ambiguous_boundary_frames"] for part in raw_parts)
        core_frames = sum(part["core_frames"] for part in raw_parts)
        raw_core_correct = sum(part["core_correct_frames"] for part in raw_parts)
        smooth_core_correct = sum(part["core_correct_frames"] for part in smooth_parts)
        tolerance_rows.append({
            "radius_frames": radius,
            "radius_seconds_at_mean_fps": radius / float(np.mean([r.fps for r in results])),
            "evaluated_frames": evaluated,
            "ambiguous_boundary_frames": ambiguous,
            "core_frames": core_frames,
            "core_coverage": core_frames / max(evaluated, 1),
            "raw_tolerant_accuracy": sum(p["tolerant_correct_frames"] for p in raw_parts) / max(evaluated, 1),
            "raw_core_accuracy": raw_core_correct / max(core_frames, 1),
            "smooth3_tolerant_accuracy": sum(p["tolerant_correct_frames"] for p in smooth_parts) / max(evaluated, 1),
            "smooth3_core_accuracy": smooth_core_correct / max(core_frames, 1),
        })
    aggregate["gt_boundary_tolerance"] = tolerance_rows
    aggregate["gt_boundary_tolerance_definition"] = {
        "tolerant_accuracy": "Prediction is accepted when its label appears anywhere in the GT ±radius window; denominator remains all labelled frames.",
        "core_accuracy": "Exact accuracy after excluding frames whose GT ±radius window contains more than one label.",
        "warning": "Tolerance is an uncertainty sensitivity analysis, not a replacement for independently reviewed GT.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    with (args.output_dir / "per_episode.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate["per_episode"][0]))
        writer.writeheader()
        writer.writerows(aggregate["per_episode"])
    with (args.output_dir / "gt_boundary_tolerance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tolerance_rows[0]))
        writer.writeheader()
        writer.writerows(tolerance_rows)
    _plot_tolerance_sweep(
        args.output_dir / "gt_boundary_tolerance.png",
        [{
            "radius_frames": row["radius_frames"],
            "tolerant_accuracy": row["raw_tolerant_accuracy"],
            "core_accuracy": row["raw_core_accuracy"],
            "core_coverage": row["core_coverage"],
        } for row in tolerance_rows],
    )
    _plot_confusion(args.output_dir / "confusion_matrix_raw.png", gt, raw, labels)
    _plot_confusion(args.output_dir / "confusion_matrix_smooth3.png", gt, smooth, labels)

    if video_paths:
        written = _concatenate_videos(
            video_paths, args.output_dir / "all_validation_gt_vs_auto.mp4", fps=30.0
        )
        if written != aggregate["total_frames"]:
            raise ValueError(
                f"combined video frame mismatch: {written} != {aggregate['total_frames']}"
            )
    readme = [
        "# Validation long-sequence automatic annotation",
        "",
        "The V-JEPA2 spatial-attention classifier predicted labels without reading GT. Existing GT was loaded only afterwards for scoring and visualization.",
        "Each source video contains multiple skills; the temporal window is reset at the start of each real recording.",
        "",
        "## Aggregate raw prediction",
        "",
        f"- Frame accuracy: {aggregate['raw']['frame_accuracy']:.2%}",
        f"- Macro-F1: {aggregate['raw']['macro_f1']:.2%}",
        f"- Edit score: {aggregate['raw']['edit_score']:.2f}",
        f"- Segment F1@10/25/50: {aggregate['raw']['segment_f1_10']:.2f} / {aggregate['raw']['segment_f1_25']:.2f} / {aggregate['raw']['segment_f1_50']:.2f}",
        f"- Predicted/GT segments: {aggregate['raw']['predicted_segments']} / {aggregate['raw']['gt_segments']}",
        "",
        "## GT boundary uncertainty sweep",
        "",
        "| ±Frames | Seconds | Tolerant accuracy | Core accuracy | Core coverage |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {row['radius_frames']} | {row['radius_seconds_at_mean_fps']:.3f} | "
            f"{row['raw_tolerant_accuracy']:.2%} | {row['raw_core_accuracy']:.2%} | "
            f"{row['core_coverage']:.2%} |"
            for row in tolerance_rows
        ],
        "",
        "## Episodes",
        "",
        "| Episode | Frames | Accuracy | Pred/GT segments |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate["per_episode"]:
        readme.append(
            f"| {row['episode']} | {row['frames']} | {row['raw_accuracy']:.2%} | "
            f"{row['raw_segments']} / {row['gt_segments']} |"
        )
    readme.extend([
        "",
        "`auto_annotations_raw.json` is the recommended annotation output. The centered smooth3 experiment is preserved only as a negative comparison because it reduced boundary accuracy.",
    ])
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(json.dumps(aggregate, indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
