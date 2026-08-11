"""Render full validation videos with interpolated spatial-attention overlays."""

from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from data_preprocess.preprocess import VJEPA2_EVAL_CROP, preprocess_rgb_frame
from skill_classifier.models import build_model
from skill_classifier.skill_dataset import load_recordings


CLASS_COLORS_RGB = (
    (78, 121, 167),
    (89, 161, 79),
    (242, 142, 43),
    (225, 87, 89),
    (176, 122, 161),
    (118, 183, 178),
)
HEADER_HEIGHT = 76
FRAME_SIZE = 384


def token_windows(features: torch.Tensor, window_size: int) -> torch.Tensor:
    windows = []
    for token_index in range(len(features)):
        start = token_index - window_size + 1
        if start >= 0:
            window = features[start : token_index + 1]
        else:
            padding = features.new_zeros((-start, *features.shape[1:]))
            window = torch.cat((padding, features[: token_index + 1]), dim=0)
        windows.append(window)
    return torch.stack(windows)


def infer_tokens(model, bundle, window_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    windows = token_windows(bundle["vjepa_orig_dense"], window_size)
    probabilities, attention = [], []
    with torch.inference_mode():
        for start in range(0, len(windows), 8):
            dense = windows[start : start + 8].to(device)
            hand = torch.zeros((len(dense), window_size, 0), device=device)
            logits = model(dense, hand)
            _, weights = model.representation(dense)
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            attention.append(weights[:, -1].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(attention)


def interpolate_tokens(
    frame_index: int,
    centers: np.ndarray,
    probabilities: np.ndarray,
    attention: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right = int(np.searchsorted(centers, frame_index, side="left"))
    if right <= 0:
        return probabilities[0], attention[0]
    if right >= len(centers):
        return probabilities[-1], attention[-1]
    left = right - 1
    span = max(1, int(centers[right] - centers[left]))
    alpha = float(np.clip((frame_index - centers[left]) / span, 0.0, 1.0))
    return (
        probabilities[left] * (1.0 - alpha) + probabilities[right] * alpha,
        attention[left] * (1.0 - alpha) + attention[right] * alpha,
    )


def put_text(canvas, text, origin, scale, color, thickness=1, align_right=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if align_right:
        width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        origin = (origin[0] - width, origin[1])
    cv2.putText(canvas, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def render_frame(
    bgr: np.ndarray,
    frame_index: int,
    bundle: dict,
    names: list[str],
    probabilities: np.ndarray,
    attention: np.ndarray,
) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crop = preprocess_rgb_frame(rgb, FRAME_SIZE, VJEPA2_EVAL_CROP)
    centers = np.asarray(bundle["token_center_frame_indices"], dtype=int)
    frame_probability, frame_attention = interpolate_tokens(
        frame_index, centers, probabilities, attention
    )
    side = round(math.sqrt(len(frame_attention)))
    heatmap = frame_attention.reshape(side, side)
    heatmap = (heatmap - heatmap.min()) / max(float(np.ptp(heatmap)), 1e-12)
    heatmap = cv2.resize(heatmap.astype(np.float32), (FRAME_SIZE, FRAME_SIZE),
                         interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(np.uint8(np.clip(heatmap * 255, 0, 255)), cv2.COLORMAP_TURBO)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(crop_bgr, 0.48, heat_color, 0.52, 0.0)

    canvas = np.full((HEADER_HEIGHT + FRAME_SIZE, FRAME_SIZE, 3), (248, 250, 252), dtype=np.uint8)
    canvas[HEADER_HEIGHT:] = overlay
    nearest_token = int(bundle["frame_to_token"][min(frame_index, len(bundle["frame_to_token"]) - 1)])
    target = int(bundle["labels_per_token"][nearest_token])
    label_name = names[target] if target >= 0 else "Unlabeled"
    prediction = int(frame_probability.argmax())
    prediction_name = names[prediction]
    confidence = float(frame_probability[prediction])
    color_rgb = CLASS_COLORS_RGB[target if target >= 0 else prediction]
    color_bgr = tuple(reversed(color_rgb))
    cv2.rectangle(canvas, (0, 0), (7, HEADER_HEIGHT - 1), color_bgr, -1)
    put_text(canvas, f"Label: {label_name}", (16, 28), 0.62, color_bgr, thickness=2)
    put_text(canvas, f"Pred: {prediction_name}  p={confidence:.2f}", (16, 57), 0.52,
             (35, 50, 64), thickness=1)
    short_recording = str(bundle["recording"]).replace("0727__", "")
    put_text(canvas, short_recording, (376, 22), 0.36, (90, 105, 118),
             thickness=1, align_right=True)
    put_text(canvas, f"frame {frame_index + 1}/{bundle['num_frames']}", (376, 48), 0.34,
             (90, 105, 118), thickness=1, align_right=True)
    return canvas


def transcode(temp_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(temp_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ],
        check=True,
    )
    temp_path.unlink()


def render_recording(model, bundle, names, window_size, output_dir, device) -> dict:
    source = Path(bundle["input_provenance"]["rgb"]["path"])
    probabilities, attention = infer_tokens(model, bundle, window_size, device)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(bundle["source_fps"])
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = str(bundle["recording"]).replace("/", "_")
    temp_path = output_dir / f".{stem}.mp4v.mp4"
    output_path = output_dir / f"{stem}_attention.mp4"
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (FRAME_SIZE, FRAME_SIZE + HEADER_HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create {temp_path}")
    rendered = 0
    while rendered < frame_count:
        ok, bgr = capture.read()
        if not ok:
            break
        writer.write(render_frame(bgr, rendered, bundle, names, probabilities, attention))
        rendered += 1
    capture.release()
    writer.release()
    if rendered != frame_count:
        raise RuntimeError(f"decoded {rendered}/{frame_count} frames from {source}")
    transcode(temp_path, output_path)
    return {
        "recording": bundle["recording"],
        "source": str(source),
        "output": str(output_path.resolve()),
        "fps": fps,
        "frames": rendered,
        "duration_seconds": rendered / fps,
        "tokens": int(bundle["num_tokens"]),
    }


def build_grid(videos: list[dict], output_dir: Path) -> Path:
    captures = [cv2.VideoCapture(item["output"]) for item in videos]
    fps = float(captures[0].get(cv2.CAP_PROP_FPS))
    frame_counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
    temp_path = output_dir / ".all_validation_attention_grid.mp4v.mp4"
    output_path = output_dir / "all_validation_attention_grid.mp4"
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (FRAME_SIZE * 3, (FRAME_SIZE + HEADER_HEIGHT) * 3),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {temp_path}")
    blank = np.zeros((FRAME_SIZE + HEADER_HEIGHT, FRAME_SIZE, 3), dtype=np.uint8)
    for frame_index in range(max(frame_counts)):
        cells = []
        for capture, count in zip(captures, frame_counts):
            if frame_index < count:
                ok, frame = capture.read()
                cells.append(frame if ok else blank)
            else:
                cells.append(blank)
        rows = [np.hstack(cells[index : index + 3]) for index in range(0, 9, 3)]
        writer.write(np.vstack(rows))
    for capture in captures:
        capture.release()
    writer.release()
    transcode(temp_path, output_path)
    return output_path


def write_gallery(output_dir: Path, videos: list[dict], grid_path: Path) -> None:
    cards = "".join(
        f"<figure><video controls preload='metadata' src='{html.escape(Path(item['output']).name)}'></video>"
        f"<figcaption>{html.escape(item['recording'])} · {item['duration_seconds']:.1f}s</figcaption></figure>"
        for item in videos
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>Full attention videos</title>
<style>body{{font:16px system-ui;max-width:1450px;margin:30px auto;background:#eef2f6;color:#15324a}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}figure{{margin:0;background:white;padding:12px;border-radius:10px}}
video{{width:100%;height:auto;background:#000}}figcaption{{padding:8px;font-weight:700}}.hero{{max-width:900px}}</style>
<h1>Full-sequence spatial-attention videos</h1><p>Top-left header: ground-truth label and model prediction. Heatmap is interpolated between V-JEPA tokens.</p>
<figure class='hero'><video controls preload='metadata' src='{html.escape(grid_path.name)}'></video><figcaption>3×3 validation overview</figcaption></figure>
<div class='grid'>{cards}</div>"""
    (output_dir / "index.html").write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    args.experiment_dir = args.experiment_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((args.experiment_dir / "config.yaml").read_text())
    checkpoint = torch.load(
        args.experiment_dir / "best_spatial_attention_mlp.pt",
        map_location="cpu", weights_only=False,
    )
    model_args = checkpoint["args"]
    names = list(model_args["active_labels"])
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
    if len(bundles) != 9:
        raise ValueError(f"3x3 overview expects nine validation recordings, got {len(bundles)}")
    videos = []
    for index, bundle in enumerate(bundles, 1):
        item = render_recording(
            model, bundle, names, int(model_args["window_size"]), args.output_dir, device
        )
        videos.append(item)
        print(f"[{index}/{len(bundles)}] {item['recording']}: {item['frames']} frames")
    grid_path = build_grid(videos, args.output_dir)
    report = {
        "experiment_dir": str(args.experiment_dir),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "labels": names,
        "attention_interpolation": "linear between adjacent token centers",
        "label_mapping": "nearest token via frame_to_token",
        "videos": videos,
        "grid": str(grid_path.resolve()),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    write_gallery(args.output_dir, videos, grid_path)
    print(f"[ok] gallery: {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
