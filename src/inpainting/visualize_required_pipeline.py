"""Create a compact pipeline video containing only required stages.

Layout:

    1 RAW | 2 HaWoR MANO | 3 HaCo CONTACT | 4 HAND+ARM MASK
          5 INPAINTED BG | 6 ROBOT RENDER | 7 FINAL

The obsolete empty AMODAL panel and redundant intermediate overlays are
intentionally omitted.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


PANEL_W = 360
PANEL_H = 480
LABEL_H = 42
CONTENT_H = PANEL_H - LABEL_H
CANVAS_W = PANEL_W * 4
CANVAS_H = PANEL_H * 2

LABELS = (
    ("1  RAW", (210, 210, 210)),
    ("2  HaWoR MANO", (255, 255, 0)),
    ("3  HaCo CONTACT", (0, 255, 0)),
    ("4  HAND + ARM MASK", (0, 80, 255)),
    ("5  INPAINTED BACKGROUND", (255, 180, 70)),
    ("6  ROBOT RENDER", (255, 120, 255)),
    ("7  FINAL COMPOSITE", (255, 255, 255)),
)


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    return cap


def read_frame(cap: cv2.VideoCapture, path: Path, frame_idx: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    return frame


def fit_content(frame: np.ndarray) -> np.ndarray:
    """Letterbox without changing the source aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(PANEL_W / width, CONTENT_H / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    content = np.zeros((CONTENT_H, PANEL_W, 3), dtype=np.uint8)
    x0 = (PANEL_W - new_w) // 2
    y0 = (CONTENT_H - new_h) // 2
    content[y0:y0 + new_h, x0:x0 + new_w] = resized
    return content


def make_panel(frame: np.ndarray, label: str, accent: tuple[int, int, int]) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[LABEL_H:] = fit_content(frame)
    cv2.rectangle(panel, (0, 0), (PANEL_W, LABEL_H), (16, 16, 16), -1)
    cv2.rectangle(panel, (0, 0), (7, LABEL_H), accent, -1)
    cv2.putText(panel, label, (18, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1),
                  (48, 48, 48), 1)
    return panel


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    active = mask.astype(bool)
    overlay = frame.copy()
    tint = np.zeros_like(frame)
    tint[:] = (0, 0, 255)
    overlay[active] = cv2.addWeighted(
        frame, 0.28, tint, 0.72, 0.0
    )[active]
    contours, _ = cv2.findContours(
        active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 4,
                     lineType=cv2.LINE_AA)
    return overlay


def extract_hawor_haco(comparison: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop the two 540-square result panels from the comparison video."""
    height, width = comparison.shape[:2]
    panel_size = width // 3
    footer_h = 42
    header_h = height - panel_size - footer_h
    if panel_size <= 0 or header_h < 0:
        raise ValueError(f"Unexpected HaWoR/HaCo comparison shape: {comparison.shape}")
    hawor = comparison[header_h:header_h + panel_size, panel_size:panel_size * 2]
    haco = comparison[header_h:header_h + panel_size, panel_size * 2:panel_size * 3]
    return hawor, haco


def compose(
    raw: np.ndarray,
    comparison: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    robot: np.ndarray,
    final: np.ndarray,
) -> np.ndarray:
    hawor, haco = extract_hawor_haco(comparison)
    mask_vis = overlay_mask(raw, mask)
    frames = (raw, hawor, haco, mask_vis, background, robot, final)
    panels = [
        make_panel(frame, label, accent)
        for frame, (label, accent) in zip(frames, LABELS)
    ]

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    for index, panel in enumerate(panels[:4]):
        x0 = index * PANEL_W
        canvas[0:PANEL_H, x0:x0 + PANEL_W] = panel

    # Center the shorter second row so there is no fake/empty eighth panel.
    bottom_x0 = PANEL_W // 2
    for index, panel in enumerate(panels[4:]):
        x0 = bottom_x0 + index * PANEL_W
        canvas[PANEL_H:CANVAS_H, x0:x0 + PANEL_W] = panel
    return canvas


def encode_episode(episode: Path, fps_override: float | None = None) -> tuple[Path, Path]:
    processed = episode / "inpainting_processed" / episode.name / "0"
    paths = {
        "raw": processed / "video_L.mp4",
        "comparison": episode / "visualization" / "hawor_haco_comparison.mp4",
        "background": processed / "inpaint_processor" / "video_human_inpaint.mkv",
        "robot": processed / "overlay_processor" / "video_robot_only.mp4",
        "final": processed / "video_overlay_rby1_xhand.mp4",
    }
    mask_path = processed / "segmentation_processor" / "masks_arm.npy"
    for path in (*paths.values(), mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    captures = {name: open_video(path) for name, path in paths.items()}
    raw_cap = captures["raw"]
    expected = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps_override or raw_cap.get(cv2.CAP_PROP_FPS) or 30.0
    masks = np.load(mask_path, mmap_mode="r")
    if len(masks) < expected:
        raise ValueError(
            f"{episode.name}: masks={len(masks)} shorter than raw={expected}"
        )

    for name, cap in captures.items():
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0 and count != expected:
            raise ValueError(
                f"{episode.name}: {name} frames={count}, expected={expected}"
            )

    output_path = processed / "pipeline_required_components.mp4"
    preview_path = processed / "pipeline_required_components_preview.jpg"
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", f"{fps:g}",
        "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    preview = None
    try:
        for frame_idx in range(expected):
            raw = read_frame(captures["raw"], paths["raw"], frame_idx)
            comparison = read_frame(
                captures["comparison"], paths["comparison"], frame_idx
            )
            background = read_frame(
                captures["background"], paths["background"], frame_idx
            )
            robot = read_frame(captures["robot"], paths["robot"], frame_idx)
            final = read_frame(captures["final"], paths["final"], frame_idx)
            canvas = compose(
                raw=raw,
                comparison=comparison,
                mask=np.asarray(masks[frame_idx]),
                background=background,
                robot=robot,
                final=final,
            )
            if frame_idx == expected // 2:
                preview = canvas.copy()
            assert encoder.stdin is not None
            encoder.stdin.write(canvas.tobytes())
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
        for cap in captures.values():
            cap.release()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {episode.name}: exit {return_code}")
    if preview is not None:
        cv2.imwrite(str(preview_path), preview)
    return output_path, preview_path


def make_overview(previews: list[Path], output_path: Path) -> None:
    images = [cv2.imread(str(path)) for path in previews]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumb_w = 720
    thumb_h = int(round(images[0].shape[0] * thumb_w / images[0].shape[1]))
    thumbs = [
        cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    if len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + 2]) for i in range(0, len(thumbs), 2)]
    cv2.imwrite(str(output_path), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--recording_glob", default="IMG_*")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    patterns = [part.strip() for part in args.recording_glob.split(",") if part.strip()]
    episodes = sorted({
        path
        for pattern in patterns
        for path in args.data_root.glob(pattern)
        if path.is_dir()
    })
    if not episodes:
        raise FileNotFoundError(f"No episodes found under {args.data_root}")

    previews = []
    for index, episode in enumerate(episodes, start=1):
        output, preview = encode_episode(episode, fps_override=args.fps)
        previews.append(preview)
        print(f"[{index}/{len(episodes)}] {episode.name}: {output}", flush=True)

    overview_path = args.data_root / "pipeline_required_overview.jpg"
    make_overview(previews, overview_path)
    print(f"Overview: {overview_path}")


if __name__ == "__main__":
    main()
