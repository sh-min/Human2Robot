"""Create an eight-panel SAM2 smoothing A/B comparison video.

Layout:

    RAW | MASK OFF | MASK ON | MASK DELTA
    INPAINT OFF | INPAINT ON | FINAL OFF | FINAL ON

In MASK DELTA, red pixels were removed by smoothing and green pixels were
added. Pixels shared by both masks are shown in muted blue.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


PANEL_W = 360
PANEL_H = 360
LABEL_H = 42
CONTENT_H = PANEL_H - LABEL_H
CANVAS_W = PANEL_W * 4
CANVAS_H = PANEL_H * 2


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    return cap


def read_frame(
    cap: cv2.VideoCapture, path: Path, frame_idx: int
) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    return frame


def fit_content(frame: np.ndarray) -> np.ndarray:
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


def make_panel(
    frame: np.ndarray,
    label: str,
    accent: tuple[int, int, int],
) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[LABEL_H:] = fit_content(frame)
    cv2.rectangle(panel, (0, 0), (PANEL_W, LABEL_H), (16, 16, 16), -1)
    cv2.rectangle(panel, (0, 0), (7, LABEL_H), accent, -1)
    scale = 0.54 if len(label) > 22 else 0.64
    cv2.putText(
        panel, label, (18, 29), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (245, 245, 245), 2, cv2.LINE_AA,
    )
    cv2.rectangle(
        panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (48, 48, 48), 1
    )
    return panel


def resize_mask(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.shape != frame.shape[:2]:
        binary = cv2.resize(
            binary,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return binary.astype(bool)


def mask_overlay(
    raw: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    active = resize_mask(mask, raw)
    out = raw.copy()
    tint = np.empty_like(raw)
    tint[:] = color
    out[active] = cv2.addWeighted(raw, 0.30, tint, 0.70, 0.0)[active]
    contours, _ = cv2.findContours(
        active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def delta_overlay(
    raw: np.ndarray,
    off_mask: np.ndarray,
    on_mask: np.ndarray,
) -> np.ndarray:
    off = resize_mask(off_mask, raw)
    on = resize_mask(on_mask, raw)
    common = off & on
    removed = off & ~on
    added = on & ~off

    out = (raw.astype(np.float32) * 0.35).astype(np.uint8)
    colors = (
        (common, (180, 100, 40)),
        (removed, (0, 0, 255)),
        (added, (0, 255, 0)),
    )
    for active, color in colors:
        if active.any():
            tint = np.empty_like(raw)
            tint[:] = color
            out[active] = cv2.addWeighted(
                raw, 0.20, tint, 0.80, 0.0
            )[active]
    return out


def frame_count(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    pd = args.processed_demo
    paths = {
        "raw": pd / "video_L.mp4",
        "bg_off": (
            pd / "inpaint_processor" / "video_human_inpaint_no_smooth.mkv"
        ),
        "bg_on": pd / "inpaint_processor" / "video_human_inpaint.mkv",
        "final_off": pd / "video_overlay_rby1_xhand_no_smooth.mp4",
        "final_on": pd / "video_overlay_rby1_xhand.mp4",
    }
    mask_off_path = (
        pd / "segmentation_processor" / "masks_arm_no_smooth.npy"
    )
    mask_on_path = pd / "segmentation_processor" / "masks_arm.npy"
    for path in (*paths.values(), mask_off_path, mask_on_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    captures = {name: open_video(path) for name, path in paths.items()}
    expected = frame_count(captures["raw"])
    fps = args.fps or captures["raw"].get(cv2.CAP_PROP_FPS) or 30.0
    off_masks = np.load(mask_off_path, mmap_mode="r")
    on_masks = np.load(mask_on_path, mmap_mode="r")

    if len(off_masks) != expected or len(on_masks) != expected:
        raise ValueError(
            f"mask frame mismatch: raw={expected}, "
            f"off={len(off_masks)}, on={len(on_masks)}"
        )
    for name, cap in captures.items():
        count = frame_count(cap)
        if count > 0 and count != expected:
            raise ValueError(
                f"{name} frame mismatch: expected={expected}, got={count}"
            )

    output = args.output or (pd / "sam2_smoothing_comparison.mp4")
    preview = output.with_name(f"{output.stem}_preview.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", f"{fps:g}",
        "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    preview_frame = None
    labels = (
        ("1  RAW", (210, 210, 210)),
        ("2  MASK - SMOOTH OFF", (0, 80, 255)),
        ("3  MASK - SMOOTH ON", (0, 200, 255)),
        ("4  DELTA  RED:-  GREEN:+", (255, 160, 70)),
        ("5  INPAINT - OFF", (255, 120, 70)),
        ("6  INPAINT - ON", (255, 210, 70)),
        ("7  FINAL - OFF", (255, 120, 255)),
        ("8  FINAL - ON", (255, 255, 255)),
    )

    try:
        for frame_idx in range(expected):
            raw = read_frame(captures["raw"], paths["raw"], frame_idx)
            bg_off = read_frame(captures["bg_off"], paths["bg_off"], frame_idx)
            bg_on = read_frame(captures["bg_on"], paths["bg_on"], frame_idx)
            final_off = read_frame(
                captures["final_off"], paths["final_off"], frame_idx
            )
            final_on = read_frame(
                captures["final_on"], paths["final_on"], frame_idx
            )
            off = np.asarray(off_masks[frame_idx])
            on = np.asarray(on_masks[frame_idx])
            frames = (
                raw,
                mask_overlay(raw, off, (0, 0, 255)),
                mask_overlay(raw, on, (0, 180, 255)),
                delta_overlay(raw, off, on),
                bg_off,
                bg_on,
                final_off,
                final_on,
            )
            panels = [
                make_panel(frame, label, accent)
                for frame, (label, accent) in zip(frames, labels)
            ]
            canvas = np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:])))
            if frame_idx == expected // 2:
                preview_frame = canvas.copy()
            assert encoder.stdin is not None
            encoder.stdin.write(canvas.tobytes())
            if (frame_idx + 1) % 100 == 0:
                print(f"{frame_idx + 1}/{expected}", flush=True)
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
        for cap in captures.values():
            cap.release()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    if preview_frame is not None:
        cv2.imwrite(str(preview), preview_frame)
    print(f"[ok] video: {output}")
    print(f"[ok] preview: {preview}")


if __name__ == "__main__":
    main()
