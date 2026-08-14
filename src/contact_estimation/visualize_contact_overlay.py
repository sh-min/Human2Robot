"""Project HaCo contact onto the source frames to verify alignment.

The split-depth map built by ``build_contact_split_depth.py`` is only as good
as the projection of HaCo's contact vertices into image space. A wrong focal
length or principal point shifts the whole contact surface and silently
corrupts the front/behind decision, so check it here before compositing.

Left panel: source RGB with the hand's vertices coloured by contact
probability (blue = no contact, red = certain contact).
Right panel: per-finger score bars with the hysteresis state.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PANEL_W = 360


def probability_color(prob: np.ndarray) -> np.ndarray:
    """Blue -> green -> red as contact probability rises (BGR)."""
    ramp = np.clip(prob * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(ramp, cv2.COLORMAP_JET).reshape(-1, 3)


def draw_side_panel(height: int, score: np.ndarray, state: np.ndarray,
                    valid: np.ndarray, frame_idx: int, total: int,
                    on_threshold: float) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), 22, np.uint8)
    cv2.putText(panel, "HaCo contact", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(panel, f"frame {frame_idx + 1}/{total}", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 150, 150), 1, cv2.LINE_AA)

    y = 96
    for side_idx, side in enumerate(SIDES):
        if not valid[side_idx]:
            continue
        cv2.putText(panel, side.upper(), (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)
        y += 26
        for f_idx, finger in enumerate(FINGERS):
            value = float(score[side_idx, f_idx])
            touching = bool(state[side_idx, f_idx])
            colour = (70, 200, 90) if touching else (110, 110, 110)
            cv2.putText(panel, finger, (24, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (210, 210, 210), 1, cv2.LINE_AA)
            x0, x1 = 116, PANEL_W - 56
            cv2.rectangle(panel, (x0, y), (x1, y + 17), (48, 48, 48), -1)
            fill = int(x0 + (x1 - x0) * min(max(value, 0.0), 1.0))
            cv2.rectangle(panel, (x0, y), (fill, y + 17), colour, -1)
            gate = int(x0 + (x1 - x0) * on_threshold)
            cv2.line(panel, (gate, y - 2), (gate, y + 19), (0, 165, 255), 1)
            cv2.putText(panel, f"{value:.2f}", (x1 + 6, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
                        cv2.LINE_AA)
            y += 26
        y += 14

    cv2.putText(panel, "bar = top-25% vertex prob", (16, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)
    cv2.putText(panel, "orange = on-threshold", (16, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 220), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--img_glob", default="*.jpg")
    parser.add_argument("--contact_dir", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--finger_contact", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--min_probability", type=float, default=0.0,
                        help="Hide vertices below this probability.")
    args = parser.parse_args()

    hawor = np.load(args.hawor_npz)
    valid = hawor["valid"]
    focal = float(hawor["img_focal"])
    verts = {side: hawor[f"verts_{side}"] for side in SIDES}

    images = sorted(args.frames_dir.glob(args.img_glob))
    contact_files = sorted(p for p in args.contact_dir.glob("*.npz")
                           if p.name != "finger_contact.npz")
    total = min(len(images), len(contact_files), valid.shape[1])
    if total == 0:
        raise RuntimeError("no frames to visualize")

    finger_path = args.finger_contact or (args.contact_dir / "finger_contact.npz")
    finger = np.load(finger_path)
    score, state = finger["score"], finger["state"]
    on_threshold = float(finger["on_threshold"])

    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    cx, cy = width / 2.0, height / 2.0

    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (width + PANEL_W, height))
        if not writer.isOpened():
            raise RuntimeError("cannot open temporary writer")

        for t in range(total):
            frame = cv2.imread(str(images[t]))
            data = np.load(contact_files[t])
            for side_idx, side in enumerate(SIDES):
                if not valid[side_idx, t] or not bool(data[f"{side}_valid"]):
                    continue
                prob = data[f"{side}_contact_probability"].astype(np.float32)
                pts = verts[side][t]
                z = pts[:, 2]
                keep = (z > 1e-3) & (prob >= args.min_probability)
                if not keep.any():
                    continue
                u = (focal * pts[keep, 0] / z[keep] + cx).astype(np.int32)
                v = (focal * pts[keep, 1] / z[keep] + cy).astype(np.int32)
                colours = probability_color(prob[keep])
                inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
                # Draw the most certain vertices last so contact stays visible
                # where the hand surface folds onto itself.
                order = np.argsort(prob[keep][inside])
                for uu, vv, bgr in zip(u[inside][order], v[inside][order],
                                       colours[inside][order]):
                    cv2.circle(frame, (int(uu), int(vv)), args.point_radius,
                               tuple(int(c) for c in bgr), -1, cv2.LINE_AA)

            panel = draw_side_panel(height, score[t], state[t], valid[:, t],
                                    t, total, on_threshold)
            writer.write(np.hstack([frame, panel]))
        writer.release()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
             "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-crf", "18", str(args.output)],
            check=True,
        )

    print(f"[ok] {args.output}  frames={total}  {width + PANEL_W}x{height}")


if __name__ == "__main__":
    main()
