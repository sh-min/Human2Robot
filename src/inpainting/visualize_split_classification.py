"""Show which robot pixels the split depth puts in front of / behind the object.

The composite only reveals the change where the object layer happens to cover
the robot, so a direct view of the classification makes the effect of a
contact-derived split surface auditable frame by frame.

Left: the scalar plane the compositor uses by default.
Right: the HaCo contact split-depth map.
Green = robot drawn in front, red = robot drawn behind, yellow outline = the
object mask, and the caption counts pixels whose class differs between the two.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

LABEL_H = 40


def scalar_split_depth(joints_left, joints_right, valid, frame_count,
                       joint, sigma):
    z = np.full(frame_count, np.nan, dtype=np.float32)
    for idx in range(frame_count):
        values = []
        if idx < joints_left.shape[0] and valid[0, idx]:
            values.append(float(joints_left[idx, joint, 2]))
        if idx < joints_right.shape[0] and valid[1, idx]:
            values.append(float(joints_right[idx, joint, 2]))
        if values:
            z[idx] = float(np.mean(values))
    good = np.flatnonzero(np.isfinite(z))
    if not len(good):
        return np.full(frame_count, np.inf, dtype=np.float32)
    z = np.interp(np.arange(frame_count), good, z[good]).astype(np.float32)
    if sigma > 0:
        radius = max(1, int(np.ceil(3 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        z = np.convolve(np.pad(z, (radius, radius), mode="edge"), kernel,
                        mode="valid").astype(np.float32)
    return z


def paint(base, robot, depth, split, object_mask):
    front = robot & (depth < split)
    behind = robot & (depth >= split)
    out = (base * 0.45).astype(np.uint8)
    out[front] = (0.35 * out[front] + 0.65 * np.array([90, 230, 90])).astype(np.uint8)
    out[behind] = (0.35 * out[behind] + 0.65 * np.array([70, 70, 235])).astype(np.uint8)
    edges = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                             np.ones((3, 3), np.uint8)).astype(bool)
    out[edges] = (40, 220, 235)
    return out, front, behind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--robot_dir", default="overlay_rb5_desk")
    parser.add_argument("--object_mask",
                        default="interaction_objects/refined/object_mask_refined.npy")
    parser.add_argument("--contact_split_depth",
                        default="haco_in/contact_split_depth.npy")
    parser.add_argument("--source_video", default="video_L.mp4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold_joint", type=int, default=5)
    parser.add_argument("--depth_sigma", type=float, default=8.0)
    parser.add_argument("--panel_width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()

    demo = args.processed_demo
    robot_depth = np.load(demo / args.robot_dir / "robot_depth.npy", mmap_mode="r")
    robot_mask = np.load(demo / args.robot_dir / "robot_mask.npy", mmap_mode="r")
    object_mask = np.load(demo / args.object_mask, mmap_mode="r")
    contact_split = np.load(demo / args.contact_split_depth, mmap_mode="r")
    pose = np.load(args.hawor_npz)

    cap = cv2.VideoCapture(str(demo / args.source_video))
    if not cap.isOpened():
        raise FileNotFoundError(demo / args.source_video)
    frame_count = min(len(robot_depth), len(robot_mask), len(object_mask),
                      len(contact_split),
                      int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    z_split = scalar_split_depth(
        pose["joints_left"], pose["joints_right"], pose["valid"],
        frame_count, args.threshold_joint, args.depth_sigma,
    )

    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    panel_w = args.panel_width
    panel_h = int(round(src_h * panel_w / src_w))
    total_w = panel_w * 2
    total_h = LABEL_H + panel_h + 34
    total_w += total_w % 2
    total_h += total_h % 2

    flipped_total = 0
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (total_w, total_h))
        for t in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            robot = np.asarray(robot_mask[t], bool)
            depth = np.asarray(robot_depth[t], np.float32)
            objm = np.asarray(object_mask[t], bool)

            left, front_a, behind_a = paint(frame, robot, depth,
                                            float(z_split[t]), objm)
            right, front_b, behind_b = paint(
                frame, robot, depth,
                np.asarray(contact_split[t], np.float32), objm)
            flipped = int((front_a ^ front_b).sum())
            flipped_in_object = int(((front_a ^ front_b) & objm).sum())
            flipped_total += flipped

            canvas = np.zeros((total_h, total_w, 3), np.uint8)
            for idx, (panel, text, accent) in enumerate((
                (left, f"scalar plane  z={z_split[t]:.3f} m", (90, 90, 90)),
                (right, "HaCo contact split surface", (90, 200, 90)),
            )):
                bar = np.full((LABEL_H, panel_w, 3), 18, np.uint8)
                cv2.rectangle(bar, (0, 0), (6, LABEL_H), accent, -1)
                cv2.putText(bar, text, (18, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (240, 240, 240), 1, cv2.LINE_AA)
                block = np.vstack([bar, cv2.resize(panel, (panel_w, panel_h),
                                                   interpolation=cv2.INTER_AREA)])
                canvas[:block.shape[0], idx * panel_w:(idx + 1) * panel_w] = block

            caption = (f"frame {t + 1}/{frame_count}   "
                       f"reclassified {flipped:,} px "
                       f"({flipped_in_object:,} under the object)   "
                       f"green=front  red=behind  yellow=object outline")
            cv2.putText(canvas, caption, (14, total_h - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1,
                        cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
        cap.release()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
             "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-crf", "18", str(args.output)],
            check=True,
        )

    print(f"[ok] {args.output}  frames={frame_count}  "
          f"mean reclassified px/frame={flipped_total / max(frame_count, 1):.0f}")


if __name__ == "__main__":
    main()
