"""Create easy-to-read HaWoR + HaCo comparison videos.

Each output frame contains:

    ORIGINAL + hand ROI | HaWoR MANO projection (zoom) | HaCo contact mesh

HaWoR vertices and joints are projected directly from ``retarget_input.npz``.
The HaCo panel reuses the per-frame 3D contact visualization produced by
``extract_hand_contact.py``; green vertices are predicted contact regions.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SIDE_INDEX = {"left": 0, "right": 1}
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def project(points: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    """Project camera-space XYZ points into image coordinates."""
    points = np.asarray(points)
    z = points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    good = np.isfinite(points).all(axis=1) & (z > 1e-6)
    uv[good, 0] = focal * points[good, 0] / z[good] + width / 2.0
    uv[good, 1] = focal * points[good, 1] / z[good] + height / 2.0
    return uv


def crop_square(image: np.ndarray, x0: float, y0: float, side: float,
                output_size: int) -> np.ndarray:
    """Crop a possibly out-of-frame square and resize it without distortion."""
    side_i = max(2, int(round(side)))
    x0_i, y0_i = int(round(x0)), int(round(y0))
    x1_i, y1_i = x0_i + side_i, y0_i + side_i
    height, width = image.shape[:2]

    canvas = np.zeros((side_i, side_i, 3), dtype=np.uint8)
    src_x0, src_y0 = max(0, x0_i), max(0, y0_i)
    src_x1, src_y1 = min(width, x1_i), min(height, y1_i)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x0_i, src_y0 - y0_i
        canvas[
            dst_y0:dst_y0 + (src_y1 - src_y0),
            dst_x0:dst_x0 + (src_x1 - src_x0),
        ] = image[src_y0:src_y1, src_x0:src_x1]
    return cv2.resize(canvas, (output_size, output_size),
                      interpolation=cv2.INTER_AREA)


def fit_square(image: np.ndarray, output_size: int) -> np.ndarray:
    """Letterbox an image into a square panel."""
    height, width = image.shape[:2]
    scale = min(output_size / width, output_size / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    panel = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    x0 = (output_size - new_w) // 2
    y0 = (output_size - new_h) // 2
    panel[y0:y0 + new_h, x0:x0 + new_w] = resized
    return panel


def choose_side(data: np.lib.npyio.NpzFile, frame_idx: int) -> str | None:
    valid = np.asarray(data["valid"])
    for side in ("left", "right"):
        side_idx = SIDE_INDEX[side]
        if 0 <= frame_idx < valid.shape[1] and bool(valid[side_idx, frame_idx]):
            return side
    return None


def hawor_zoom_panel(
    frame: np.ndarray,
    verts: np.ndarray,
    joints: np.ndarray,
    focal: float,
    panel_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Render a zoomed MANO point cloud and skeleton over the RGB frame."""
    height, width = frame.shape[:2]
    uv_joints = project(joints, focal, width, height)
    uv_verts = project(verts, focal, width, height)
    good_joints = np.isfinite(uv_joints).all(axis=1)

    if not good_joints.any():
        return fit_square(frame, panel_size), (0, 0, width - 1, height - 1)

    hand_uv = uv_joints[good_joints]
    center = hand_uv.mean(axis=0)
    span = np.ptp(hand_uv, axis=0)
    crop_side = float(np.clip(max(span.max() * 2.25, 300.0), 300.0,
                              max(width, height)))
    x0 = float(center[0] - crop_side / 2.0)
    y0 = float(center[1] - crop_side / 2.0)
    panel = crop_square(frame, x0, y0, crop_side, panel_size)

    scale = panel_size / crop_side
    panel_verts = (uv_verts - np.array([x0, y0], dtype=np.float32)) * scale
    panel_joints = (uv_joints - np.array([x0, y0], dtype=np.float32)) * scale

    overlay = panel.copy()
    for point in panel_verts:
        if np.isfinite(point).all():
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= x < panel_size and 0 <= y < panel_size:
                cv2.circle(overlay, (x, y), 2, (255, 255, 0), -1,
                           lineType=cv2.LINE_AA)
    panel = cv2.addWeighted(panel, 0.42, overlay, 0.58, 0.0)

    for start, end in HAND_EDGES:
        if start >= len(panel_joints) or end >= len(panel_joints):
            continue
        p0, p1 = panel_joints[start], panel_joints[end]
        if np.isfinite(p0).all() and np.isfinite(p1).all():
            cv2.line(
                panel,
                tuple(np.round(p0).astype(int)),
                tuple(np.round(p1).astype(int)),
                (0, 255, 255),
                3,
                lineType=cv2.LINE_AA,
            )
    for point in panel_joints:
        if np.isfinite(point).all():
            cv2.circle(panel, tuple(np.round(point).astype(int)), 5,
                       (0, 80, 255), -1, lineType=cv2.LINE_AA)

    roi = (
        int(round(x0)),
        int(round(y0)),
        int(round(x0 + crop_side)),
        int(round(y0 + crop_side)),
    )
    return panel, roi


def haco_panel(viz_path: Path, side: str | None, panel_size: int) -> np.ndarray:
    """Extract the requested 400x400 HaCo mesh panel from its visualization."""
    viz = cv2.imread(str(viz_path))
    if viz is None or viz.shape[0] < 400 or viz.shape[1] < 800:
        return np.full((panel_size, panel_size, 3), 235, dtype=np.uint8)

    mesh_size = viz.shape[0]
    rgb_width = viz.shape[1] - 2 * mesh_size
    side_offset = 0 if side != "right" else mesh_size
    x0 = max(0, rgb_width + side_offset)
    mesh = viz[:, x0:x0 + mesh_size]
    if mesh.size == 0:
        return np.full((panel_size, panel_size, 3), 235, dtype=np.uint8)
    return cv2.resize(mesh, (panel_size, panel_size),
                      interpolation=cv2.INTER_AREA)


def draw_header(canvas: np.ndarray, x0: int, width: int, title: str,
                subtitle: str) -> None:
    cv2.putText(canvas, title, (x0 + 18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.86, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (x0 + 18, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (175, 205, 220), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0 + width - 1, 0), (x0 + width - 1, canvas.shape[0]),
             (65, 65, 65), 1)


def compose_frame(
    frame: np.ndarray,
    data: np.lib.npyio.NpzFile,
    frame_idx: int,
    frame_name: str,
    haco_viz_path: Path,
    contact_path: Path,
    focal: float,
    panel_size: int,
    recording: str,
) -> np.ndarray:
    side = choose_side(data, frame_idx)
    if side is not None:
        verts = np.asarray(data[f"verts_{side}"][frame_idx])
        joints = np.asarray(data[f"joints_{side}"][frame_idx])
        hawor, roi = hawor_zoom_panel(
            frame, verts, joints, focal=focal, panel_size=panel_size
        )
    else:
        hawor = fit_square(frame, panel_size)
        roi = (0, 0, frame.shape[1] - 1, frame.shape[0] - 1)

    original_marked = frame.copy()
    x0, y0, x1, y1 = roi
    cv2.rectangle(
        original_marked,
        (max(0, x0), max(0, y0)),
        (min(frame.shape[1] - 1, x1), min(frame.shape[0] - 1, y1)),
        (255, 0, 255),
        8,
        lineType=cv2.LINE_AA,
    )
    original = fit_square(original_marked, panel_size)
    haco = haco_panel(haco_viz_path, side, panel_size)

    contact_count = 0
    if side is not None and contact_path.is_file():
        contact = np.load(contact_path)
        mask_key = f"{side}_contact_mask"
        if mask_key in contact:
            contact_count = int(np.asarray(contact[mask_key]).sum())

    header_h, footer_h = 66, 42
    canvas = np.zeros(
        (header_h + panel_size + footer_h, panel_size * 3, 3),
        dtype=np.uint8,
    )
    canvas[header_h:header_h + panel_size, 0:panel_size] = original
    canvas[header_h:header_h + panel_size, panel_size:panel_size * 2] = hawor
    canvas[header_h:header_h + panel_size, panel_size * 2:] = haco

    side_text = side.upper() if side else "NO VALID HAND"
    draw_header(canvas, 0, panel_size, "ORIGINAL", "magenta box = zoom region")
    draw_header(
        canvas, panel_size, panel_size,
        "HaWoR MANO", f"{side_text} | cyan vertices + yellow joints",
    )
    draw_header(
        canvas, panel_size * 2, panel_size,
        "HaCo CONTACT", f"{side_text} | green contact vertices: {contact_count}",
    )
    cv2.putText(
        canvas,
        f"{recording}   frame {frame_name}",
        (18, header_h + panel_size + 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return canvas


def encode_episode(
    episode: Path,
    focal: float,
    fps: int,
    panel_size: int,
) -> tuple[Path, Path]:
    rgb_dir = episode / "rgb"
    hawor_path = episode / "rgb_hawor" / "retarget_input.npz"
    contact_dir = episode / "contact"
    viz_dir = contact_dir / "viz"
    if not rgb_dir.is_dir() or not hawor_path.is_file() or not viz_dir.is_dir():
        raise FileNotFoundError(f"Missing HaWoR/HaCo inputs under {episode}")

    images = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    data = np.load(hawor_path)
    start_idx = int(data["start_idx"])

    output_dir = episode / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hawor_haco_comparison.mp4"
    preview_path = output_dir / "hawor_haco_preview.jpg"

    output_width = panel_size * 3
    output_height = 66 + panel_size + 42
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    preview = None
    try:
        for image_idx, image_path in enumerate(images):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")
            frame_idx = image_idx - start_idx
            canvas = compose_frame(
                frame=frame,
                data=data,
                frame_idx=frame_idx,
                frame_name=image_path.stem,
                haco_viz_path=viz_dir / f"{image_path.stem}.png",
                contact_path=contact_dir / f"{image_path.stem}.npz",
                focal=focal,
                panel_size=panel_size,
                recording=episode.name,
            )
            if image_idx == len(images) // 2:
                preview = canvas.copy()
            assert encoder.stdin is not None
            encoder.stdin.write(canvas.tobytes())
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
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
    thumb_w = 810
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
    parser.add_argument(
        "--recording_glob",
        default="IMG_*",
        help="Comma-separated episode names/globs under data_root",
    )
    parser.add_argument("--focal", type=float, default=1220.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--panel_size", type=int, default=540)
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
        output, preview = encode_episode(
            episode, focal=args.focal, fps=args.fps, panel_size=args.panel_size
        )
        previews.append(preview)
        print(f"[{index}/{len(episodes)}] {episode.name}: {output}")

    overview_path = args.data_root / "hawor_haco_overview.jpg"
    make_overview(previews, overview_path)
    print(f"Overview: {overview_path}")


if __name__ == "__main__":
    main()
