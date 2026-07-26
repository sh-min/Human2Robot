"""Build a demo folder from raw rgb input (frames directory OR video file).

Output layout (matches phantom's convention so injected bbox/hand_data work):
    <data_root>/<demo_name>/<demo_num>/video_L.mp4
    <processed_root>/<demo_name>/<demo_num>/video_L.mp4   (copy of the above)

`--input` may be:
    - a directory of sequential jpg/png frames  (uses --fps, default 10)
    - a video file readable by cv2/mediapy      (FPS taken from the file)

Usage (frames):
    python prepare_demo.py \
        --input /data/RFM_proj/cam0_hawor/extracted_images \
        --data_root  /data/RFM_proj/cam0_inpaint_raw \
        --processed_root /result/cam0_inpaint \
        --demo_name  cam0 --demo_num 0 --fps 10

Usage (video file):
    python prepare_demo.py \
        --input /data/cam0_raw.mp4 \
        --data_root /data/RFM_proj/cam0_inpaint_raw \
        --processed_root /result/cam0_inpaint
"""
import argparse
import shutil
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _build_from_frames(frames_dir: Path, dst_mp4: Path, fps: float, glob: str) -> int:
    paths = sorted(frames_dir.glob(glob))
    if not paths:
        paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No images matched in {frames_dir}")
    frames = np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(dst_mp4), frames, fps=fps, codec="libx264")
    return len(paths)


def _build_from_video(src: Path, dst_mp4: Path, fps_override: float | None) -> tuple[int, float]:
    """Re-encode the source video to libx264 mp4 so downstream sees a uniform format.

    FPS comes from the source unless --fps was explicitly set."""
    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    cap.release()
    fps = fps_override if fps_override is not None else src_fps
    frames = media.read_video(str(src))
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(dst_mp4), frames, fps=fps, codec="libx264")
    return len(frames), fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="Directory of frames OR a video file")
    ap.add_argument("--data_root", type=Path, required=True,
                    help="Output data_root_dir (will hold raw demo with video_L.mp4)")
    ap.add_argument("--processed_root", type=Path, required=True,
                    help="Output processed_data_root_dir (copy destination)")
    ap.add_argument("--demo_name", default="cam0")
    ap.add_argument("--demo_num",  default="0")
    ap.add_argument("--fps", type=float, default=None,
                    help="Output FPS. For frames dirs defaults to 10; for video files "
                         "defaults to the source's own FPS.")
    ap.add_argument("--glob", default="*.jpg",
                    help="(frames-dir mode only) glob pattern for images")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rebuild video_L.mp4 and refresh the processed copy")
    args = ap.parse_args()

    raw_dir = args.data_root / args.demo_name / args.demo_num
    raw_mp4 = raw_dir / "video_L.mp4"

    if raw_mp4.exists() and not args.overwrite:
        print(f"[skip] {raw_mp4} exists")
    elif args.input.is_dir():
        fps = args.fps if args.fps is not None else 10.0
        n = _build_from_frames(args.input, raw_mp4, fps, args.glob)
        print(f"[ok] wrote {raw_mp4}  ({n} frames @ {fps} fps, from frames dir)")
    elif args.input.is_file():
        n, fps = _build_from_video(args.input, raw_mp4, args.fps)
        print(f"[ok] wrote {raw_mp4}  ({n} frames @ {fps:.2f} fps, from {args.input.name})")
    else:
        raise FileNotFoundError(f"--input not a dir or file: {args.input}")

    processed_dir = args.processed_root / args.demo_name / args.demo_num
    if processed_dir.exists() and not args.overwrite:
        print(f"[skip] {processed_dir} already copied")
    elif processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_mp4, processed_dir / "video_L.mp4")
        print(f"[ok] refreshed {processed_dir / 'video_L.mp4'}")
    else:
        processed_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(raw_dir, processed_dir)
        print(f"[ok] copied raw demo -> {processed_dir}")


if __name__ == "__main__":
    main()
