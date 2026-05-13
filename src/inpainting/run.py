"""End-to-end inpainting + xhand overlay pipeline.

All stages are local scripts in this directory — no phantom dependency.

Stages:
    1. prepare_demo.py        rgb frames     -> video_L.mp4 in demo layout
    2. inject_hawor_data.py   HaWoR keypoints -> bbox_data + hand_data + video_rgb_imgs.mkv
    3. segment_arms.py        SAM2            -> segmentation_processor/masks_arm.npy
    4. inpaint_hands.py       E2FGVI          -> inpaint_processor/video_human_inpaint.mkv
    5. render_xhand_overlay.py pyrender + xhand qpos -> video_overlay_xhand.mkv

Each stage skips itself if its primary output already exists, so this script
is re-runnable.

Usage:
    conda activate <inpaint-env>
    # `--input` accepts either a directory of jpg/png frames or a raw video file.
    python run.py \
        --input /data/RFM_proj/cam0_hawor/extracted_images \
        --hawor_npz  /data/RFM_proj/cam0_hawor/retarget_input.npz \
        --right_pkl  /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl   /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl \
        --data_root      /data/RFM_proj/cam0_inpaint_raw \
        --processed_root /result/cam0_inpaint
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def _run(cmd, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="Directory of jpg/png frames OR a raw video file")
    ap.add_argument("--hawor_npz", type=Path, required=True,
                    help="retarget_input.npz from src/hand_estimation/")
    ap.add_argument("--right_pkl", type=Path, required=True,
                    help="qpos_xhand_right.pkl from src/retargeting/")
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True,
                    help="Holds raw demo (video_L.mp4 layout)")
    ap.add_argument("--processed_root", type=Path, required=True,
                    help="Holds all intermediate + final outputs")
    ap.add_argument("--demo_name", default="cam0")
    ap.add_argument("--demo_num",  default="0")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override output fps. Frames-dir default=10, "
                         "video-file default=source fps.")
    ap.add_argument("--glob", default="*.jpg",
                    help="(frames-dir mode only) glob pattern for images")
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    args = ap.parse_args()

    processed_demo = args.processed_root / args.demo_name / args.demo_num

    # Stage 1: prepare demo (video_L.mp4 + copytree)
    prepare_cmd = [sys.executable, str(HERE / "prepare_demo.py"),
                   "--input", args.input,
                   "--data_root",  args.data_root,
                   "--processed_root", args.processed_root,
                   "--demo_name", args.demo_name,
                   "--demo_num",  args.demo_num,
                   "--glob", args.glob]
    if args.fps is not None:
        prepare_cmd += ["--fps", str(args.fps)]
    _run(prepare_cmd)

    # Stage 2: inject HaWoR data (bbox + hand_data + ffv1 video)
    _run([sys.executable, str(HERE / "inject_hawor_data.py"),
          "--processed_demo", processed_demo,
          "--hawor_npz", args.hawor_npz])

    # Stage 3: SAM2 arm segmentation
    masks_npy = processed_demo / "segmentation_processor" / "masks_arm.npy"
    if masks_npy.exists():
        print(f"\n[skip] {masks_npy} exists")
    else:
        _run([sys.executable, str(HERE / "segment_arms.py"),
              "--processed_demo", processed_demo])

    # Stage 4: E2FGVI inpainting
    inpaint_mkv = processed_demo / "inpaint_processor" / "video_human_inpaint.mkv"
    if inpaint_mkv.exists() and inpaint_mkv.stat().st_size > 0:
        print(f"\n[skip] {inpaint_mkv} exists ({inpaint_mkv.stat().st_size // 1024 // 1024} MB)")
    else:
        _run([sys.executable, str(HERE / "inpaint_hands.py"),
              "--processed_demo", processed_demo])

    # Stage 5: pyrender xhand overlay
    overlay_mkv = processed_demo / "video_overlay_xhand.mkv"
    if overlay_mkv.exists() and overlay_mkv.stat().st_size > 0:
        print(f"\n[skip] {overlay_mkv} exists")
    else:
        _run([sys.executable, "-u", str(HERE / "render_xhand_overlay.py"),
              "--processed_demo", processed_demo,
              "--hawor_npz", args.hawor_npz,
              "--right_pkl", args.right_pkl,
              "--left_pkl",  args.left_pkl,
              "--hand", args.hand])

    print(f"\n[done] final overlay: {overlay_mkv}")


if __name__ == "__main__":
    main()
