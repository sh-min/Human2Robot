"""End-to-end orchestrator for the locked-in layered overlay pipeline.

Stages (each skips itself if its primary output already exists):

    1. prepare_demo.py                      rgb/video → video_L.mp4 (demo layout)
    2. inject_hawor_data.py                 HaWoR → bbox + hand_data + video_rgb_imgs.mkv
    3. segment_arms.py                      SAM2 → segmentation_processor/masks_arm.npy
    4. inpaint_hands.py --mode legacy       E2FGVI → inpaint_processor/video_human_inpaint.mkv
    5. render_xhand_overlay_depth.py        pyrender → overlay_processor/robot_{rgb,depth,mask}.npy
    6. estimate_depth.py                    Depth Anything V2 → depth_processor/depth_raw.npy
    7. align_depth.py                       HaWoR Z anchors → depth_processor/depth_aligned.npy
    8. crop_cube_layer.py                   top-q depth + center-CC → cube_layer/cube_mask_raw.npy
    9. regularize_and_cut_cube.py           open/CC/close/hull/outlier/SDF →
                                                            cube_layer/cube_mask_clean.npy
   10. composite_layered.py                 4-layer alpha-blend →
                                                            overlay_processor_layered/video_overlay.mp4

Usage:
    conda activate inpaint
    python run_layered.py \
        --input        /data/RFM_proj/cam0_hawor/extracted_images \
        --hawor_npz    /data/RFM_proj/cam0_hawor/retarget_input.npz \
        --right_pkl    /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl     /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl \
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
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--processed_root", type=Path, required=True)
    ap.add_argument("--demo_name", default="cam0")
    ap.add_argument("--demo_num",  default="0")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")

    # depth + cube layer knobs
    ap.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--cube_quantile", type=float, default=0.25,
                    help="quantile of non-hand depth used as cube isolation threshold")
    ap.add_argument("--sdf_sigma", type=float, default=8.0,
                    help="temporal SDF Gaussian sigma (frames) for cube mask")
    ap.add_argument("--area_mad_k", type=float, default=5.0,
                    help="MAD multiplier for area-outlier rejection")
    ap.add_argument("--centroid_max_jump", type=float, default=120.0,
                    help="centroid outlier threshold (px)")

    # compositor knobs
    ap.add_argument("--threshold_joint", type=int, default=5,
                    help="MANO joint used for front/behind split (default 5 = idx-MCP)")
    ap.add_argument("--zmcp_sigma_t", type=float, default=8.0)
    ap.add_argument("--edge_sigma", type=float, default=1.5)

    args = ap.parse_args()

    pd = args.processed_root / args.demo_name / args.demo_num

    # Stage 1: prepare demo
    prepare_cmd = [sys.executable, str(HERE / "prepare_demo.py"),
                   "--input", args.input,
                   "--data_root", args.data_root,
                   "--processed_root", args.processed_root,
                   "--demo_name", args.demo_name,
                   "--demo_num",  args.demo_num,
                   "--glob", args.glob]
    if args.fps is not None:
        prepare_cmd += ["--fps", str(args.fps)]
    _run(prepare_cmd)

    # Stage 2: inject HaWoR data
    _run([sys.executable, str(HERE / "inject_hawor_data.py"),
          "--processed_demo", pd,
          "--hawor_npz", args.hawor_npz])

    # Stage 3: SAM2 hand seg
    arm_npy = pd / "segmentation_processor" / "masks_arm.npy"
    if arm_npy.exists():
        print(f"\n[skip] {arm_npy} exists")
    else:
        _run([sys.executable, str(HERE / "segment_arms.py"), "--processed_demo", pd])

    # Stage 4: legacy E2FGVI on hand mask → inpainted bg
    inp_bg = pd / "inpaint_processor" / "video_human_inpaint.mkv"
    if inp_bg.exists() and inp_bg.stat().st_size > 0:
        print(f"\n[skip] {inp_bg} exists")
    else:
        _run([sys.executable, str(HERE / "inpaint_hands.py"),
              "--processed_demo", pd, "--mode", "legacy"])

    # Stage 5: pyrender robot RGBD
    robot_mask_npy = pd / "overlay_processor" / "robot_mask.npy"
    if robot_mask_npy.exists():
        print(f"\n[skip] {robot_mask_npy} exists")
    else:
        _run([sys.executable, "-u", str(HERE / "render_xhand_overlay_depth.py"),
              "--processed_demo", pd,
              "--hawor_npz", args.hawor_npz,
              "--right_pkl", args.right_pkl,
              "--left_pkl",  args.left_pkl,
              "--hand", args.hand])

    # Stage 6: depth estimation (raw video)
    depth_raw = pd / "depth_processor" / "depth_raw.npy"
    if depth_raw.exists():
        print(f"\n[skip] {depth_raw} exists")
    else:
        _run([sys.executable, str(HERE / "estimate_depth.py"),
              "--processed_demo", pd, "--encoder", args.encoder])

    # Stage 7: metric alignment
    depth_aligned = pd / "depth_processor" / "depth_aligned.npy"
    if depth_aligned.exists():
        print(f"\n[skip] {depth_aligned} exists")
    else:
        _run([sys.executable, str(HERE / "align_depth.py"), "--processed_demo", pd])

    # Stage 8: rough cube mask
    cube_raw = pd / "cube_layer" / "cube_mask_raw.npy"
    if cube_raw.exists():
        print(f"\n[skip] {cube_raw} exists")
    else:
        _run([sys.executable, str(HERE / "crop_cube_layer.py"),
              "--processed_demo", pd,
              "--quantile", str(args.cube_quantile),
              "--cc_pick", "center"])

    # Stage 9: regularize + temporal SDF smoothing
    cube_clean = pd / "cube_layer" / "cube_mask_clean.npy"
    if cube_clean.exists():
        print(f"\n[skip] {cube_clean} exists")
    else:
        _run([sys.executable, str(HERE / "regularize_and_cut_cube.py"),
              "--processed_demo", pd,
              "--sdf_sigma", str(args.sdf_sigma),
              "--area_mad_k", str(args.area_mad_k),
              "--centroid_max_jump", str(args.centroid_max_jump)])

    # Stage 10: final layered composite
    final_mp4 = pd / "overlay_processor_layered" / "video_overlay.mp4"
    if final_mp4.exists():
        print(f"\n[skip] {final_mp4} exists")
    else:
        _run([sys.executable, str(HERE / "composite_layered.py"),
              "--processed_demo", pd,
              "--hawor_npz", args.hawor_npz,
              "--cube_mask_npy", "cube_layer/cube_mask_clean.npy",
              "--threshold_joint", str(args.threshold_joint),
              "--zmcp_sigma_t", str(args.zmcp_sigma_t),
              "--edge_sigma", str(args.edge_sigma)])

    print(f"\n[done] final layered overlay: {final_mp4}")


if __name__ == "__main__":
    main()
