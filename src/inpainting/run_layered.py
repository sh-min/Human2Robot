"""End-to-end orchestrator for the locked-in layered overlay pipeline.

Stages (each skips itself if its primary output already exists). By DEFAULT the
pipeline runs 1,2,3,4,5,10 with the Isaac robot renderer and NO object layer.
Stages marked [cube] only run when --cube_layer is passed.

    1. prepare_demo.py                      rgb/video → video_L.mp4 (demo layout)
    2. inject_hawor_data.py                 HaWoR → bbox + hand_data + video_rgb_imgs.mkv
    3. segment_arms.py                      SAM2 → segmentation_processor/masks_arm.npy
  3.5. segment_cube.py            [cube]    SAM2 object mask → cube_mask_raw.npy
    4. inpaint_hands.py --mode legacy       E2FGVI → inpaint_processor/video_human_inpaint.mkv
    5. robot RGBD → overlay_processor/robot_{rgb,depth,mask}.npy
         isaac (default): isaac_stage5.sh → RB5-850 arm + xhand hand (Isaac Sim)
         pyrender (--render_backend pyrender): render_xhand_overlay_depth.py (xhand + RBY1 arm)
    6. estimate_depth.py          [cube]    Depth Anything V2 → depth_processor/depth_raw.npy
    7. align_depth.py             [cube]    HaWoR Z anchors → depth_processor/depth_aligned.npy
  8-9. run_cube_segmentation.py   [cube]    SAM2 modal → Diffusion-VAS amodal → cube_mask_amodal.npy
   10. composite_layered.py                 alpha-blend robot over inpainted bg (+ object
                                            layer when --cube_layer) →
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
    ap.add_argument("--arm_kpts", type=Path, default=None,
                    help="EgoDex arm_kpts_2d.npz for whole-arm SAM2 seeding. "
                         "If omitted, looks for <input>/arm_kpts_2d.npz; absent "
                         "→ segment_arms falls back to hand-bbox seeding.")

    ap.add_argument("--render_backend", choices=["isaac", "pyrender"], default="isaac",
                    help="Stage 5 robot renderer. isaac (default) = RB5-850 arm + xhand "
                         "hand in Isaac Sim (via isaac_stage5.sh). pyrender = legacy "
                         "xhand + RBY1 arm.")
    ap.add_argument("--cube_layer", action="store_true",
                    help="enable the object/cube layer: segment_cube (SAM2, stage 3.5) "
                         "+ depth (6,7) + Diffusion-VAS amodal (8,9), composited over the "
                         "robot. OFF by default — the object layer and its stages are "
                         "skipped entirely.")

    # depth + cube layer knobs (only used when --cube_layer is set)
    ap.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--cube_quantile", type=float, default=0.25,
                    help="non-hand depth quantile for the SAM2 seed-frame bootstrap")
    ap.add_argument("--cube_overlap", type=int, default=0,
                    help="frames shared between Diffusion-VAS 25-frame windows "
                         "(0 = non-overlapping; 4-6 smooths window seams)")
    ap.add_argument("--no_content_completion", action="store_true",
                    help="skip the Diffusion-VAS amodal-RGB pass (amodal mask only)")
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

    # Resolve EgoDex arm keypoints (whole-arm seeding) if available.
    arm_kpts = args.arm_kpts
    if arm_kpts is None and args.input.is_dir():
        cand = args.input / "arm_kpts_2d.npz"
        arm_kpts = cand if cand.exists() else None

    # Stage 3: SAM2 hand/arm seg
    arm_npy = pd / "segmentation_processor" / "masks_arm.npy"
    if arm_npy.exists():
        print(f"\n[skip] {arm_npy} exists")
    else:
        seg_cmd = [sys.executable, str(HERE / "segment_arms.py"), "--processed_demo", pd]
        if arm_kpts is not None:
            seg_cmd += ["--arm_kpts", str(arm_kpts)]
        _run(seg_cmd)

    # Stage 3.5: modal object seg BEFORE inpaint, so the object can be protected
    # from removal. (run_cube_segmentation later reuses cube_mask_raw and only
    # runs the Diffusion-VAS amodal pass.) Only runs when the cube layer is enabled.
    cube_raw = pd / "cube_layer" / "cube_mask_raw.npy"
    if args.cube_layer:
        if cube_raw.exists():
            print(f"\n[skip] {cube_raw} exists")
        else:
            _run([sys.executable, str(HERE / "segment_cube.py"), "--processed_demo", pd])

    # Stage 4: legacy E2FGVI on hand mask → inpainted bg (object protected)
    inp_bg = pd / "inpaint_processor" / "video_human_inpaint.mkv"
    if inp_bg.exists() and inp_bg.stat().st_size > 0:
        print(f"\n[skip] {inp_bg} exists")
    else:
        inpaint_cmd = [sys.executable, str(HERE / "inpaint_hands.py"),
                       "--processed_demo", pd, "--mode", "legacy"]
        if args.cube_layer:
            inpaint_cmd += ["--protect_mask", str(cube_raw)]
        _run(inpaint_cmd)

    # Stage 5: robot RGBD. Default = Isaac (RB5-850 arm + xhand hand) via the
    # cross-env wrapper; pyrender (xhand + RBY1 arm) is the legacy fallback.
    robot_mask_npy = pd / "overlay_processor" / "robot_mask.npy"
    if robot_mask_npy.exists():
        print(f"\n[skip] {robot_mask_npy} exists")
    elif args.render_backend == "isaac":
        _run(["bash", str(HERE / "isaac_stage5.sh"),
              str(pd), str(args.hawor_npz),
              str(args.right_pkl), str(args.left_pkl), args.hand])
    else:
        _run([sys.executable, "-u", str(HERE / "render_xhand_overlay_depth.py"),
              "--processed_demo", pd,
              "--hawor_npz", args.hawor_npz,
              "--right_pkl", args.right_pkl,
              "--left_pkl",  args.left_pkl,
              "--hand", args.hand])

    # Stages 6-9 only exist to build the object/cube layer (depth drives the
    # cube's contact shadow + front/behind ordering; 8-9 are the amodal object
    # mask). Skipped entirely unless --cube_layer is set.
    if args.cube_layer:
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

        # Stages 8-9: cube segmentation (SAM2 modal → Diffusion-VAS amodal)
        cube_amodal = pd / "cube_layer" / "cube_mask_amodal.npy"
        if cube_amodal.exists():
            print(f"\n[skip] {cube_amodal} exists")
        else:
            cube_cmd = [sys.executable, str(HERE / "run_cube_segmentation.py"),
                        "--processed_demo", pd,
                        "--cube_quantile", str(args.cube_quantile),
                        "--overlap", str(args.cube_overlap)]
            if args.no_content_completion:
                cube_cmd.append("--no_content_completion")
            _run(cube_cmd)

    # Stage 10: final layered composite
    final_mp4 = pd / "overlay_processor_layered" / "video_overlay.mp4"
    if final_mp4.exists():
        print(f"\n[skip] {final_mp4} exists")
    else:
        composite_cmd = [sys.executable, str(HERE / "composite_layered.py"),
                         "--processed_demo", pd,
                         "--hawor_npz", args.hawor_npz,
                         "--cube_mask_npy", "cube_layer/cube_mask_amodal.npy",
                         "--threshold_joint", str(args.threshold_joint),
                         "--zmcp_sigma_t", str(args.zmcp_sigma_t),
                         "--edge_sigma", str(args.edge_sigma)]
        if not args.cube_layer:
            composite_cmd.append("--no_cube")
        _run(composite_cmd)

    print(f"\n[done] final layered overlay: {final_mp4}")


if __name__ == "__main__":
    main()
