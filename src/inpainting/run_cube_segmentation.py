"""Cube segmentation orchestrator — stages 8 + 9 of the layered pipeline.

Cube segmentation is three sub-steps that produce the cube layer mask:

    8a. segment_cube.py            SAM2 (depth-seeded)  → cube_mask_raw.npy   (modal)
    8b. amodal_cube.py             Diffusion-VAS        → cube_mask_amodal.npy (amodal)
    9.  regularize_and_cut_cube.py open/CC/close/SDF    → cube_mask_clean.npy

8a and 9 run in the `inpaint` conda env; 8b runs in the `diffusion_vas`
env (heavy SVD deps), invoked via `conda run`. Each sub-step skips itself
if its output already exists — delete the file to recompute.

Because Diffusion-VAS already produces a true amodal silhouette, stage 9
runs with --no_hull (the convex-hull amodal hack is no longer needed).

Usage:
    python run_cube_segmentation.py --processed_demo /result/.../cam0/0
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def _run(cmd) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--diffusion_vas_env", default="diffusion_vas",
                    help="conda env name for the Diffusion-VAS amodal stage")
    # 8a segment_cube
    ap.add_argument("--cube_quantile", type=float, default=0.25,
                    help="non-hand depth quantile for the SAM2 seed bootstrap")
    # 8b amodal_cube
    ap.add_argument("--overlap", type=int, default=0,
                    help="frames shared between Diffusion-VAS 25-frame windows")
    ap.add_argument("--no_content_completion", action="store_true",
                    help="skip the amodal-RGB pass (amodal mask only)")
    # 9 regularize
    ap.add_argument("--sdf_sigma", type=float, default=8.0)
    ap.add_argument("--area_mad_k", type=float, default=5.0)
    ap.add_argument("--centroid_max_jump", type=float, default=120.0)
    args = ap.parse_args()

    pd = args.processed_demo
    cube = pd / "cube_layer"

    # 8a — SAM2 modal cube mask
    if (cube / "cube_mask_raw.npy").exists():
        print(f"\n[skip] {cube / 'cube_mask_raw.npy'} exists")
    else:
        _run([sys.executable, str(HERE / "segment_cube.py"),
              "--processed_demo", pd, "--quantile", str(args.cube_quantile)])

    # 8b — Diffusion-VAS amodal completion (runs in the diffusion_vas env)
    if (cube / "cube_mask_amodal.npy").exists():
        print(f"\n[skip] {cube / 'cube_mask_amodal.npy'} exists")
    else:
        amodal_cmd = ["conda", "run", "-n", args.diffusion_vas_env,
                      "--no-capture-output", "python", str(HERE / "amodal_cube.py"),
                      "--processed_demo", str(pd), "--overlap", str(args.overlap)]
        if args.no_content_completion:
            amodal_cmd.append("--no_content_completion")
        _run(amodal_cmd)

    # 9 — regularize the amodal mask (no convex hull — amodal is already real)
    if (cube / "cube_mask_clean.npy").exists():
        print(f"\n[skip] {cube / 'cube_mask_clean.npy'} exists")
    else:
        _run([sys.executable, str(HERE / "regularize_and_cut_cube.py"),
              "--processed_demo", pd,
              "--raw_mask", "cube_layer/cube_mask_amodal.npy",
              "--no_hull",
              "--sdf_sigma", str(args.sdf_sigma),
              "--area_mad_k", str(args.area_mad_k),
              "--centroid_max_jump", str(args.centroid_max_jump)])

    print(f"\n[done] cube segmentation → {cube / 'cube_mask_clean.npy'}")


if __name__ == "__main__":
    main()
