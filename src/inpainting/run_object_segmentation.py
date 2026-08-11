"""Object segmentation orchestrator — stage 8 of the layered pipeline.

    8. SAM2 + Depth + Diffusion-VAS → object_mask_amodal.npy (amodal silhouette)
         segment_object.py   (SAM2, inpaint env)     → object_mask_raw.npy  (modal)
         amodal_object.py    (VAS, diffusion_vas env) → object_mask_amodal.npy (amodal)

Stage 8 is a fixed pipeline: SAM2 modal mask → Diffusion-VAS amodal
completion (which internally uses Depth Anything V2). The amodal mask
is post-processed in amodal_object.py (top-percentile threshold, bbox
clipping, morph ops, largest-CC, SDF temporal smoothing).

The object layer in the final composite uses the inpainted background
(from E2FGVI) at object_mask pixels — no separate content completion needed.

8's sub-steps run in different conda envs (inpaint vs diffusion_vas).

Usage:
    python run_object_segmentation.py --processed_demo /result/.../cam0/0
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
    ap.add_argument("--diffusion_vas_python", default=sys.executable,
                    help="Python interpreter for the Diffusion-VAS amodal stage. "
                         "Defaults to the current interpreter (the merged uv env "
                         "now has diffusers, so no separate conda env is needed).")
    # SAM2
    ap.add_argument("--object_quantile", type=float, default=0.25,
                    help="non-hand depth quantile for the SAM2 seed bootstrap")
    # Diffusion-VAS
    ap.add_argument("--overlap", type=int, default=0,
                    help="frames shared between Diffusion-VAS 25-frame windows")
    ap.add_argument("--top_percentile", type=float, default=1.0,
                    help="per-frame top-N%% threshold for amodal mask")
    ap.add_argument("--smooth_sigma", type=float, default=2.0,
                    help="SDF temporal smoothing sigma in frames")
    ap.add_argument("--bbox_margin", type=int, default=25,
                    help="px margin around modal bbox for noise clipping")
    args = ap.parse_args()

    pd = args.processed_demo
    object_dir = pd / "object_layer"

    # ── Stage 8: SAM2 + Depth + Diffusion-VAS ───────────────────────────
    if (object_dir / "object_mask_amodal.npy").exists():
        print(f"\n[skip] stage 8 — {object_dir / 'object_mask_amodal.npy'} exists")
    else:
        # SAM2 modal object mask (inpaint env)
        if not (object_dir / "object_mask_raw.npy").exists():
            _run([sys.executable, str(HERE / "segment_object.py"),
                  "--processed_demo", pd,
                  "--quantile", str(args.object_quantile)])

        # Diffusion-VAS amodal segmentation (same interpreter by default)
        _run([args.diffusion_vas_python, str(HERE / "amodal_object.py"),
              "--processed_demo", str(pd),
              "--overlap", str(args.overlap),
              "--top_percentile", str(args.top_percentile),
              "--smooth_sigma", str(args.smooth_sigma),
              "--bbox_margin", str(args.bbox_margin)])

    print(f"\n[done] object segmentation → {object_dir / 'object_mask_amodal.npy'}")


if __name__ == "__main__":
    main()
