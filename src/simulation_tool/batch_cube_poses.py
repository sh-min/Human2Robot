"""Run per-skill cube-pose optimization for every non-TRANS segment in a
sequence, using its first frame.

Reads <episode>/predictions.pt (skill-classifier output) and dispatches each
segment to optimize_cube_pose with the right holding hand + target +Z axis.

Skill → holding hand:
    R / U / D / B  →  left hand holds
    F / L          →  right hand holds

Skill → cube +Z target axis (cam frame, sign-free):
    R / L  →  cam +x
    U / D  →  cam +y
    F / B  →  cam +z

Usage:
    conda activate RFM_retarget
    cd <repo_root>/src/simulation_tool
    python batch_cube_poses.py \
        --episode /path/to/<episode> \
        [--predictions predictions.pt]  [--mask_subdir rgb_mask]
        [--rgb_subdir rgb]  [--hawor_subdir rgb_hawor]
"""
import argparse
import os
import subprocess
import sys

import torch

LETTER_TO_HAND = {"R": "left", "U": "left", "D": "left", "B": "left",
                  "F": "right", "L": "right"}
LETTER_TO_AXIS = {"R": "x", "L": "x",
                  "U": "y", "D": "y",
                  "F": "z", "B": "z"}


def parse_skill(name):
    """'RCW'/'RCCW' → ('R', 'CW'/'CCW').  TRANS returns None."""
    if name == "TRANS":
        return None
    if name.endswith("CCW"):
        return name[:-3], "CCW"
    if name.endswith("CW"):
        return name[:-2], "CW"
    raise ValueError(f"unrecognised skill: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True,
                    help="Episode root (contains predictions.pt, rgb/, rgb_mask/, rgb_hawor/)")
    ap.add_argument("--predictions", default="predictions.pt")
    ap.add_argument("--mask_subdir", default="rgb_mask")
    ap.add_argument("--rgb_subdir", default="rgb")
    ap.add_argument("--hawor_subdir", default="rgb_hawor")
    ap.add_argument("--mask_pattern", default="rgb_frame{:05d}.png")
    ap.add_argument("--cube_size", type=float, default=0.05)
    ap.add_argument("--img_focal", type=float, default=497.77)
    ap.add_argument("--axis_weight", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--silhouette_mode", choices=["line", "dt"], default="dt")
    ap.add_argument("--out_subdir", default="cube_poses",
                    help="Output dir under <hawor_subdir>/")
    args = ap.parse_args()

    ep = os.path.abspath(args.episode)
    pred_path = os.path.join(ep, args.predictions)
    hawor_dir = os.path.join(ep, args.hawor_subdir)
    npz_path = os.path.join(hawor_dir, "retarget_input.npz")
    mask_dir = os.path.join(ep, args.mask_subdir)
    rgb_dir = os.path.join(ep, args.rgb_subdir)
    out_dir = os.path.join(hawor_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    for p in (pred_path, npz_path, mask_dir, rgb_dir):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    d = torch.load(pred_path, map_location="cpu", weights_only=False)
    labels = d["labels"]
    segments = d["segments"]
    print(f"loaded {len(segments)} segments  labels={labels}")

    here = os.path.dirname(os.path.abspath(__file__))
    optim_script = os.path.join(here, "optimize_cube_pose.py")
    py = sys.executable

    n_ok = n_skip = 0
    for k, (s, e, label_idx) in enumerate(segments):
        name = labels[int(label_idx)]
        parsed = parse_skill(name)
        if parsed is None:
            n_skip += 1
            continue
        letter, direction = parsed
        hand = LETTER_TO_HAND[letter]
        axis = LETTER_TO_AXIS[letter]
        first_frame = int(s)
        mask_path = os.path.join(mask_dir, args.mask_pattern.format(first_frame))
        rgb_path = os.path.join(rgb_dir, args.mask_pattern.format(first_frame))
        if not os.path.exists(mask_path):
            print(f"[{k:02d} {name}] frame {first_frame}: mask missing → skip")
            n_skip += 1
            continue

        tag = f"seg{k:02d}_f{first_frame:05d}_{name}"
        out_pose = os.path.join(out_dir, f"{tag}.npz")
        out_viz = os.path.join(out_dir, f"{tag}_viz.png")
        print(f"\n=== {tag}  hand={hand}  axis={axis} ===")

        cmd = [
            py, optim_script,
            "--npz", npz_path,
            "--mask", mask_path,
            "--rgb", rgb_path,
            "--hand", hand,
            "--frame", str(first_frame),
            "--cube_size", str(args.cube_size),
            "--img_focal", str(args.img_focal),
            "--target_axis", axis,
            "--axis_weight", str(args.axis_weight),
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--silhouette_mode", args.silhouette_mode,
            "--out_pose", out_pose,
            "--out_viz", out_viz,
        ]
        try:
            r = subprocess.run(cmd, check=True, capture_output=True, text=True)
            for line in r.stdout.strip().splitlines()[-3:]:
                print("  " + line)
            n_ok += 1
        except subprocess.CalledProcessError as ex:
            print(f"  FAILED:\n{ex.stderr.strip().splitlines()[-3:]}")
            n_skip += 1

    print(f"\nDone. ok={n_ok}  skipped={n_skip}  outputs in {out_dir}")


if __name__ == "__main__":
    main()
