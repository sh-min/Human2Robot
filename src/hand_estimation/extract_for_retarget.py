"""HaWoR -> npz extractor for retargeting.

Wraps the HaWoR repo (vendored at <repo_root>/third_party/HaWoR) so we don't
have to cd into it. Outputs a single .npz with MANO joints + parameters per
frame, for both hands.

Usage (conda activate hawor):
    cd <repo_root>/src/hand_estimation
    python extract_for_retarget.py \
        --rgb_dir   /path/to/rgb_frames \
        --img_glob  "rgb_frame*.png" \
        --img_focal 497.77 \
        --skip_slam

Output:
    <rgb_dir parent>/<rgb_dir basename>_hawor/retarget_input.npz
"""
import argparse
import json
import os
import subprocess
import sys
from glob import glob

# Make the vendored HaWoR importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAWOR_DIR = os.path.join(REPO_ROOT, "third_party", "HaWoR")
if not os.path.isdir(HAWOR_DIR):
    raise RuntimeError(f"HaWoR not found at {HAWOR_DIR}")
sys.path.insert(0, HAWOR_DIR)
# HaWoR also imports things relative to its own root (e.g. weights/, _DATA/),
# so we run with cwd = HaWoR_DIR.
os.chdir(HAWOR_DIR)

import cv2
import joblib
import numpy as np
import torch
from natsort import natsorted

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import (
    hawor_motion_estimation,
    hawor_infiller,
)
from scripts.scripts_test_video.hawor_slam import hawor_slam
from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.eval_utils.custom_utils import load_slam_cam


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def prepare_jpg_frames(rgb_dir, dst_dir, glob_pat="rgb_frame*.png", quality=95):
    pngs = natsorted(glob(os.path.join(rgb_dir, glob_pat)))
    if not pngs:
        raise RuntimeError(f"No images matched {glob_pat!r} in {rgb_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(dst_dir) if f.endswith(".jpg"))
    if len(existing) == len(pngs) and existing:
        print(f"[skip] jpg frames already prepared ({len(existing)})")
        return
    print(f"[convert] {len(pngs)} png -> jpg (quality={quality}) into {dst_dir}")
    for i, p in enumerate(pngs):
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        cv2.imwrite(
            os.path.join(dst_dir, f"{i:04d}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_dir", type=str, default=None,
                   help="Directory of RGB image sequence.")
    p.add_argument("--img_glob", type=str, default="rgb_frame*.png",
                   help="Glob pattern selecting RGB frames inside --rgb_dir. "
                        "Default skips depth/aux files.")
    p.add_argument("--jpg_quality", type=int, default=95)
    p.add_argument("--video_path", type=str, default=None,
                   help="Used directly when --rgb_dir is not given. Should be "
                        "<workdir>.mp4 even if the file does not exist; the "
                        "stem is used to derive seq_folder.")
    p.add_argument("--workdir", type=str, default=None,
                   help="Override for the output / cache directory. Default: "
                        "<rgb_dir parent>/<rgb_dir basename>_hawor")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--img_focal", type=float, default=None)
    p.add_argument("--input_type", type=str, default="file")
    p.add_argument("--checkpoint", type=str,
                   default="./weights/hawor/checkpoints/hawor.ckpt")
    p.add_argument("--infiller_weight", type=str,
                   default="./weights/hawor/checkpoints/infiller.pt")
    p.add_argument("--vis_mode", type=str, default="world")
    p.add_argument("--skip_slam", action="store_true",
                   help="Skip SLAM + infiller; output stays in cam frame and "
                        "frames without a hand detection are left invalid.")
    p.add_argument("--out_npz", type=str, default=None,
                   help="Defaults to <seq_folder>/retarget_input.npz")
    args = p.parse_args()

    if args.rgb_dir is not None:
        rgb_dir = os.path.abspath(args.rgb_dir.rstrip("/"))
        seq_name = os.path.basename(rgb_dir)
        parent = os.path.dirname(rgb_dir)
        workdir = args.workdir or os.path.join(parent, f"{seq_name}_hawor")
        os.makedirs(workdir, exist_ok=True)
        prepare_jpg_frames(
            rgb_dir,
            os.path.join(workdir, "extracted_images"),
            glob_pat=args.img_glob,
            quality=args.jpg_quality,
        )
        # Set video_path so detect_track_video derives seq_folder == workdir.
        args.video_path = os.path.join(
            os.path.dirname(workdir), f"{os.path.basename(workdir)}.mp4"
        )
    elif args.video_path is None:
        raise ValueError("Provide --rgb_dir or --video_path")

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(
        args, start_idx, end_idx, seq_folder
    )

    if args.skip_slam:
        T_total = len(imgfiles)
        pred_trans = torch.zeros(2, T_total, 3)
        pred_rot = torch.zeros(2, T_total, 3)
        pred_hand_pose = torch.zeros(2, T_total, 15, 3)
        pred_betas = torch.zeros(2, T_total, 10)
        pred_valid = torch.zeros(2, T_total)

        for idx in [0, 1]:
            for frame_ck in frame_chunks_all.get(idx, []):
                pred_path = os.path.join(
                    seq_folder, "cam_space", str(idx),
                    f"{frame_ck[0]}_{frame_ck[-1]}.json",
                )
                if not os.path.exists(pred_path):
                    continue
                with open(pred_path) as f:
                    d = json.load(f)
                init_trans = torch.tensor(d["init_trans"])
                init_root_R = torch.tensor(d["init_root_orient"])
                init_hand_R = torch.tensor(d["init_hand_pose"])
                init_betas = torch.tensor(d["init_betas"])
                init_root = rotation_matrix_to_angle_axis(init_root_R)
                init_hand = rotation_matrix_to_angle_axis(init_hand_R)

                pred_trans[idx, frame_ck] = init_trans[0]
                pred_rot[idx, frame_ck] = init_root[0]
                pred_hand_pose[idx, frame_ck] = init_hand[0]
                pred_betas[idx, frame_ck] = init_betas[0]
                pred_valid[idx, frame_ck] = 1
        pred_valid = pred_valid > 0
        R_c2w = t_c2w = None
    else:
        slam_path = os.path.join(
            seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
        )
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        _, _, R_c2w, t_c2w = load_slam_cam(slam_path)

        infill_cache = os.path.join(seq_folder, "world_space_res.pth")
        if os.path.exists(infill_cache):
            print(f"[skip] infiller, loading cached {infill_cache}")
            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = (
                joblib.load(infill_cache)
            )
        else:
            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
                args, start_idx, end_idx, frame_chunks_all
            )
        pred_hand_pose = pred_hand_pose.reshape(*pred_hand_pose.shape[:-1], 15, 3)

    # MANO forward to get 21 joint positions per hand.
    out_l = run_mano_left(
        pred_trans[0:1], pred_rot[0:1], pred_hand_pose[0:1], betas=pred_betas[0:1]
    )
    out_r = run_mano(
        pred_trans[1:2], pred_rot[1:2], pred_hand_pose[1:2], betas=pred_betas[1:2]
    )
    joints_left = _to_np(out_l["joints"])[0]
    joints_right = _to_np(out_r["joints"])[0]

    out_npz = args.out_npz or os.path.join(seq_folder, "retarget_input.npz")
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    save_kwargs = dict(
        joints_left=joints_left,
        joints_right=joints_right,
        mano_trans=_to_np(pred_trans),
        mano_global_orient=_to_np(pred_rot),
        mano_hand_pose=_to_np(pred_hand_pose),
        mano_betas=_to_np(pred_betas),
        valid=_to_np(pred_valid),
        img_focal=np.float32(img_focal),
        start_idx=np.int64(start_idx),
        end_idx=np.int64(end_idx),
        frame_is_cam_space=np.bool_(args.skip_slam),
    )
    if R_c2w is not None:
        save_kwargs["R_c2w"] = _to_np(R_c2w)
        save_kwargs["t_c2w"] = _to_np(t_c2w)
    np.savez(out_npz, **save_kwargs)

    print(f"[done] saved {out_npz}")
    print(f"  joints_left  {joints_left.shape}")
    print(f"  joints_right {joints_right.shape}")
    print(f"  valid        {_to_np(pred_valid).shape}  (sum L/R: "
          f"{int(_to_np(pred_valid)[0].sum())}/{int(_to_np(pred_valid)[1].sum())})")


if __name__ == "__main__":
    main()
