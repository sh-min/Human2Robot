"""Inject HaWoR keypoints + derived bboxes into a phantom demo folder.

Phantom's pipeline normally runs HaMeR for 3D hand pose and Epic-Kitchens for
2D hand bboxes. Both fail on cam0_hawor-style top-down views, so the SAM2
arm_segmentation gets no prompt and E2FGVI then has nothing to inpaint.

This script bypasses both detectors by writing:
    - hand_processor/hand_data_{left,right}.npz  (3D joints + projected 2D)
    - bbox_processor/bbox_data.npz                (bboxes derived from 2D kpts)
    - video_rgb_imgs.mkv                          (re-encoded ffv1, required by E2FGVI)

Inputs:
    retarget_input.npz  (from src/hand_estimation/extract_for_retarget.py)
        joints_left/right:    (T, 21, 3) MANO cam-frame joints
        valid:                (2, T)     left/right validity
        img_focal:            scalar

Run AFTER `prepare_demo.py` and BEFORE phantom mode=[arm_segmentation,hand_inpaint].

Usage:
    python inject_hawor_data.py \
        --processed_demo /result/cam0_inpaint/cam0/0 \
        --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz
"""
import argparse
from pathlib import Path

import cv2
import mediapy as media
import numpy as np

BBOX_PAD_RATIO = 0.4  # 40% padding around hand keypoints to cover forearm


def _project_3d_to_2d(joints_3d: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    Z = np.clip(joints_3d[..., 2], 1e-6, None)
    u = focal * joints_3d[..., 0] / Z + cx
    v = focal * joints_3d[..., 1] / Z + cy
    return np.stack([u, v], axis=-1)


def _bbox_from_kpts(kpts_2d: np.ndarray, img_w: int, img_h: int, pad: float):
    x1 = kpts_2d[..., 0].min(axis=-1)
    y1 = kpts_2d[..., 1].min(axis=-1)
    x2 = kpts_2d[..., 0].max(axis=-1)
    y2 = kpts_2d[..., 1].max(axis=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = np.maximum(x2 - x1, y2 - y1) * (1 + pad)
    x1 = np.clip(cx - side / 2, 0, img_w - 1)
    y1 = np.clip(cy - side / 2, 0, img_h - 1)
    x2 = np.clip(cx + side / 2, 0, img_w - 1)
    y2 = np.clip(cy + side / 2, 0, img_h - 1)
    bboxes = np.stack([x1, y1, x2, y2], axis=-1)
    ctrs = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], axis=-1)
    min_dist = np.stack([x1, y1, img_w - 1 - x2, img_h - 1 - y2], axis=-1).min(axis=-1)
    return bboxes, ctrs, min_dist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    args = ap.parse_args()

    # Discover video dimensions
    video_path = args.processed_demo / "video_L.mp4"
    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"[info] video {img_w}x{img_h}, T_video={n_frames}")

    # Load HaWoR data and truncate to video length
    ri = np.load(args.hawor_npz)
    T = min(n_frames, ri["joints_left"].shape[0])
    joints_l = ri["joints_left"][:T].astype(np.float64)
    joints_r = ri["joints_right"][:T].astype(np.float64)
    valid_l = ri["valid"][0, :T].astype(bool)
    valid_r = ri["valid"][1, :T].astype(bool)
    focal = float(ri["img_focal"])
    cx, cy = img_w / 2.0, img_h / 2.0
    print(f"[info] T_use={T}, focal={focal:.2f}, valid L/R={valid_l.sum()}/{valid_r.sum()}")

    kpts_2d_l = _project_3d_to_2d(joints_l, focal, cx, cy)
    kpts_2d_r = _project_3d_to_2d(joints_r, focal, cx, cy)

    # Write hand_processor/hand_data_*.npz
    hp = args.processed_demo / "hand_processor"
    hp.mkdir(parents=True, exist_ok=True)
    frame_idx = np.arange(T, dtype=np.int64)
    np.savez_compressed(hp / "hand_data_left.npz",
                        frame_indices=frame_idx, hand_detected=valid_l,
                        kpts_2d=kpts_2d_l, kpts_3d=joints_l)
    np.savez_compressed(hp / "hand_data_right.npz",
                        frame_indices=frame_idx, hand_detected=valid_r,
                        kpts_2d=kpts_2d_r, kpts_3d=joints_r)
    print(f"[ok] wrote {hp}/hand_data_{{left,right}}.npz")

    # Write bbox_processor/bbox_data.npz (overwrites any failed bbox stage output)
    bboxes_l, ctrs_l, mind_l = _bbox_from_kpts(kpts_2d_l, img_w, img_h, BBOX_PAD_RATIO)
    bboxes_r, ctrs_r, mind_r = _bbox_from_kpts(kpts_2d_r, img_w, img_h, BBOX_PAD_RATIO)
    bboxes_l[~valid_l] = 0; ctrs_l[~valid_l] = 0; mind_l[~valid_l] = 0
    bboxes_r[~valid_r] = 0; ctrs_r[~valid_r] = 0; mind_r[~valid_r] = 0
    bp = args.processed_demo / "bbox_processor"
    bp.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bp / "bbox_data.npz",
                        left_hand_detected=valid_l,  right_hand_detected=valid_r,
                        left_bboxes=bboxes_l,         right_bboxes=bboxes_r,
                        left_bboxes_ctr=ctrs_l,       right_bboxes_ctr=ctrs_r,
                        left_bbox_min_dist_to_edge=mind_l,
                        right_bbox_min_dist_to_edge=mind_r)
    print(f"[ok] wrote {bp}/bbox_data.npz "
          f"(max mind-to-edge L/R={mind_l.max():.0f}/{mind_r.max():.0f})")

    # E2FGVI's hand_inpaint stage reads video_rgb_imgs.mkv (ffv1), not video_L.mp4
    dst = args.processed_demo / "video_rgb_imgs.mkv"
    if dst.exists():
        print(f"[skip] {dst} exists")
    else:
        frames = media.read_video(str(video_path))
        media.write_video(str(dst), frames, fps=10, codec="ffv1")
        print(f"[ok] wrote {dst}")


if __name__ == "__main__":
    main()
