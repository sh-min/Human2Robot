"""Preprocess each recording into a single bundled `features.pt`.

For every recording dir matching `--recording_glob` under `--data_root`:
    1. Run V-JEPA over `rgb/`        → vjepa_orig
    2. Run V-JEPA over `hand_cube_mask_overlayed/` (if present) → vjepa_orig_masked
    3. Read `result.json` and convert MANO rot mats to axis-angle → mano [T, 96]
    4. Read `gt_labels.json` (if present) → labels_per_token [T]

Each recording dir must contain:
    rgb/                        (required)
    result.json                 (required)
    hand_cube_mask_overlayed/   (optional)
    gt_labels.json              (optional)

Output:
    {recording_dir}/features.pt
        vjepa_orig:        [T, 1024]
        vjepa_orig_masked: [T, 1024]   (optional)
        mano:              [T, 96]     downsampled to token rate
        labels_per_token:  [T]         int (-1 if no GT)
        num_frames, num_tokens, recording

Usage:
    python -m skill_segmentor.preprocess \
        --data_root data/cube_dataset/0412_train \
        --recording_glob "saved_frames_*" \
        --checkpoint ckpt/v-jepa2/vitl.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.labels import ACTION_LABELS
from data_preprocess.feature_extractor import VJEPAFeatureExtractor, load_pretrained_encoder
from utils.utils import rotmat_to_axis_angle

TUBELET = 2  # frames per token

VJEPA_MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255.0
VJEPA_STD = torch.tensor([0.229, 0.224, 0.225]) * 255.0


# ---------- V-JEPA ----------

def extract_vjepa(source, feat_extractor, device, crop_size, num_frames, batch_size):
    """Run V-JEPA over a frame directory or video, return [T, D]."""
    source = Path(source)
    frames_np = []
    if source.is_dir():
        frame_paths = sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((crop_size, crop_size), Image.BILINEAR)
            frames_np.append(np.array(img))
    elif source.is_file():
        cap = cv2.VideoCapture(str(source))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (crop_size, crop_size),
                             interpolation=cv2.INTER_LINEAR)
            frames_np.append(rgb)
        cap.release()

    nf = len(frames_np)
    if nf == 0:
        return None
    num_tokens = nf // TUBELET

    # Load + normalize
    frames = torch.from_numpy(np.stack(frames_np)).float().permute(0, 3, 1, 2)
    frames = (frames - VJEPA_MEAN[None, :, None, None]) / VJEPA_STD[None, :, None, None]

    # Pad to multiple of clip_len
    pad_to = ((nf + num_frames - 1) // num_frames) * num_frames
    if pad_to > nf:
        pad = frames[-1:].expand(pad_to - nf, -1, -1, -1)
        frames = torch.cat([frames, pad], dim=0)

    num_clips = pad_to // num_frames
    clips = frames.view(num_clips, num_frames, 3, crop_size, crop_size).permute(0, 2, 1, 3, 4)

    all_tokens = []
    for i in range(0, num_clips, batch_size):
        batch = clips[i:i + batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = feat_extractor(batch)
        all_tokens.append(out.cpu().float())
    all_tokens = torch.cat(all_tokens, dim=0)  # [num_clips, N, D]

    tokens_per_clip = num_frames // TUBELET
    num_spatial = (crop_size // 16) ** 2
    embed_dim = all_tokens.shape[-1]
    all_tokens = all_tokens.view(num_clips, tokens_per_clip, num_spatial, embed_dim)
    all_tokens = all_tokens.mean(dim=2)              # spatial mean → [num_clips, T_tok, D]
    all_tokens = all_tokens.view(-1, embed_dim)[:num_tokens]  # [T, D]
    return all_tokens, nf


# ---------- MANO ----------

def extract_mano(rec_dir, num_frames):
    """Read result.json and return [num_frames, 96] axis-angle features."""
    json_path = rec_dir / "result.json"
    if not json_path.exists():
        # Current HaWoR pipeline writes axis-angle parameters directly.
        npz_path = rec_dir / "rgb_hawor" / "retarget_input.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path)
        n = min(num_frames, data["mano_global_orient"].shape[1])
        feat = np.zeros((num_frames, 2, 48), dtype=np.float32)
        for side in range(2):
            feat[:n, side, :3] = data["mano_global_orient"][side, :n]
            feat[:n, side, 3:] = data["mano_hand_pose"][side, :n].reshape(n, 45)
            valid = data["valid"][side, :n].astype(bool)
            feat[:n, side][~valid] = 0
        return feat.reshape(num_frames, -1)
    rgb_dir = rec_dir / "rgb"
    frame_names = sorted(
        f.stem for f in rgb_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    feat = np.zeros((num_frames, 2, 48), dtype=np.float32)
    with open(json_path) as f:
        data = json.load(f)
    for fi, fn in enumerate(frame_names):
        if fn not in data:
            continue
        for hand in data[fn]:
            side = int(hand["is_right"])
            mp = hand.get("mano_params", {})
            if "hand_pose" not in mp:
                continue
            go = rotmat_to_axis_angle(mp["global_orient"][0])
            hp = np.concatenate([rotmat_to_axis_angle(mp["hand_pose"][j]) for j in range(15)])
            feat[fi, side] = np.concatenate([go, hp])
    return feat.reshape(num_frames, -1)  # [F, 96]


def downsample_to_tokens(frames_arr, num_tokens):
    """Average pairs of frames → [num_tokens, D]."""
    out = []
    for t in range(num_tokens):
        f0 = t * TUBELET
        f1 = min(f0 + 1, frames_arr.shape[0] - 1)
        out.append((frames_arr[f0] + frames_arr[f1]) / 2.0)
    return torch.from_numpy(np.stack(out))


# ---------- Labels ----------

def labels_per_token(rec_dir, num_tokens):
    gt_path = rec_dir / "gt_labels.json"
    if not gt_path.exists():
        return torch.full((num_tokens,), -1, dtype=torch.int32)
    gt = json.load(open(gt_path))
    arr = np.full(num_tokens, -1, dtype=np.int32)
    for seg in gt.get("segments", []):
        if seg["label"] not in ACTION_LABELS:
            continue
        idx = ACTION_LABELS.index(seg["label"])
        s_tok = max(0, int(seg["start_frame"]) // TUBELET)
        e_tok = min(num_tokens - 1, int(seg["end_frame"]) // TUBELET)
        arr[s_tok:e_tok + 1] = idx
    return torch.from_numpy(arr)


# ---------- Driver ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--recording_glob", type=str, required=True,
                        help="e.g. 'saved_frames_*' or 'episode_*'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="V-JEPA checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per V-JEPA clip")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if features.pt already exists")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading V-JEPA encoder ...")
    encoder = load_pretrained_encoder(
        checkpoint_path=args.checkpoint, device=device,
        model_name="vit_large", crop_size=args.crop_size,
        patch_size=16, num_frames=args.num_frames, tubelet_size=TUBELET,
    )
    feat_extractor = VJEPAFeatureExtractor(encoder, pool="none").to(device)

    data_root = Path(args.data_root)
    patterns = [p.strip() for p in args.recording_glob.split(",") if p.strip()]
    recordings = sorted({
        d
        for pattern in patterns
        for d in data_root.glob(pattern)
        if d.is_dir()
    })
    print(f"Found {len(recordings)} recordings under {data_root}")

    for ri, rec in enumerate(recordings):
        out_path = rec / "features.pt"
        if out_path.exists() and not args.overwrite:
            print(f"  [{ri+1}/{len(recordings)}] {rec.name}: skip (already exists)")
            continue
        rgb_dir = rec / "rgb"
        if not rgb_dir.is_dir():
            print(f"  [{ri+1}/{len(recordings)}] {rec.name}: SKIP (no rgb/)")
            continue

        print(f"  [{ri+1}/{len(recordings)}] {rec.name}")

        # 1. V-JEPA on rgb/
        result = extract_vjepa(rgb_dir, feat_extractor, device,
                               args.crop_size, args.num_frames, args.batch_size)
        if result is None:
            print(f"    SKIP: empty rgb/")
            continue
        vjepa_orig, num_frames = result
        num_tokens = vjepa_orig.shape[0]
        bundle = {
            "vjepa_orig": vjepa_orig,
            "num_frames": num_frames,
            "num_tokens": num_tokens,
            "recording": rec.name,
        }

        # 2. V-JEPA on masked frames (optional)
        masked_dir = rec / "hand_cube_mask_overlayed"
        if masked_dir.is_dir():
            res2 = extract_vjepa(masked_dir, feat_extractor, device,
                                 args.crop_size, args.num_frames, args.batch_size)
            if res2 is not None:
                vjepa_masked, _ = res2
                bundle["vjepa_orig_masked"] = vjepa_masked

        # 2b. V-JEPA on the completed robot-hand replacement video.
        robot_root = rec / "inpainting_processed" / rec.name / "0"
        robot_candidates = [
            robot_root / "video_overlay_rby1_xhand.mp4",
            robot_root / "video_overlay_xhand.mp4",
        ]
        robot_video = next((p for p in robot_candidates if p.is_file()), None)
        if robot_video is not None:
            res_robot = extract_vjepa(
                robot_video, feat_extractor, device,
                args.crop_size, args.num_frames, args.batch_size,
            )
            if res_robot is not None:
                vjepa_robot, _ = res_robot
                bundle["vjepa_robot"] = vjepa_robot

        # 3. MANO from result.json (frame-rate → token-rate)
        mano_frames = extract_mano(rec, num_frames)
        if mano_frames is None:
            print(f"    WARN: no result.json")
        else:
            bundle["mano"] = downsample_to_tokens(mano_frames, num_tokens)

        # 4. Per-token labels from gt_labels.json
        bundle["labels_per_token"] = labels_per_token(rec, num_tokens)

        torch.save(bundle, out_path)
        keys = [k for k, v in bundle.items() if hasattr(v, "shape")]
        print(f"    saved {out_path.name}  T={num_tokens}  feats={keys}")

    print("Done.")


if __name__ == "__main__":
    main()
