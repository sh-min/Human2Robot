"""
Inference on long-horizon multi-skill episodes.

Given a long video with multiple skills performed sequentially:
1. Extract V-JEPA features (EgoDex-pretrained)
2. Load WiLoR hand poses
3. Run trained skill classifier with sliding window
4. Output: per-frame skill predictions + visualization video

Usage (V-JEPA + hand):
    cd /virtual_lab/ljw_rvlab/byeonggyeol/3dgs-visual-grounding/skill2policy
    PYTHONPATH=$PWD/src python -m skill_classifier.infer_long_horizon \
        --data_dir data/kitchen_dataset/0325 \
        --vjepa_ckpt ckpt/egodex/vitl16-256px-16f/latest.pt \
        --classifier_ckpt output/skill_classifier/mlp_w8_0325_0138/best_mlp.pt

Usage (MANO hand-only):
    cd /virtual_lab/ljw_rvlab/byeonggyeol/3dgs-visual-grounding/skill2policy
    PYTHONPATH=$PWD/src python -m skill_classifier.infer_long_horizon \
        --data_dir data/kitchen_dataset/0412_val \
        --classifier_ckpt output/skill_classifier/mlp_w8_0416_1547/best_mlp.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.labels import ACTION_LABELS
from skill_classifier.models import build_model

# Module-level label list — overridden in main() from checkpoint when needed
LABELS = list(ACTION_LABELS)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _frame_path(rgb_dir, frame_name):
    root = Path(rgb_dir)
    for suffix in IMAGE_SUFFIXES:
        candidate = root / f"{frame_name}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no image for {frame_name} under {root}")


def discover_episodes(data_dir, mano=False, image_dir="rgb"):
    """Find all episode directories in data_dir."""
    data_dir = Path(data_dir)
    episodes = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        rgb_dir = d / image_dir
        if not rgb_dir.is_dir():
            continue
        frames = sorted(
            (
                path
                for path in rgb_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: path.stem,
        )
        if not frames:
            continue
        json_path = d / "result.json"
        hawor_path = d / "rgb_hawor" / "retarget_input.npz"
        feature_path = d / "features.pt"
        if not json_path.exists() and not hawor_path.exists() and not feature_path.exists():
            continue
        episodes.append({
            "name": d.name,
            "rgb_dir": str(rgb_dir),
            "ep_dir": str(d),
            "json_path": str(json_path) if json_path.exists() else None,
            "hawor_path": str(hawor_path) if hawor_path.exists() else None,
            "features_path": str(feature_path) if feature_path.exists() else None,
            "num_frames": len(frames),
            "frame_names": [f.stem for f in frames],
        })
    return episodes


def load_frames(rgb_dir, frame_names, crop_size):
    frames = []
    for fn in frame_names:
        from PIL import Image
        img = Image.open(_frame_path(rgb_dir, fn)).convert("RGB")
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        frames.append(np.array(img))
    return np.stack(frames)


def load_hand_kpts(json_path, frame_names):
    """Load hand keypoints. JSON keys are frame names like 'rgb_frame00000'."""
    nf = len(frame_names)
    kpts = np.zeros((nf, 2, 21, 3), dtype=np.float32)
    with open(json_path) as f:
        data = json.load(f)
    for fi, fn in enumerate(frame_names):
        if fn not in data:
            continue
        for hand in data[fn]:
            is_right = int(hand["is_right"])
            kpts[fi, is_right] = np.array(hand["kpts_3d"], dtype=np.float32)
    return kpts


def load_hand_pose_mano(json_path, frame_names):
    """Load global_orient + hand_pose + kpts_3d from result.json. Returns [F, 2, 207]."""
    nf = len(frame_names)
    mano = np.zeros((nf, 2, 207), dtype=np.float32)  # 9 + 135 + 63
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
            go = np.array(mp["global_orient"], dtype=np.float32).reshape(-1)  # 9
            hp = np.array(mp["hand_pose"],     dtype=np.float32).reshape(-1)  # 135
            kp = np.array(hand["kpts_3d"],     dtype=np.float32).reshape(-1)  # 63
            mano[fi, side] = np.concatenate([go, hp, kp])
    return mano


def load_mano_axis_angle(json_path, frame_names):
    """Load MANO params from result.json and convert to axis-angle.

    Returns [F, 96]: per hand 48 dims (global_orient 3 + hand_pose 45) x 2 hands.
    """
    from utils.utils import rotmat_to_axis_angle
    nf = len(frame_names)
    feat = np.zeros((nf, 2, 48), dtype=np.float32)
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
            go = rotmat_to_axis_angle(mp["global_orient"][0])  # (3,)
            hp = np.concatenate([rotmat_to_axis_angle(mp["hand_pose"][j]) for j in range(15)])  # (45,)
            feat[fi, side] = np.concatenate([go, hp])
    return torch.from_numpy(feat.reshape(nf, -1))  # [F, 96]


def extract_vjepa_features(encoder, feat_extractor, frames_np, mean, std, device,
                           clip_len=16, tubelet=2, batch_size=4):
    """Extract per-token V-JEPA features from frames."""
    t = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2)
    t = (t - mean[None, :, None, None]) / std[None, :, None, None]

    nf = t.shape[0]
    num_tokens = nf // tubelet
    pad_to = ((nf + clip_len - 1) // clip_len) * clip_len
    if pad_to > nf:
        pad = t[-1:].expand(pad_to - nf, -1, -1, -1)
        t = torch.cat([t, pad], dim=0)

    num_clips = pad_to // clip_len
    crop_size = t.shape[-1]
    clips = t.view(num_clips, clip_len, 3, crop_size, crop_size).permute(0, 2, 1, 3, 4)

    all_tokens = []
    for i in range(0, num_clips, batch_size):
        batch = clips[i:i + batch_size].to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = feat_extractor(batch)  # [B, N, D]
        all_tokens.append(out.cpu().float())
    all_tokens = torch.cat(all_tokens, dim=0)

    tokens_per_clip = clip_len // tubelet
    num_spatial = (crop_size // 16) ** 2
    embed_dim = all_tokens.shape[-1]
    all_tokens = all_tokens.view(num_clips, tokens_per_clip, num_spatial, embed_dim)
    all_tokens = all_tokens.mean(dim=2)  # spatial mean
    all_tokens = all_tokens.reshape(-1, embed_dim)[:num_tokens]

    return all_tokens  # [T, D]


def run_classifier(model, vjepa_feats, hand_kpts, window_size, tubelet, device):
    """
    Run skill classifier on every token position.
    Returns per-token predictions and confidence.
    """
    num_tokens = vjepa_feats.shape[0]
    D_v = vjepa_feats.shape[1]

    # Downsample hand to token rate
    hand_tokens = []
    for t in range(num_tokens):
        f0 = t * tubelet
        f1 = min(t * tubelet + 1, hand_kpts.shape[0] - 1)
        h = (hand_kpts[f0] + hand_kpts[f1]) / 2.0
        hand_tokens.append(torch.from_numpy(h.reshape(-1)))
    hand_tokens = torch.stack(hand_tokens)  # [T, 126]
    D_h = hand_tokens.shape[1]

    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for t in range(num_tokens):
            # Build window
            start = t - window_size + 1
            if start >= 0:
                v_win = vjepa_feats[start:t + 1]
                h_win = hand_tokens[start:t + 1]
            else:
                pad_len = -start
                v_win = torch.cat([torch.zeros(pad_len, D_v), vjepa_feats[:t + 1]])
                h_win = torch.cat([torch.zeros(pad_len, D_h), hand_tokens[:t + 1]])

            v_win = v_win.unsqueeze(0).to(device)  # [1, W, D_v]
            h_win = h_win.unsqueeze(0).to(device)  # [1, W, D_h]
            logits = model(v_win, h_win)  # [1, C]
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_probs.append(probs[0].cpu().numpy())

    return np.array(all_preds), np.stack(all_probs)


def run_classifier_aligned(
    model,
    vjepa_feats,
    hand_tokens,
    window_size,
    device,
):
    """Run the classifier on already token-aligned bundle features."""

    vjepa = torch.as_tensor(vjepa_feats, dtype=torch.float32)
    hand = torch.as_tensor(hand_tokens, dtype=torch.float32)
    if vjepa.ndim != 2 or hand.ndim != 2 or len(vjepa) != len(hand):
        raise ValueError(
            f"aligned V-JEPA/MANO shapes do not match: {vjepa.shape}/{hand.shape}"
        )
    all_preds = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for token_index in range(len(vjepa)):
            start = token_index - window_size + 1
            if start >= 0:
                vjepa_window = vjepa[start : token_index + 1]
                hand_window = hand[start : token_index + 1]
            else:
                padding = -start
                vjepa_window = torch.cat(
                    [torch.zeros(padding, vjepa.shape[1]), vjepa[: token_index + 1]]
                )
                hand_window = torch.cat(
                    [torch.zeros(padding, hand.shape[1]), hand[: token_index + 1]]
                )
            logits = model(
                vjepa_window.unsqueeze(0).to(device),
                hand_window.unsqueeze(0).to(device),
            )
            probabilities = F.softmax(logits, dim=-1)
            all_preds.append(int(logits.argmax(dim=-1).item()))
            all_probs.append(probabilities[0].cpu().numpy())
    return np.asarray(all_preds), np.stack(all_probs)


def run_classifier_mano_only(model, hand_feats, window_size, device):
    """Run hand-only classifier per frame (no V-JEPA, no token downsampling)."""
    num_frames = hand_feats.shape[0]
    D_h = hand_feats.shape[1]
    all_preds, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for t in range(num_frames):
            start = t - window_size + 1
            if start >= 0:
                h_win = hand_feats[start:t + 1]
            else:
                pad_len = -start
                h_win = torch.cat([torch.zeros(pad_len, D_h), hand_feats[:t + 1]])
            v_win = torch.zeros(1, window_size, 0).to(device)
            h_win = h_win.unsqueeze(0).to(device)
            logits = model(v_win, h_win)
            probs = F.softmax(logits, dim=-1)
            all_preds.append(logits.argmax(dim=-1).item())
            all_probs.append(probs[0].cpu().numpy())
    return np.array(all_preds), np.stack(all_probs)


def _token_mapping(num_frames, num_tokens, tubelet, frame_to_token=None):
    if frame_to_token is not None:
        mapping = np.asarray(frame_to_token, dtype=np.int64)
        if mapping.shape != (num_frames,):
            raise ValueError(
                f"frame_to_token shape {mapping.shape} != ({num_frames},)"
            )
        if mapping.min() < 0 or mapping.max() >= num_tokens:
            raise ValueError("frame_to_token contains an out-of-range token")
        return mapping
    mapping = np.arange(num_frames, dtype=np.int64) // int(tubelet)
    return np.clip(mapping, 0, num_tokens - 1)


def plot_timeline(
    preds,
    probs,
    num_frames,
    tubelet,
    title,
    save_path,
    gt_per_frame=None,
    frame_to_token=None,
):
    """Plot skill prediction timeline."""
    palette = sns.color_palette("husl", n_colors=len(LABELS))
    num_tokens = len(preds)
    mapping = _token_mapping(
        num_frames, num_tokens, tubelet, frame_to_token
    )
    preds_per_frame = preds[mapping]
    probs_per_frame = probs[mapping]

    has_gt = gt_per_frame is not None
    n_rows = 3 if has_gt else 2
    height_ratios = [1, 1, 3] if has_gt else [1, 3]
    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 6 + (1 if has_gt else 0)),
                             gridspec_kw={"height_ratios": height_ratios})

    from matplotlib.patches import Patch

    # Row 0: predicted label bar
    ax = axes[0]
    start = 0
    while start < num_frames:
        label = int(preds_per_frame[start])
        end = start + 1
        while end < num_frames and int(preds_per_frame[end]) == label:
            end += 1
        ax.barh(
            0,
            end - start,
            left=start,
            color=palette[label],
            height=1,
            edgecolor="none",
        )
        start = end
    ax.set_xlim(0, num_frames)
    ax.set_yticks([0])
    ax.set_yticklabels(["Pred"], fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold")

    legend_elements = [Patch(facecolor=palette[i], label=LABELS[i])
                       for i in range(len(LABELS))]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=6)

    # Row 1 (optional): GT label bar
    if has_gt:
        ax_gt = axes[1]
        # draw per-frame GT (1px wide columns, group by runs for efficiency)
        i = 0
        while i < num_frames:
            lbl = int(gt_per_frame[i])
            j = i + 1
            while j < num_frames and int(gt_per_frame[j]) == lbl:
                j += 1
            if lbl >= 0:
                ax_gt.barh(0, j - i, left=i, color=palette[lbl], height=1, edgecolor="none")
            else:
                ax_gt.barh(0, j - i, left=i, color=(0.85, 0.85, 0.85), height=1, edgecolor="none")
            i = j
        ax_gt.set_xlim(0, num_frames)
        ax_gt.set_yticks([0])
        ax_gt.set_yticklabels(["GT"], fontsize=9)

    # Last row: confidence heatmap
    ax = axes[-1]
    im = ax.imshow(probs_per_frame.T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[0, num_frames, len(LABELS) - 0.5, -0.5])
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("Frame", fontsize=11)
    plt.colorbar(im, ax=ax, label="Confidence", shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Timeline: {save_path}")


def render_prob_bar(probs_t, palette_255, panel_w=220, panel_h=256, bar_h=40, gt_idx=-1):
    """
    Render a vertical panel showing per-skill horizontal probability bars.
    gt_idx: if >= 0, highlight that row as the GT label.
    Returns numpy RGB array [panel_h, panel_w, 3].
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (panel_w, panel_h), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    n = len(LABELS)
    row_h = (panel_h - bar_h) // n   # reserve top bar_h for header
    max_bar_w = panel_w - 90          # leave space for label

    # Header
    draw.rectangle([0, 0, panel_w, bar_h - 1], fill=(50, 50, 50))
    draw.text((8, 10), "Skill Confidence", fill=(220, 220, 220))
    if gt_idx >= 0:
        draw.text((8, 24), f"GT: {LABELS[gt_idx]}", fill=(250, 220, 50))

    for i, label in enumerate(LABELS):
        y = bar_h + i * row_h
        p = float(probs_t[i])
        bar_w = int(p * max_bar_w)
        color = palette_255[i]

        is_gt = (i == gt_idx)
        # Background row — brighter for GT
        bg_color = (80, 70, 20) if is_gt else (40, 40, 40)
        draw.rectangle([0, y, panel_w, y + row_h - 2], fill=bg_color)
        # GT highlight border
        if is_gt:
            draw.rectangle([0, y, panel_w - 1, y + row_h - 2], outline=(250, 220, 50), width=2)
        # Probability bar
        if bar_w > 0:
            draw.rectangle([85, y + 3, 85 + bar_w, y + row_h - 5], fill=color)
        # Label
        label_color = (250, 220, 50) if is_gt else (200, 200, 200)
        draw.text((4, y + row_h // 2 - 6), f"{label}", fill=label_color)
        # Value text
        draw.text((85 + max_bar_w + 4, y + row_h // 2 - 6), f"{p:.2f}", fill=(180, 180, 180))

    return np.array(img)


def make_annotated_video(rgb_dir, frame_names, preds, probs, tubelet, save_path,
                         crop_size=256, fps=15, gt_per_frame=None,
                         frame_to_token=None):
    """Create video with skill label bar (bottom) + probability bars (right)."""
    from PIL import Image, ImageDraw

    palette = sns.color_palette("husl", n_colors=len(LABELS))
    palette_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in palette]

    bar_h = 40       # bottom label bar height
    panel_w = 220    # right probability panel width
    H, W = crop_size, crop_size
    total_w = W + panel_w
    total_h = H + bar_h

    tmp_path = save_path.replace(".mp4", "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (total_w, total_h))

    num_tokens = len(preds)
    mapping = _token_mapping(
        len(frame_names), num_tokens, tubelet, frame_to_token
    )
    for fi, fn in enumerate(frame_names):
        t = int(mapping[fi])

        pred_idx = preds[t]
        pred_label = LABELS[pred_idx]
        conf = probs[t, pred_idx]
        color = palette_255[pred_idx]

        # Original frame
        img = Image.open(_frame_path(rgb_dir, fn)).convert("RGB")
        img = img.resize((W, H), Image.BILINEAR)

        # Bottom label bar — green if pred matches GT, red if not, neutral if no GT
        gt_idx = int(gt_per_frame[fi]) if gt_per_frame is not None and gt_per_frame[fi] >= 0 else -1
        if gt_idx < 0:
            bar_color = color                        # original skill color (no GT)
        elif pred_idx == gt_idx:
            bar_color = (34, 139, 34)               # forest green
        else:
            bar_color = (180, 30, 30)               # red
        bar = Image.new("RGB", (total_w, bar_h), bar_color)
        draw = ImageDraw.Draw(bar)
        text = f"PRED: {pred_label}  (conf: {conf:.2f})"
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((total_w - tw) // 2, (bar_h - th) // 2), text, fill=(255, 255, 255))
        prob_panel = render_prob_bar(probs[t], palette_255, panel_w=panel_w, panel_h=H, bar_h=bar_h,
                                     gt_idx=gt_idx)
        prob_panel_img = Image.fromarray(prob_panel)

        # Compose
        combined = Image.new("RGB", (total_w, total_h), (0, 0, 0))
        combined.paste(img, (0, 0))
        combined.paste(prob_panel_img, (W, 0))
        combined.paste(bar, (0, H))

        frame_bgr = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    # Re-encode to H.264 for compatibility with VSCode / browsers
    os.system(f"ffmpeg -y -i {tmp_path} -vcodec libx264 -crf 18 -pix_fmt yuv420p {save_path} -loglevel error")
    os.remove(tmp_path)
    print(f"  Video: {save_path}")


def load_gt_labels(gt_path, num_frames):
    """
    Load gt_labels.json and return a per-frame label array (int).
    Frames not covered by any segment → -1.
    Labels not in LABELS (e.g. Trans when transition labels are excluded) → -1.
    """
    with open(gt_path) as f:
        gt = json.load(f)
    per_frame = np.full(num_frames, -1, dtype=np.int32)
    for seg in gt.get("segments", []):
        label = seg["label"]
        if label not in LABELS:
            continue
        idx = LABELS.index(label)
        s = max(0, int(seg["start_frame"]))
        e = min(num_frames - 1, int(seg["end_frame"]))
        per_frame[s:e + 1] = idx
    return per_frame


def evaluate_against_gt(preds_per_frame, gt_per_frame, num_frames):
    """
    Compare per-frame predictions vs GT.
    Only evaluates frames where GT is assigned (gt != -1).
    Returns a dict with overall accuracy + per-class stats.
    """
    mask = gt_per_frame >= 0
    if mask.sum() == 0:
        return None

    p = preds_per_frame[mask]
    g = gt_per_frame[mask]
    n_eval = int(mask.sum())
    n_skip = int(num_frames - n_eval)

    overall_acc = float((p == g).sum()) / n_eval

    per_class = {}
    for i, label in enumerate(LABELS):
        gt_mask = g == i
        if gt_mask.sum() == 0:
            continue
        correct = int((p[gt_mask] == i).sum())
        total   = int(gt_mask.sum())
        per_class[label] = {"correct": correct, "total": total,
                            "acc": correct / total}

    return {
        "overall_acc": overall_acc,
        "n_eval_frames": n_eval,
        "n_skipped_frames": n_skip,
        "per_class": per_class,
    }


def print_eval_results(results, ep_name):
    if results is None:
        print("  [GT eval] No GT frames to evaluate.")
        return
    print(f"\n  [GT eval] {ep_name}")
    print(f"  Overall accuracy : {results['overall_acc']*100:.1f}%  "
          f"({results['n_eval_frames']} frames evaluated, "
          f"{results['n_skipped_frames']} skipped / unlabeled)")
    print(f"  {'Label':<8} {'Correct':>7} {'Total':>7} {'Acc':>7}")
    print(f"  {'-'*32}")
    for label, s in results["per_class"].items():
        print(f"  {label:<8} {s['correct']:>7} {s['total']:>7} {s['acc']*100:>6.1f}%")


def main():
    global LABELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to long-horizon episodes (e.g. data/kitchen_dataset/0412_val)")
    parser.add_argument("--vjepa_ckpt", type=str, default=None,
                        help="V-JEPA checkpoint (not needed for hand-only models)")
    parser.add_argument("--classifier_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/long_horizon")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--mano", action="store_true",
                        help="Use mano hand_pose features (hand_dim=270) instead of kpts (126)")
    parser.add_argument("--no_video", action="store_true",
                        help="Skip annotated video generation")
    parser.add_argument("--image_dir", type=str, default="rgb",
                        help="Subdirectory name for images (default: rgb)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load classifier checkpoint to get training args
    cls_ckpt = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=False)
    cls_args = cls_ckpt["args"]
    model_name = cls_args["model"]
    window_size = cls_args["window_size"]
    vjepa_dim = cls_args.get("vjepa_dim", 1024)
    hand_dim = cls_args.get("hand_dim", 270 if args.mano else 126)
    exp_id = Path(args.classifier_ckpt).parent.name
    mano_only = (vjepa_dim == 0)

    # Set labels from checkpoint
    LABELS = cls_args.get("active_labels", list(ACTION_LABELS))
    num_classes = len(LABELS)

    print(f"Classifier: {model_name}, window_size={window_size}")
    print(f"  val_acc={cls_ckpt.get('val_acc', '?')}, epoch={cls_ckpt.get('epoch', '?')}")
    print(f"  Labels ({num_classes}): {LABELS}")
    if mano_only:
        print("  Mode: MANO hand-only (no V-JEPA)")

    # Rebuild classifier
    model_kwargs = dict(
        vjepa_dim=vjepa_dim, hand_dim=hand_dim,
        window_size=window_size, num_classes=num_classes,
    )
    if model_name == "mlp":
        model_kwargs.update(
            hidden_dims=tuple(cls_args.get("hidden_dims", [512, 256])),
            dropout=cls_args.get("dropout", 0.3),
            pool=cls_args.get("pool", "mean"),
        )
    elif model_name == "transformer":
        model_kwargs.update(
            d_model=cls_args.get("d_model", 256),
            nhead=cls_args.get("nhead", 4),
            num_layers=cls_args.get("num_layers", 2),
            dropout=cls_args.get("dropout", 0.1),
        )
    classifier = build_model(model_name, **model_kwargs).to(device)
    classifier.load_state_dict(cls_ckpt["model"])
    classifier.eval()

    # Discover episodes
    episodes = discover_episodes(args.data_dir, mano=args.mano, image_dir=args.image_dir)
    print(f"Found {len(episodes)} episodes")
    if not episodes:
        raise ValueError(f"no usable episodes under {args.data_dir}")

    variant = cls_args.get("variant", "mano_only")
    bundle_feature_key = {
        "mano_only": None,
        "vjepa_orig": "vjepa_orig",
        "masked_vjepa_orig": "vjepa_orig_masked",
        "vjepa_robot": "vjepa_robot",
    }.get(variant)
    if variant != "mano_only" and bundle_feature_key is None:
        raise ValueError(f"unknown classifier variant: {variant}")
    use_feature_bundles = all(ep["features_path"] is not None for ep in episodes)
    expected_sampling_signature = cls_args.get("sampling_signature")
    if expected_sampling_signature is not None:
        expected_sampling_signature = tuple(expected_sampling_signature)
        if expected_sampling_signature[0] != "legacy_dense" and not use_feature_bundles:
            raise ValueError(
                "this classifier requires aligned features.pt bundles; "
                "direct dense inference would violate its sampling contract"
            )

    # Load V-JEPA only for the legacy direct-extraction fallback. New 4-FPS
    # checkpoints consume their persisted alignment bundle instead.
    encoder, feat_extractor = None, None
    if not mano_only and not use_feature_bundles:
        if args.vjepa_ckpt is None:
            raise ValueError("--vjepa_ckpt is required for direct V-JEPA extraction")
        from data_preprocess.feature_extractor import VJEPAFeatureExtractor, load_pretrained_encoder
        print("Loading V-JEPA encoder...")
        encoder = load_pretrained_encoder(
            checkpoint_path=args.vjepa_ckpt, device=device,
            model_name="vit_large", crop_size=args.crop_size,
            patch_size=16, num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
        )
        feat_extractor = VJEPAFeatureExtractor(encoder, pool="none").to(device)
    elif use_feature_bundles:
        print("Using persisted aligned features.pt bundles")

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in episodes:
        name = ep["name"]
        nf = ep["num_frames"]
        print(f"\n{'='*60}")
        print(f"Episode: {name} ({nf} frames)")
        frame_to_token = None

        if use_feature_bundles:
            from data_preprocess.preprocess import (
                feature_sampling_signature,
                validate_feature_bundle,
            )

            bundle = torch.load(
                ep["features_path"], map_location="cpu", weights_only=True
            )
            validate_feature_bundle(bundle)
            actual_signature = feature_sampling_signature(bundle)
            if expected_sampling_signature is None:
                if actual_signature[0] != "legacy_dense":
                    raise ValueError(
                        "classifier checkpoint has no sampling signature and "
                        f"cannot consume {actual_signature[0]} bundles"
                    )
            elif actual_signature != expected_sampling_signature:
                raise ValueError(
                    f"classifier/bundle sampling mismatch: "
                    f"{expected_sampling_signature} != {actual_signature}"
                )
            hand_tokens = bundle["mano"]
            if hand_tokens.shape[1] != hand_dim:
                raise ValueError(
                    f"classifier hand_dim {hand_dim} != bundle {hand_tokens.shape[1]}"
                )
            if bundle_feature_key is None:
                vjepa_feats = torch.zeros(len(hand_tokens), 0)
            else:
                if bundle_feature_key not in bundle:
                    raise ValueError(
                        f"bundle missing classifier feature {bundle_feature_key}"
                    )
                vjepa_feats = bundle[bundle_feature_key]
            if vjepa_feats.shape[1] != vjepa_dim:
                raise ValueError(
                    f"classifier vjepa_dim {vjepa_dim} != bundle {vjepa_feats.shape[1]}"
                )
            if cls_args.get("vjepa_diff", False) and vjepa_feats.shape[1] > 0:
                vjepa_difference = torch.zeros_like(vjepa_feats)
                vjepa_difference[1:] = vjepa_feats[1:] - vjepa_feats[:-1]
                vjepa_feats = vjepa_difference
            preds, probs = run_classifier_aligned(
                classifier,
                vjepa_feats,
                hand_tokens,
                window_size,
                device,
            )
            frame_to_token = np.asarray(bundle["frame_to_token"], dtype=np.int64)
            if frame_to_token.shape != (nf,):
                raise ValueError(
                    f"bundle/source frame mismatch: {frame_to_token.shape} vs {nf}"
                )
            num_tokens = len(preds)
            tubelet = int(bundle["tubelet_size"])
        elif mano_only:
            # Hand-only: extract MANO axis-angle from result.json
            print("  Extracting MANO axis-angle features...")
            hand_feats = load_mano_axis_angle(ep["json_path"], ep["frame_names"])
            tubelet = 1  # per-frame, no token downsampling

            print("  Running classifier...")
            preds, probs = run_classifier_mano_only(
                classifier, hand_feats, window_size, device,
            )
            num_tokens = nf
        else:
            # V-JEPA + hand mode
            mean = torch.tensor([0.485, 0.456, 0.406]) * 255.0
            std = torch.tensor([0.229, 0.224, 0.225]) * 255.0
            tubelet = args.tubelet_size

            frames_np = load_frames(ep["rgb_dir"], ep["frame_names"], args.crop_size)
            if hand_dim == 0:
                # V-JEPA only — no hand input
                hand_kpts = np.zeros((nf, 0), dtype=np.float32)
            elif hand_dim == 96:
                hand_kpts = load_mano_axis_angle(ep["json_path"], ep["frame_names"]).numpy()
            elif args.mano:
                hand_kpts = load_hand_pose_mano(ep["json_path"], ep["frame_names"])
            else:
                hand_kpts = load_hand_kpts(ep["json_path"], ep["frame_names"])

            print("  Extracting V-JEPA features...")
            vjepa_feats = extract_vjepa_features(
                encoder, feat_extractor, frames_np, mean, std, device,
                clip_len=args.num_frames, tubelet=tubelet,
                batch_size=args.batch_size,
            )
            if cls_args.get("vjepa_diff", False):
                print("  Applying V-JEPA diff...")
                vjepa_d = torch.zeros_like(vjepa_feats)
                vjepa_d[1:] = vjepa_feats[1:] - vjepa_feats[:-1]
                vjepa_feats = vjepa_d
            num_tokens = vjepa_feats.shape[0]

            print("  Running classifier...")
            preds, probs = run_classifier(
                classifier, vjepa_feats, hand_kpts, window_size,
                tubelet, device,
            )

        print(f"  {nf} frames -> {num_tokens} steps")

        mapping = _token_mapping(
            nf, len(preds), tubelet, frame_to_token
        )
        preds_per_frame = preds[mapping]

        # Print skill sequence summary in original source-frame coordinates.
        segments = []
        cur_label, cur_start = preds_per_frame[0], 0
        for frame_index in range(1, nf):
            if preds_per_frame[frame_index] != cur_label:
                segments.append((cur_start, frame_index - 1, int(cur_label)))
                cur_label, cur_start = preds_per_frame[frame_index], frame_index
        segments.append((cur_start, nf - 1, int(cur_label)))

        print("  Predicted skill sequence:")
        for s_start, s_end, s_label in segments:
            dur_frames = s_end - s_start + 1
            print(f"    [{s_start:4d}-{s_end:4d}] {LABELS[s_label]:>5s}  "
                  f"({dur_frames} frames)")

        # Save results
        ep_dir = os.path.join(args.output_dir, name)
        os.makedirs(ep_dir, exist_ok=True)

        # GT evaluation (if gt_labels.json exists)
        gt_path = Path(args.data_dir) / name / "gt_labels.json"
        gt_per_frame = None
        eval_results = None
        if gt_path.exists():
            gt_per_frame = load_gt_labels(str(gt_path), nf)
            eval_results = evaluate_against_gt(preds_per_frame, gt_per_frame, nf)
            print_eval_results(eval_results, name)
        else:
            print(f"  [GT eval] No gt_labels.json found, skipping evaluation.")

        # Save raw predictions
        torch.save({
            "preds": preds,
            "probs": probs,
            "preds_per_frame": preds_per_frame,
            "num_frames": nf,
            "segments": segments,
            "frame_to_token": mapping,
            "sampling_signature": expected_sampling_signature,
            "eval_results": eval_results,
            "labels": LABELS,
        }, os.path.join(ep_dir, "predictions.pt"))

        if eval_results is not None:
            with open(os.path.join(ep_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f, indent=2)

        # Timeline plot
        plot_timeline(preds, probs, nf, tubelet,
                      f"Skill Predictions: {name}",
                      os.path.join(ep_dir, "timeline.png"),
                      gt_per_frame=gt_per_frame,
                      frame_to_token=mapping)

        # Annotated video
        if not args.no_video and ep.get("rgb_dir"):
            print("  Making annotated video...")
            make_annotated_video(
                ep["rgb_dir"], ep["frame_names"], preds, probs,
                tubelet, os.path.join(ep_dir, f"{exp_id}_{name}.mp4"),
                crop_size=args.crop_size, fps=args.fps,
                gt_per_frame=gt_per_frame,
                frame_to_token=mapping,
            )

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
