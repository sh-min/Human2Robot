"""Offline action-prediction MSE on a held-out LeRobot dataset.

Loads a trained LeRobot policy checkpoint and the val LeRobotDataset,
runs ``policy.select_action`` frame-by-frame within each episode, and
reports per-dimension and aggregate MSE against the ground-truth action.

Usage:
    MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy_config.eval_offline \\
        --checkpoint output/train/full_100k/checkpoints/last/pretrained_model \\
        --val_dataset data/lerobot_xhand_val \\
        --output_dir output/eval_offline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


def _load_policy_and_processors(checkpoint: str, device: str, ds_meta):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    cfg.pretrained_path = checkpoint
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
    )
    return policy, preprocessor, postprocessor


@torch.inference_mode()
def evaluate(
    checkpoint: str,
    val_dataset_root: str,
    *,
    device: str = "cuda",
    output_dir: str = "output/eval_offline",
):
    dataset = LeRobotDataset(
        repo_id="xhand/local",
        root=val_dataset_root,
    )
    policy, preprocessor, postprocessor = _load_policy_and_processors(
        checkpoint, device=device, ds_meta=dataset.meta,
    )

    # Discover episode boundaries from dataset metadata.
    episodes_table = dataset.meta.episodes
    n_eps = len(episodes_table)
    print(f"Loaded val dataset: {n_eps} episodes, {dataset.num_frames} frames")

    per_episode = []
    sq_errs = []
    abs_errs = []

    for ep_idx in range(n_eps):
        ep_row = episodes_table[ep_idx]
        start = ep_row["dataset_from_index"]
        end = ep_row["dataset_to_index"]
        ep_len = end - start

        policy.reset()
        preds = []
        truths = []
        for global_idx in range(start, end):
            sample = dataset[global_idx]
            # Build the observation dict; add batch dim, leave device to preprocessor.
            obs = {
                k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
                for k, v in sample.items() if k.startswith("observation.")
            }
            obs = preprocessor(obs)
            out = policy.select_action(obs)
            out = postprocessor(out)
            a_pred = out.detach().cpu().numpy()
            if a_pred.ndim == 3:
                a_pred = a_pred[0, 0]
            elif a_pred.ndim == 2:
                a_pred = a_pred[0]

            a_true = sample["action"].numpy() if torch.is_tensor(sample["action"]) else np.asarray(sample["action"])
            preds.append(a_pred)
            truths.append(a_true)

        preds_arr = np.stack(preds, axis=0)
        truths_arr = np.stack(truths, axis=0)
        sq = np.square(preds_arr - truths_arr)
        ab = np.abs(preds_arr - truths_arr)
        sq_errs.append(sq)
        abs_errs.append(ab)
        per_episode.append({
            "episode_index": ep_idx,
            "length": int(ep_len),
            "mse_overall": float(sq.mean()),
            "mae_overall": float(ab.mean()),
            "mse_per_dim": sq.mean(axis=0).tolist(),
        })
        print(f"  ep {ep_idx}: T={ep_len}  MSE={sq.mean():.5f}  MAE={ab.mean():.5f}")

    all_sq = np.concatenate(sq_errs, axis=0)
    all_ab = np.concatenate(abs_errs, axis=0)
    agg = {
        "n_episodes": n_eps,
        "n_frames": int(all_sq.shape[0]),
        "mse_overall": float(all_sq.mean()),
        "mae_overall": float(all_ab.mean()),
        "mse_per_dim": all_sq.mean(axis=0).tolist(),
        "mae_per_dim": all_ab.mean(axis=0).tolist(),
        "per_episode": per_episode,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregate MSE={agg['mse_overall']:.5f}  MAE={agg['mae_overall']:.5f}")
    print(f"Wrote {out / 'metrics.json'}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="lerobot-train output 'pretrained_model' dir")
    ap.add_argument("--val_dataset", required=True, help="root directory of the val LeRobot dataset")
    ap.add_argument("--output_dir", default="output/eval_offline")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    evaluate(args.checkpoint, args.val_dataset, device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
