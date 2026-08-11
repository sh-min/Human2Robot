"""Collect skill classifier experiments + long-horizon results into CSVs.

Outputs:
    output/results/experiments.csv     One row per checkpoint with metadata + val_acc
    output/results/long_horizon.csv    One row per (experiment, episode) with accuracy

Run after every batch of experiments. Re-running is idempotent.

Usage:
    python collect_results.py
"""

import csv
import json
import os
import re
from pathlib import Path

import torch

CLS_ROOT = Path("output/skill_classifier")
LH_ROOT = Path("output/long_horizon")
OUT = Path("output/results")
OUT.mkdir(parents=True, exist_ok=True)


def infer_variant(train_dir):
    """Map train_feature_dir → input variant name."""
    if "merged_masked" in train_dir:
        return "masked"
    if "merged" in train_dir:
        return "vjepa_mano"
    if "mano_features" in train_dir or "skill_mano" in train_dir:
        return "mano_only"
    return "?"


def infer_ckpt(train_dir):
    """Infer V-JEPA checkpoint source from feature dir name.

    `_orig` suffix → official V-JEPA2 baseline (no fine-tune).
    Anything else → EgoDex fine-tuned (default for older runs).
    """
    return "orig" if "_orig" in train_dir else "finetuned"


def infer_ext(train_dir):
    return "_ext" in train_dir


def collect_experiments():
    rows = []
    for exp_dir in sorted(CLS_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        # Find best checkpoint
        best = next(exp_dir.glob("best_*.pt"), None)
        if best is None:
            continue
        try:
            c = torch.load(best, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  SKIP {exp_dir.name}: {e}")
            continue
        a = c.get("args", {})
        train_dir = a.get("train_feature_dir", "")
        rows.append({
            "exp_id": exp_dir.name,
            "model": a.get("model", "?"),
            "variant": infer_variant(train_dir),
            "window_size": a.get("window_size", "?"),
            "vjepa_dim": a.get("vjepa_dim", "?"),
            "vjepa_diff": bool(a.get("vjepa_diff", False)),
            "ext": infer_ext(train_dir),
            "vjepa_ckpt": infer_ckpt(train_dir),
            "include_trans": a.get("include_trans", "?"),
            "num_classes": len(a.get("active_labels", [])),
            "val_acc": float(c.get("val_acc", 0.0)),
            "best_epoch": int(c.get("epoch", 0)),
            "train_feature_dir": train_dir,
        })
    return rows


def collect_long_horizon(exp_rows):
    """For each experiment that has long_horizon eval, emit per-episode rows."""
    rows = []
    exp_meta = {r["exp_id"]: r for r in exp_rows}
    for lh_dir in sorted(LH_ROOT.iterdir()):
        if not lh_dir.is_dir():
            continue
        exp_id = lh_dir.name
        meta = exp_meta.get(exp_id)
        if not meta:
            continue
        for ep_dir in sorted(lh_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            f = ep_dir / "eval_results.json"
            if not f.exists():
                continue
            r = json.load(open(f))
            sc, st, tc, tt = 0, 0, 0, 0
            for lb, s in r["per_class"].items():
                if lb == "Trans":
                    tc += s["correct"]; tt += s["total"]
                else:
                    sc += s["correct"]; st += s["total"]
            rows.append({
                "exp_id": exp_id,
                "episode": ep_dir.name,
                "model": meta["model"],
                "variant": meta["variant"],
                "window_size": meta["window_size"],
                "vjepa_diff": meta["vjepa_diff"],
                "ext": meta["ext"],
                "vjepa_ckpt": meta["vjepa_ckpt"],
                "overall_acc": r["overall_acc"] * 100,
                "skill_acc": (sc / st * 100) if st > 0 else None,
                "trans_acc": (tc / tt * 100) if tt > 0 else None,
                "n_eval_frames": r.get("n_eval_frames"),
            })
    return rows


def write_csv(path, rows):
    if not rows:
        print(f"  (no rows for {path})")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path} ({len(rows)} rows)")


def main():
    print("Scanning experiments...")
    exp_rows = collect_experiments()
    write_csv(OUT / "experiments.csv", exp_rows)

    print("Scanning long-horizon results...")
    lh_rows = collect_long_horizon(exp_rows)
    write_csv(OUT / "long_horizon.csv", lh_rows)


if __name__ == "__main__":
    main()
