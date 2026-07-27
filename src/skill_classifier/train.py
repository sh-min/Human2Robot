"""
Train skill classifier on precomputed V-JEPA + hand pose features.

Usage:
    cd /virtual_lab/ljw_rvlab/byeonggyeol/3dgs-visual-grounding/skill2policy
    PYTHONPATH=$PWD/src python -m skill_classifier.train \
        --config src/skill_classifier/config/transformer.yaml

Experiment output:
    output/skill_classifier/{exp_id}/
        config.yaml          # resolved config (YAML used + CLI overrides)
        best_{model}.pt
        last_{model}.pt
        e{N}_{model}.pt      # periodic checkpoints
"""

import argparse
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Sampler
import wandb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.labels import ACTION_LABELS, TRANSITION_LABEL
from skill_classifier.models import build_model
from skill_classifier.skill_dataset import (
    SkillWindowDataset, load_recordings, VARIANT_VJEPA_KEY,
)


def compute_f1(all_labels, all_preds, num_classes):
    """Compute macro and weighted F1 without sklearn."""
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    f1s, supports = [], []
    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        supports.append((all_labels == c).sum())
    f1s, supports = np.array(f1s), np.array(supports, dtype=float)
    macro = f1s.mean()
    weighted = (f1s * supports).sum() / supports.sum() if supports.sum() > 0 else 0.0
    return float(macro), float(weighted)


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path, title="Confusion Matrix"):
    """Save a confusion matrix figure with raw counts and normalized (recall) views."""
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for gt, pred in zip(all_labels, all_preds):
        cm[gt][pred] += 1

    # Row-normalized (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, fmt, subtitle in [
        (axes[0], cm, "d", "Raw Counts"),
        (axes[1], cm_norm, ".2f", "Normalized (Recall)"),
    ]:
        im = ax.imshow(data, cmap="Blues", aspect="equal")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"{title} — {subtitle}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Annotate cells
        thresh = data.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                val = format(data[i, j], fmt)
                ax.text(j, i, val, ha="center", va="center", fontsize=6,
                        color="white" if data[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def compute_class_weights(dataset, num_classes, strategy="inverse"):
    """Compute class weights from dataset sample distribution.

    Args:
        dataset: SkillWindowDataset instance
        num_classes: number of classes
        strategy: 'none', 'inverse', or 'sqrt_inverse'

    Returns:
        Tensor of shape [num_classes] or None if strategy is 'none'.
    """
    if strategy == "none":
        return None
    label_counts = Counter(
        (s[2] if isinstance(s, tuple) else s["label"]) for s in dataset.samples
    )
    total = sum(label_counts.values())
    weights = []
    for c in range(num_classes):
        count = label_counts.get(c, 1)
        w = total / (num_classes * count)
        if strategy == "sqrt_inverse":
            w = math.sqrt(w)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


class BalancedClassSampler(Sampler):
    """Per-epoch sampler that caps over-represented classes via random sub-sampling.

    Every epoch, all samples from under-cap classes are included in full.
    For classes exceeding the cap, a fresh random subset is drawn — so the
    model sees different transition (or other majority-class) samples each epoch.
    """

    def __init__(self, dataset, max_samples="mean"):
        self.indices_per_class = {}
        for idx, s in enumerate(dataset.samples):
            # samples are (rec_idx, t, label) tuples
            label = s[2] if isinstance(s, tuple) else s["label"]
            self.indices_per_class.setdefault(label, []).append(idx)

        counts = [len(v) for v in self.indices_per_class.values()]
        if max_samples == "mean":
            self.cap = int(np.mean(counts))
        elif max_samples == "median":
            self.cap = int(np.median(counts))
        else:
            self.cap = int(max_samples)

        # Log effective cap
        for cls, idxs in sorted(self.indices_per_class.items()):
            tag = " (capped)" if len(idxs) > self.cap else ""
            print(f"  class {cls:2d}: {len(idxs):5d} samples{tag}")
        print(f"  cap per class: {self.cap}")

    def __iter__(self):
        indices = []
        for cls_indices in self.indices_per_class.values():
            if len(cls_indices) <= self.cap:
                indices.extend(cls_indices)
            else:
                indices.extend(
                    np.random.choice(cls_indices, self.cap, replace=False).tolist()
                )
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return sum(min(len(v), self.cap) for v in self.indices_per_class.values())


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for vjepa, hand, labels in loader:
        vjepa, hand, labels = vjepa.to(device), hand.to(device), labels.to(device)
        logits = model(vjepa, hand)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=13):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for vjepa, hand, labels in loader:
        vjepa, hand, labels = vjepa.to(device), hand.to(device), labels.to(device)
        logits = model(vjepa, hand)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1_macro, f1_weighted = compute_f1(all_labels, all_preds, num_classes)
    return total_loss / total, acc, f1_macro, f1_weighted, all_preds, all_labels


def load_config(config_path):
    """Load YAML config as dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_exp_id(cfg):
    ts = datetime.now().strftime("%m%d_%H%M")
    return f"{cfg['model']}_w{cfg['window_size']}_{ts}"


def apply_overrides(cfg, overrides):
    """
    Apply key=value overrides to cfg dict.
    Values are auto-cast: int, float, bool, or string.
    List values use comma separation: hidden_dims=512,256
    """
    def cast(v):
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value format, got: {item!r}")
        k, v = item.split("=", 1)
        if k.endswith("recording_glob"):
            cfg[k] = v
        elif "," in v:
            cfg[k] = [cast(x) for x in v.split(",")]
        else:
            cfg[k] = cast(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment ID (auto-generated from model+timestamp if not set)")
    parser.add_argument("--set", nargs="*", default=[],
                        metavar="key=value",
                        help="Override config values, e.g. --set epochs=500 lr=1e-3")
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    if cli.set:
        apply_overrides(cfg, cli.set)

    # Experiment ID and save dir
    exp_id = cli.exp_id or make_exp_id(cfg)
    save_dir = os.path.join(cfg["output_dir"], exp_id)
    os.makedirs(save_dir, exist_ok=True)

    # Save resolved config
    resolved_cfg_path = os.path.join(save_dir, "config.yaml")
    with open(resolved_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)

    # Build args namespace for rest of code
    import types
    args = types.SimpleNamespace(**cfg)
    args.exp_id = exp_id

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Experiment: {exp_id}")
    print(f"Config saved to: {save_dir}/config.yaml")

    # wandb
    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=exp_id,
            config=cfg,
        )

    # Split
    # Load per-recording bundled features.
    # train_data_root + train_recording_glob → globs {root}/{glob}/features.pt
    train_root = args.train_data_root
    val_root = args.val_data_root
    train_glob = getattr(args, "train_recording_glob", "*")
    val_glob = getattr(args, "val_recording_glob", "*")
    print(f"Train: {train_root}/{train_glob}/features.pt")
    print(f"Val:   {val_root}/{val_glob}/features.pt")
    train_recs = load_recordings(train_root, train_glob)
    val_recs = load_recordings(val_root, val_glob)
    print(f"Train: {len(train_recs)} recordings, Val: {len(val_recs)} recordings")

    # Filter per-token labels for include_trans
    include_trans = getattr(args, "include_trans", True)
    if not include_trans:
        trans_idx = ACTION_LABELS.index(TRANSITION_LABEL)
        for rec in train_recs + val_recs:
            mask = rec["labels_per_token"] == trans_idx
            rec["labels_per_token"][mask] = -1
        active_labels = [a for a in ACTION_LABELS if a != TRANSITION_LABEL]
        print(f"{TRANSITION_LABEL} tokens masked out from training/eval samples.")
    else:
        active_labels = list(ACTION_LABELS)
    num_classes = len(active_labels)

    variant = getattr(args, "variant", "mano_only")
    if variant not in VARIANT_VJEPA_KEY:
        raise ValueError(f"Unknown variant: {variant}")
    vjepa_diff = getattr(args, "vjepa_diff", False)
    if vjepa_diff:
        print("V-JEPA diff mode: using vjepa[t]-vjepa[t-1]")

    train_ds = SkillWindowDataset(train_recs, window_size=args.window_size,
                                  variant=variant, vjepa_diff=vjepa_diff)
    val_ds = SkillWindowDataset(val_recs, window_size=args.window_size,
                                variant=variant, vjepa_diff=vjepa_diff)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    hand_dim = train_ds.hand_dim
    vjepa_dim_inferred = train_ds.vjepa_dim

    # Balanced sampling: cap over-represented classes per epoch
    balance_cfg = getattr(args, "balance_sampling", "none")
    if balance_cfg and balance_cfg != "none":
        print(f"Balanced sampling ({balance_cfg}):")
        sampler = BalancedClassSampler(train_ds, max_samples=balance_cfg)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Save dims/labels/variant into args so they're stored in checkpoints
    args.hand_dim = hand_dim
    args.vjepa_dim = vjepa_dim_inferred
    args.active_labels = active_labels
    print(f"variant: {variant}  vjepa_dim: {vjepa_dim_inferred}  hand_dim: {hand_dim}")
    print(f"Active labels ({num_classes}): {active_labels}")

    # Build model
    model_kwargs = dict(
        vjepa_dim=vjepa_dim_inferred,
        hand_dim=hand_dim,
        window_size=args.window_size,
        num_classes=num_classes,
    )
    if args.model == "mlp":
        model_kwargs.update(
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            pool=args.pool,
        )
    elif args.model == "transformer":
        model_kwargs.update(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    model = build_model(args.model, **model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({num_params:,} params)")
    print(model)

    # Class weights for imbalanced data
    weight_strategy = getattr(args, "class_weights", "none")
    class_wts = compute_class_weights(train_ds, num_classes, strategy=weight_strategy)
    if class_wts is not None:
        class_wts = class_wts.to(device)
        print(f"Class weights ({weight_strategy}):")
        for name, w in zip(active_labels, class_wts):
            print(f"  {name:5s}: {w:.4f}")
    else:
        print("Class weights: none (uniform)")

    # Optimizer + cosine schedule with linear warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=class_wts, label_smoothing=args.label_smoothing)

    # Train
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.grad_clip,
        )
        val_loss, val_acc, val_f1_macro, val_f1_weighted, preds, labels = evaluate(
            model, val_loader, criterion, device, num_classes=num_classes,
        )

        cur_lr = optimizer.param_groups[0]["lr"]

        # Logging
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1_macro": val_f1_macro,
            "val/f1_weighted": val_f1_weighted,
            "lr": cur_lr,
        }
        if not args.no_wandb:
            wandb.log(log_dict)

        # Confusion matrix: save plot at best or every save_every epochs
        should_plot_cm = (val_acc > best_val_acc
                          or (args.save_every > 0 and epoch % args.save_every == 0)
                          or epoch == args.epochs)
        if should_plot_cm:
            cm_path = os.path.join(save_dir, f"confusion_matrix_e{epoch}.png")
            cm_fig = plot_confusion_matrix(
                labels, preds, active_labels, cm_path,
                title=f"Epoch {epoch}",
            )
            print(f"  Confusion matrix saved: {cm_path}")
            if not args.no_wandb:
                wandb.log({"val/confusion_matrix": wandb.Image(cm_path)})

        # Best model
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1_macro": val_f1_macro,
                "args": vars(args),
            }, os.path.join(save_dir, f"best_{args.model}.pt"))
        # Periodic checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "args": vars(args),
            }, os.path.join(save_dir, f"e{epoch}_{args.model}.pt"))

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(f"  [{epoch:3d}/{args.epochs}] "
                  f"loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                  f"f1={val_f1_macro:.3f} lr={cur_lr:.2e}"
                  f"{' *' if improved else ''}")

    # Save last
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "args": vars(args),
    }, os.path.join(save_dir, f"last_{args.model}.pt"))

    print(f"\nBest val acc: {best_val_acc:.3f}")
    print(f"Experiment: {exp_id}")
    print(f"Checkpoints saved to {save_dir}/")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
