"""Compare calibration, latent clustering and classifier-head latency.

The script evaluates saved V-JEPA skill classifiers on the exact validation
split recorded in each checkpoint.  Object-grounded checkpoints additionally
load their immutable object-context sidecars.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    log_loss,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch.utils.data import DataLoader

from skill_classifier.models import build_model
from skill_classifier.skill_dataset import SkillWindowDataset, load_recordings


def _saved(value, default):
    return default if value is None else value


def _build_model(checkpoint, dataset):
    saved_args = checkpoint["args"]
    name = saved_args["model"]
    kwargs = {
        "vjepa_dim": dataset.vjepa_dim,
        "hand_dim": dataset.hand_dim,
        "window_size": int(saved_args["window_size"]),
        "num_classes": len(saved_args["active_labels"]),
        "hidden_dims": tuple(saved_args["hidden_dims"]),
        "dropout": float(saved_args["dropout"]),
    }
    if name in ("object_mask_attention_mlp", "object_text_prototype_mlp"):
        kwargs.update(
            object_prompt_count=dataset.object_prompt_count,
            object_mask_spatial_tokens=dataset.object_mask_spatial_tokens,
            object_projection_dim=int(_saved(saved_args.get("object_projection_dim"), 64)),
            use_global_features=bool(_saved(saved_args.get("use_global_features"), True)),
            use_object_features=bool(_saved(saved_args.get("use_object_features"), True)),
            use_confidence_features=bool(_saved(saved_args.get("use_confidence_features"), True)),
            use_occupancy_features=bool(_saved(saved_args.get("use_occupancy_features"), True)),
            use_confidence_gate=bool(_saved(saved_args.get("use_confidence_gate"), True)),
        )
        if name == "object_text_prototype_mlp":
            kwargs.update(
                text_embedding_dim=int(_saved(saved_args.get("text_embedding_dim"), 512)),
                text_head_mode=str(_saved(saved_args.get("text_head_mode"), "prototype")),
            )
    model = build_model(name, **kwargs)
    model.load_state_dict(checkpoint["model"])
    return model


def _dataset(saved_args):
    root = saved_args["val_data_root"]
    recordings = load_recordings(root, saved_args["val_recording_glob"])
    return SkillWindowDataset(
        recordings,
        window_size=int(saved_args["window_size"]),
        variant=saved_args["variant"],
        vjepa_diff=bool(saved_args.get("vjepa_diff", False)),
        hand_representation=saved_args.get("hand_representation", "none"),
        object_context_key=saved_args.get("object_context_key"),
    )


def _ece(probabilities, labels, bins=10):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    table = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            mask |= confidence == 0
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            value += count / len(labels) * abs(accuracy - mean_confidence)
        table.append({
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "accuracy": accuracy,
            "confidence": mean_confidence,
        })
    return float(value), table


def _brier(probabilities, labels, classes):
    targets = np.eye(classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


@torch.inference_mode()
def _evaluate(model, loader, device, object_model):
    logits_all, labels_all, representations = [], [], []
    for vjepa, context, labels in loader:
        vjepa, context = vjepa.to(device), context.to(device)
        logits = model(vjepa, context)
        if object_model:
            representation, _ = model.representation(vjepa, context)
        else:
            representation, _ = model.representation(vjepa)
        logits_all.append(logits.cpu())
        labels_all.append(labels)
        representations.append(representation.cpu())
    logits = torch.cat(logits_all)
    probability = logits.softmax(dim=1).numpy()
    return probability, torch.cat(labels_all).numpy(), torch.cat(representations).numpy()


@torch.inference_mode()
def _latency(model, sample, device, warmup, iterations):
    vjepa, context, _ = sample
    vjepa, context = vjepa.to(device), context.to(device)
    for _ in range(warmup):
        model(vjepa, context)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        model(vjepa, context)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    batch = len(vjepa)
    return {
        "batch_size": batch,
        "iterations": iterations,
        "batch_latency_ms": elapsed * 1000.0 / iterations,
        "sample_latency_ms": elapsed * 1000.0 / (iterations * batch),
        "samples_per_second": iterations * batch / elapsed,
    }


def _parse_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--model must be DISPLAY_NAME=CHECKPOINT")
        name, path = value.split("=", 1)
        specs.append((name, Path(path)))
    return specs


def _plot_reliability(results, output):
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.plot([0, 1], [0, 1], "--", color="black", label="perfect calibration")
    for result in results:
        occupied = [row for row in result["calibration_bins"] if row["count"]]
        axis.plot(
            [row["confidence"] for row in occupied],
            [row["accuracy"] for row in occupied],
            marker="o",
            label=f"{result['name']} (ECE={result['ece']:.3f})",
        )
    axis.set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title("Reliability diagram on the fixed validation split")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_embeddings(results, labels, class_names, output):
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 7), squeeze=False)
    colors = plt.get_cmap("tab10")
    for axis, result in zip(axes[0], results):
        points = result.pop("tsne_points")
        for class_index, class_name in enumerate(class_names):
            mask = labels == class_index
            axis.scatter(points[mask, 0], points[mask, 1], s=24, alpha=0.75,
                         color=colors(class_index), label=class_name)
        axis.set_title(
            f"{result['name']}\nARI={result['cluster_ari']:.3f}, "
            f"NMI={result['cluster_nmi']:.3f}, silhouette={result['silhouette']:.3f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0, -1].legend(loc="best", fontsize=8)
    fig.suptitle("t-SNE of learned classifier representations (color = ground truth)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="DISPLAY_NAME=CHECKPOINT; repeat for each model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-iterations", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    results, shared_labels, shared_class_names = [], None, None

    for display_name, checkpoint_path in _parse_specs(args.model):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_args = checkpoint["args"]
        dataset = _dataset(saved_args)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = _build_model(checkpoint, dataset).to(device).eval()
        object_model = saved_args["model"] in ("object_mask_attention_mlp", "object_text_prototype_mlp")
        probabilities, labels, representation = _evaluate(model, loader, device, object_model)
        if shared_labels is None:
            shared_labels = labels
            shared_class_names = list(saved_args["active_labels"])
        elif not np.array_equal(shared_labels, labels):
            raise ValueError("models must use the same ordered validation samples")

        ece, calibration_bins = _ece(probabilities, labels)
        pca_dimensions = min(50, representation.shape[0] - 1, representation.shape[1])
        reduced = PCA(n_components=pca_dimensions, random_state=42).fit_transform(representation)
        cluster = KMeans(n_clusters=len(shared_class_names), random_state=42, n_init=20).fit_predict(reduced)
        perplexity = min(30.0, max(5.0, (len(labels) - 1) / 3.0))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                    learning_rate="auto", random_state=42).fit_transform(reduced)
        latency_loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)),
                                    shuffle=False, num_workers=0)
        latency = _latency(model, next(iter(latency_loader)), device,
                           args.latency_warmup, args.latency_iterations)
        predictions = probabilities.argmax(axis=1)
        result = {
            "name": display_name,
            "checkpoint": str(checkpoint_path.resolve()),
            "model": saved_args["model"],
            "window_size": int(saved_args["window_size"]),
            "validation_samples": len(labels),
            "accuracy": float((predictions == labels).mean()),
            "nll": float(log_loss(labels, probabilities, labels=np.arange(len(shared_class_names)))),
            "brier": _brier(probabilities, labels, len(shared_class_names)),
            "ece": ece,
            "calibration_bins": calibration_bins,
            "cluster_ari": float(adjusted_rand_score(labels, cluster)),
            "cluster_nmi": float(normalized_mutual_info_score(labels, cluster)),
            "silhouette": float(silhouette_score(reduced, labels)),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            **latency,
            "tsne_points": tsne,
        }
        results.append(result)

    _plot_reliability(results, args.output_dir / "reliability_diagram.png")
    _plot_embeddings(results, shared_labels, shared_class_names,
                     args.output_dir / "embedding_clusters.png")
    serializable = []
    for result in results:
        copy = dict(result)
        copy.pop("tsne_points", None)
        serializable.append(copy)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(serializable, indent=2) + "\n")
    flat_keys = [key for key in serializable[0] if key != "calibration_bins"]
    with (args.output_dir / "diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in flat_keys} for row in serializable])
    print(json.dumps([{key: value for key, value in row.items() if key != "calibration_bins"}
                      for row in serializable], indent=2))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
