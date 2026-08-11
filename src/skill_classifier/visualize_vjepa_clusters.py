"""Visualize whether V-JEPA skill embeddings form coherent groups.

The report compares two representations of every labelled token:

* the frozen V-JEPA window representation used as classifier input;
* the penultimate representation learned by the selected MLP checkpoint.

It saves a PCA/t-SNE/K-Means dashboard, representative source frames, a CSV
with every plotted point, machine-readable metrics, and a small HTML index.
The unsupervised metrics are diagnostics, not a claim that t-SNE geometry is
quantitatively meaningful.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import normalize

from skill_classifier.models import build_model


def _patterns(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _feature_paths(root: Path, glob_string: str) -> list[Path]:
    return sorted(
        {
            path
            for pattern in _patterns(glob_string)
            for path in root.glob(f"{pattern}/features.pt")
        }
    )


def _window_representation(features: np.ndarray, index: int, size: int, pool: str) -> np.ndarray:
    start = index - size + 1
    if start >= 0:
        window = features[start : index + 1]
    else:
        window = np.concatenate(
            [np.zeros((-start, features.shape[1]), dtype=features.dtype), features[: index + 1]],
            axis=0,
        )
    return window.reshape(-1) if pool == "concat" else window.mean(axis=0)


def _source_path(bundle: dict, bundle_path: Path) -> str:
    signature = bundle.get("input_provenance", {}).get("rgb") or {}
    value = signature.get("path")
    if value:
        return str(value)
    return str((bundle_path.parent / "rgb").resolve())


def load_points(config: dict) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
    train_root = Path(config["train_data_root"])
    val_root = Path(config["val_data_root"])
    train_paths = _feature_paths(train_root, config.get("train_recording_glob", "*"))
    val_paths = _feature_paths(val_root, config.get("val_recording_glob", "*"))
    overlap = set(train_paths) & set(val_paths)
    if overlap:
        raise ValueError(f"train/validation feature overlap: {sorted(map(str, overlap))[:3]}")
    if not train_paths or not val_paths:
        raise ValueError(
            f"empty split: train={len(train_paths)} validation={len(val_paths)}"
        )

    requested_labels = list(config["action_labels"])
    variant = config.get("variant", "vjepa_orig")
    feature_key = {
        "vjepa_orig": "vjepa_orig",
        "masked_vjepa_orig": "vjepa_orig_masked",
        "vjepa_robot": "vjepa_robot",
    }.get(variant)
    if feature_key is None:
        raise ValueError(f"clustering requires a V-JEPA variant, got {variant!r}")

    window_size = int(config.get("window_size", 8))
    pool = config.get("pool", "mean")
    representations: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict] = []

    for split, paths in (("train", train_paths), ("validation", val_paths)):
        for path in paths:
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            bundled_labels = list(bundle.get("action_labels", requested_labels))
            if bundled_labels != requested_labels:
                raise ValueError(
                    f"{path}: labels {bundled_labels} != configured {requested_labels}"
                )
            if feature_key not in bundle:
                raise ValueError(f"{path}: missing {feature_key}")
            features = torch.as_tensor(bundle[feature_key]).float().numpy()
            if config.get("vjepa_diff", False):
                diff = np.zeros_like(features)
                diff[1:] = features[1:] - features[:-1]
                features = diff
            token_labels = torch.as_tensor(bundle["labels_per_token"]).numpy()
            centers = torch.as_tensor(
                bundle.get("token_center_frame_indices", torch.arange(len(features)))
            ).numpy()
            if len(features) != len(token_labels) or len(features) != len(centers):
                raise ValueError(f"{path}: token-alignment length mismatch")
            recording = str(bundle.get("recording", path.parent.name))
            source = _source_path(bundle, path)
            for token_index, label in enumerate(token_labels):
                label = int(label)
                if label < 0:
                    continue
                if label >= len(requested_labels):
                    raise ValueError(f"{path}: out-of-range label {label}")
                representations.append(
                    _window_representation(features, token_index, window_size, pool)
                )
                labels.append(label)
                rows.append(
                    {
                        "split": split,
                        "recording": recording,
                        "feature_bundle": str(path.resolve()),
                        "source": source,
                        "token_index": token_index,
                        "source_frame": int(centers[token_index]),
                        "label_index": label,
                        "label": requested_labels[label],
                    }
                )

    return (
        np.stack(representations).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        rows,
        requested_labels,
    )


def classifier_representation(
    raw: np.ndarray, checkpoint_path: Path, class_names: list[str]
) -> tuple[np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = dict(checkpoint["args"])
    if args.get("model") != "mlp":
        raise ValueError("classifier-adapted clustering currently requires an MLP checkpoint")
    model = build_model(
        "mlp",
        vjepa_dim=int(args["vjepa_dim"]),
        hand_dim=int(args["hand_dim"]),
        window_size=int(args["window_size"]),
        num_classes=len(args.get("active_labels", class_names)),
        hidden_dims=tuple(args["hidden_dims"]),
        dropout=float(args["dropout"]),
        pool=args.get("pool", "mean"),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if model.hand_dim != 0:
        raise ValueError("this report is intentionally V-JEPA-only; checkpoint has hand features")
    adapted = []
    with torch.inference_mode():
        values = torch.from_numpy(raw)
        for start in range(0, len(values), 1024):
            adapted.append(model.net[:-1](values[start : start + 1024]).numpy())
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": int(checkpoint["epoch"]),
        "validation_accuracy": float(checkpoint["val_acc"]),
        "adapted_dimension": int(adapted[0].shape[1]),
    }
    return np.concatenate(adapted), metadata


def reduce_features(values: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit = normalize(values, norm="l2")
    components = max(2, min(50, len(unit) - 1, unit.shape[1]))
    pca = PCA(n_components=components, random_state=seed)
    compact = pca.fit_transform(unit)
    pca_2d = compact[:, :2]
    if len(unit) < 8:
        return compact, pca_2d, pca_2d.copy()
    perplexity = min(40.0, max(5.0, (len(unit) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(compact)
    return compact, pca_2d, tsne


def contingency(clusters: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    table = np.zeros((count, count), dtype=np.int64)
    for cluster, label in zip(clusters, labels):
        table[int(cluster), int(label)] += 1
    return table


def _silhouette(values: np.ndarray, assignments: np.ndarray, seed: int) -> float | None:
    if len(np.unique(assignments)) < 2 or len(values) <= len(np.unique(assignments)):
        return None
    sample_size = min(5000, len(values))
    return float(
        silhouette_score(
            values,
            assignments,
            sample_size=sample_size if sample_size < len(values) else None,
            random_state=seed,
        )
    )


def cluster_metrics(
    compact: np.ndarray, labels: np.ndarray, clusters: np.ndarray, table: np.ndarray, seed: int
) -> dict:
    return {
        "label_silhouette": _silhouette(compact, labels, seed),
        "kmeans_silhouette": _silhouette(compact, clusters, seed),
        "adjusted_rand_index": float(adjusted_rand_score(labels, clusters)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(labels, clusters)
        ),
        "homogeneity": float(homogeneity_score(labels, clusters)),
        "completeness": float(completeness_score(labels, clusters)),
        "v_measure": float(v_measure_score(labels, clusters)),
        "cluster_purity": float(table.max(axis=1).sum() / table.sum()),
    }


def class_centroid_similarity(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    unit = normalize(values, norm="l2")
    centroids = np.stack([unit[labels == index].mean(axis=0) for index in range(count)])
    centroids = normalize(centroids, norm="l2")
    return centroids @ centroids.T


def _scatter_labels(ax, points, labels, splits, names, palette, title):
    for index, name in enumerate(names):
        mask = labels == index
        ax.scatter(
            points[mask, 0], points[mask, 1], s=14, alpha=0.56,
            color=palette[index], label=name, linewidths=0,
        )
    val = splits == "validation"
    ax.scatter(
        points[val, 0], points[val, 1], s=30, facecolors="none",
        edgecolors="#111111", linewidths=0.65, label="validation",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def dashboard(
    output: Path,
    raw_pca: np.ndarray,
    raw_tsne: np.ndarray,
    adapted_tsne: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    clusters: np.ndarray,
    names: list[str],
    similarity: np.ndarray,
    table: np.ndarray,
    raw_metrics: dict,
    adapted_metrics: dict,
    metadata: dict,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", len(names))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    _scatter_labels(
        axes[0, 0], raw_pca, labels, splits, names, palette,
        "Frozen V-JEPA — PCA",
    )
    _scatter_labels(
        axes[0, 1], raw_tsne, labels, splits, names, palette,
        "Frozen V-JEPA — t-SNE",
    )
    _scatter_labels(
        axes[0, 2], adapted_tsne, labels, splits, names, palette,
        "Classifier-adapted embedding — t-SNE",
    )
    axes[0, 2].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    cluster_palette = sns.color_palette("husl", len(names))
    for cluster in range(len(names)):
        mask = clusters == cluster
        axes[1, 0].scatter(
            adapted_tsne[mask, 0], adapted_tsne[mask, 1], s=14,
            alpha=0.6, color=cluster_palette[cluster], label=f"cluster {cluster}",
            linewidths=0,
        )
    axes[1, 0].set_title("Unsupervised K-Means on adapted embedding", fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    sns.heatmap(
        similarity, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f",
        xticklabels=names, yticklabels=names, square=True, ax=axes[1, 1],
        cbar_kws={"shrink": 0.75},
    )
    axes[1, 1].set_title("Adapted class-centroid cosine similarity", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=35)

    row_sum = np.maximum(table.sum(axis=1, keepdims=True), 1)
    sns.heatmap(
        table / row_sum, vmin=0, vmax=1, cmap="Blues", annot=table, fmt="d",
        xticklabels=names, yticklabels=[f"C{i}" for i in range(len(names))],
        ax=axes[1, 2], cbar_kws={"label": "row fraction", "shrink": 0.75},
    )
    axes[1, 2].set_title("K-Means cluster × ground-truth label", fontweight="bold")
    axes[1, 2].set_xlabel("Ground truth")
    axes[1, 2].set_ylabel("Cluster")
    axes[1, 2].tick_params(axis="x", rotation=35)

    def fmt(value):
        return "n/a" if value is None else f"{value:.3f}"

    metrics_text = (
        f"Points: {len(labels):,}   train: {(splits == 'train').sum():,}   "
        f"validation: {(splits == 'validation').sum():,}\n"
        f"Selected checkpoint: epoch {metadata['epoch']}   "
        f"validation accuracy: {metadata['validation_accuracy']:.1%}\n\n"
        f"Frozen label silhouette:   {fmt(raw_metrics['label_silhouette'])}\n"
        f"Adapted label silhouette:  {fmt(adapted_metrics['label_silhouette'])}\n"
        f"Adapted K-Means silhouette:{fmt(adapted_metrics['kmeans_silhouette'])}\n"
        f"Cluster purity:             {adapted_metrics['cluster_purity']:.3f}\n"
        f"NMI / ARI:                  "
        f"{adapted_metrics['normalized_mutual_information']:.3f} / "
        f"{adapted_metrics['adjusted_rand_index']:.3f}"
    )
    fig.text(
        0.015, 0.012, metrics_text, family="monospace", fontsize=10,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f4f7fa", "edgecolor": "#789"},
    )
    fig.suptitle(
        "Robot-overlay V-JEPA skill embedding and clustering report",
        fontsize=21, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def source_frame(path_string: str, frame_index: int) -> Image.Image | None:
    path = Path(path_string)
    if path.is_dir():
        images = sorted(
            item for item in path.iterdir()
            if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not images:
            return None
        frame_index = min(max(frame_index, 0), len(images) - 1)
        return Image.open(images[frame_index]).convert("RGB")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _fit_tile(image: Image.Image | None, size: tuple[int, int]) -> Image.Image:
    if image is None:
        return Image.new("RGB", size, "#30343b")
    width, height = image.size
    scale = max(size[0] / width, size[1] / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def representative_sheet(
    output: Path,
    adapted: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    rows: list[dict],
    names: list[str],
    table: np.ndarray,
) -> None:
    unit = normalize(adapted, norm="l2")

    def nearest(mask: np.ndarray) -> int:
        candidates = np.flatnonzero(mask)
        center = normalize(unit[candidates].mean(axis=0, keepdims=True))[0]
        return int(candidates[np.argmax(unit[candidates] @ center)])

    label_indices = [nearest(labels == index) for index in range(len(names))]
    cluster_indices = [nearest(clusters == index) for index in range(len(names))]
    tile = (240, 150)
    header = 34
    caption = 48
    margin = 16
    columns = len(names)
    width = margin * 2 + columns * tile[0]
    height = margin * 3 + 2 * (header + tile[1] + caption)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = _load_font(18, bold=True)
    text_font = _load_font(13)
    small_font = _load_font(11)

    sections = [
        ("Nearest sample to each ground-truth class centroid", label_indices, False),
        ("Nearest sample to each unsupervised K-Means centroid", cluster_indices, True),
    ]
    y = margin
    for section_title, indices, is_cluster in sections:
        draw.text((margin, y), section_title, fill="#111827", font=title_font)
        y += header
        for column, point_index in enumerate(indices):
            row = rows[point_index]
            image = _fit_tile(
                source_frame(row["source"], int(row["source_frame"])), tile
            )
            x = margin + column * tile[0]
            sheet.paste(image, (x, y))
            if is_cluster:
                majority = names[int(np.argmax(table[column]))]
                heading = f"Cluster {column} → {majority}"
            else:
                heading = names[column]
            draw.rectangle((x, y + tile[1], x + tile[0], y + tile[1] + caption), fill="#f3f4f6")
            draw.text((x + 6, y + tile[1] + 4), heading, fill="#111827", font=text_font)
            draw.text(
                (x + 6, y + tile[1] + 24),
                f"{row['recording']}  f{row['source_frame']}  {row['split']}",
                fill="#4b5563", font=small_font,
            )
        y += tile[1] + caption + margin
    sheet.save(output)


def write_csv(
    output: Path,
    rows: list[dict],
    clusters: np.ndarray,
    raw_pca: np.ndarray,
    raw_tsne: np.ndarray,
    adapted_pca: np.ndarray,
    adapted_tsne: np.ndarray,
) -> None:
    fieldnames = list(rows[0]) + [
        "cluster", "raw_pca_x", "raw_pca_y", "raw_tsne_x", "raw_tsne_y",
        "adapted_pca_x", "adapted_pca_y", "adapted_tsne_x", "adapted_tsne_y",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            values = dict(row)
            values.update(
                {
                    "cluster": int(clusters[index]),
                    "raw_pca_x": float(raw_pca[index, 0]),
                    "raw_pca_y": float(raw_pca[index, 1]),
                    "raw_tsne_x": float(raw_tsne[index, 0]),
                    "raw_tsne_y": float(raw_tsne[index, 1]),
                    "adapted_pca_x": float(adapted_pca[index, 0]),
                    "adapted_pca_y": float(adapted_pca[index, 1]),
                    "adapted_tsne_x": float(adapted_tsne[index, 0]),
                    "adapted_tsne_y": float(adapted_tsne[index, 1]),
                }
            )
            writer.writerow(values)


def write_html(output: Path, summary: dict) -> None:
    metrics = summary["adapted_metrics"]
    cluster_rows = "".join(
        "<tr>"
        f"<td>{cluster['cluster']}</td><td>{html.escape(cluster['majority_label'])}</td>"
        f"<td>{cluster['majority_fraction']:.1%}</td><td>{cluster['size']}</td>"
        "</tr>"
        for cluster in summary["clusters"]
    )
    body = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>V-JEPA clustering report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#172033}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}} .card{{background:#f3f6fa;border:1px solid #d7e0ea;border-radius:10px;padding:12px 18px}}
img{{max-width:100%;border:1px solid #d7dce3;border-radius:8px}} table{{border-collapse:collapse}} th,td{{border:1px solid #d7dce3;padding:7px 12px;text-align:left}}
.note{{background:#fff8df;border-left:4px solid #e7b33e;padding:10px 14px}} a{{color:#175cd3}}
</style></head><body>
<h1>Robot-overlay V-JEPA 군집 시각화</h1>
<div class="cards">
<div class="card">표본 <strong>{summary['points']:,}</strong></div>
<div class="card">클러스터 순도 <strong>{metrics['cluster_purity']:.3f}</strong></div>
<div class="card">NMI <strong>{metrics['normalized_mutual_information']:.3f}</strong></div>
<div class="card">ARI <strong>{metrics['adjusted_rand_index']:.3f}</strong></div>
<div class="card">검증 정확도 <strong>{summary['checkpoint']['validation_accuracy']:.1%}</strong></div>
</div>
<h2>임베딩 비교</h2><img src="vjepa_clustering_dashboard.png" alt="embedding dashboard">
<h2>군집 대표 프레임</h2><img src="representative_tokens.png" alt="representative frames">
<h2>클러스터 구성</h2><table><thead><tr><th>클러스터</th><th>다수 라벨</th><th>비율</th><th>표본 수</th></tr></thead><tbody>{cluster_rows}</tbody></table>
<p class="note">t-SNE는 보기 위한 비선형 투영입니다. 실제 분리 정도는 silhouette, NMI, ARI, 순도와 검증 confusion matrix를 함께 봐야 합니다.</p>
<p><a href="cluster_assignments.csv">전체 좌표 CSV</a> · <a href="clustering_summary.json">요약 JSON</a> · <a href="../performance_dashboard.png">학습 성능</a> · <a href="../confusion_matrix_best.png">confusion matrix</a></p>
</body></html>"""
    output.write_text(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = experiment_dir / f"best_{config.get('model', 'mlp')}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    raw, labels, rows, class_names = load_points(config)
    adapted, checkpoint_metadata = classifier_representation(
        raw, checkpoint_path, class_names
    )
    raw_compact, raw_pca, raw_tsne = reduce_features(raw, args.seed)
    adapted_compact, adapted_pca, adapted_tsne = reduce_features(adapted, args.seed)
    kmeans = KMeans(n_clusters=len(class_names), n_init=20, random_state=args.seed)
    clusters = kmeans.fit_predict(adapted_compact)
    table = contingency(clusters, labels, len(class_names))
    raw_clusters = KMeans(
        n_clusters=len(class_names), n_init=20, random_state=args.seed
    ).fit_predict(raw_compact)
    raw_table = contingency(raw_clusters, labels, len(class_names))
    raw_metrics = cluster_metrics(
        raw_compact, labels, raw_clusters, raw_table, args.seed
    )
    adapted_metrics = cluster_metrics(
        adapted_compact, labels, clusters, table, args.seed
    )
    splits = np.asarray([row["split"] for row in rows])
    similarity = class_centroid_similarity(adapted, labels, len(class_names))

    dashboard(
        output_dir / "vjepa_clustering_dashboard.png",
        raw_pca, raw_tsne, adapted_tsne, labels, splits, clusters, class_names,
        similarity, table, raw_metrics, adapted_metrics, checkpoint_metadata,
    )
    representative_sheet(
        output_dir / "representative_tokens.png",
        adapted, labels, clusters, rows, class_names, table,
    )
    write_csv(
        output_dir / "cluster_assignments.csv", rows, clusters,
        raw_pca, raw_tsne, adapted_pca, adapted_tsne,
    )

    cluster_summaries = []
    for cluster in range(len(class_names)):
        majority = int(np.argmax(table[cluster]))
        size = int(table[cluster].sum())
        cluster_summaries.append(
            {
                "cluster": cluster,
                "size": size,
                "majority_label": class_names[majority],
                "majority_fraction": float(table[cluster, majority] / max(size, 1)),
                "label_counts": {
                    name: int(table[cluster, index])
                    for index, name in enumerate(class_names)
                },
            }
        )
    summary = {
        "schema_version": 1,
        "config": str(config_path.resolve()),
        "points": len(rows),
        "recordings": len({row["feature_bundle"] for row in rows}),
        "splits": {
            "train": int((splits == "train").sum()),
            "validation": int((splits == "validation").sum()),
        },
        "class_names": class_names,
        "class_counts": {
            name: int((labels == index).sum())
            for index, name in enumerate(class_names)
        },
        "checkpoint": checkpoint_metadata,
        "raw_metrics": raw_metrics,
        "adapted_metrics": adapted_metrics,
        "clusters": cluster_summaries,
    }
    (output_dir / "clustering_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    write_html(output_dir / "index.html", summary)
    print(f"[ok] clustering dashboard: {(output_dir / 'vjepa_clustering_dashboard.png').resolve()}")
    print(f"[ok] clustering report:    {(output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
