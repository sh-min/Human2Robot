"""Visualize learned V-JEPA patch attention on representative validation frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data_preprocess.preprocess import VJEPA2_EVAL_CROP, preprocess_rgb_frame
from skill_classifier.models import build_model
from skill_classifier.skill_dataset import SkillWindowDataset, load_recordings


def load_rgb(source: Path, frame_index: int) -> np.ndarray:
    if source.is_dir():
        paths = sorted(
            path for path in source.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        bgr = cv2.imread(str(paths[frame_index]), cv2.IMREAD_COLOR)
    else:
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"cannot decode {source} frame {frame_index}")
    if bgr is None:
        raise RuntimeError(f"cannot read {source} frame {frame_index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    checkpoint = torch.load(
        args.experiment_dir / "best_spatial_attention_mlp.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_args = checkpoint["args"]
    class_names = list(model_args["active_labels"])
    model = build_model(
        "spatial_attention_mlp",
        vjepa_dim=int(model_args["vjepa_dim"]),
        hand_dim=int(model_args["hand_dim"]),
        window_size=int(model_args["window_size"]),
        num_classes=len(class_names),
        hidden_dims=tuple(model_args["hidden_dims"]),
        dropout=float(model_args["dropout"]),
    ).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    recordings = load_recordings(
        config["val_data_root"], config["val_recording_glob"]
    )
    dataset = SkillWindowDataset(
        recordings,
        window_size=int(config["window_size"]),
        variant=config["variant"],
        vjepa_diff=bool(config.get("vjepa_diff", False)),
        hand_representation=config["hand_representation"],
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    probabilities = []
    predictions = []
    with torch.inference_mode():
        for dense, hand, _ in loader:
            logits = model(dense.cuda(), hand.cuda())
            probabilities.append(torch.softmax(logits, dim=-1).cpu())
            predictions.append(logits.argmax(dim=-1).cpu())
    probabilities = torch.cat(probabilities)
    predictions = torch.cat(predictions)

    selected = []
    for class_index, class_name in enumerate(class_names):
        candidates = [
            index for index, sample in enumerate(dataset.samples)
            if sample[2] == class_index
        ]
        correct = [index for index in candidates if int(predictions[index]) == class_index]
        pool = correct or candidates
        chosen = max(pool, key=lambda index: float(probabilities[index, class_index]))
        rec_index, token_index, label = dataset.samples[chosen]
        dense, hand, _ = dataset[chosen]
        with torch.inference_mode():
            _, weights = model.representation(dense[None].cuda())
        patch_weights = weights[0, -1].cpu().numpy()
        side = round(math.sqrt(len(patch_weights)))
        if side * side != len(patch_weights):
            raise ValueError("attention patches are not a square grid")
        heatmap = patch_weights.reshape(side, side)
        entropy = -float(np.sum(patch_weights * np.log(patch_weights + 1.0e-12)))
        entropy /= math.log(len(patch_weights))
        bundle = recordings[rec_index]
        source = Path(bundle["input_provenance"]["rgb"]["path"])
        source_frame = int(bundle["token_center_frame_indices"][token_index])
        rgb = preprocess_rgb_frame(
            load_rgb(source, source_frame), 384, VJEPA2_EVAL_CROP
        )
        selected.append(
            {
                "class_name": class_name,
                "class_index": class_index,
                "dataset_index": chosen,
                "recording": bundle["recording"],
                "token_index": token_index,
                "source_frame": source_frame,
                "source": str(source),
                "prediction": class_names[int(predictions[chosen])],
                "confidence": float(probabilities[chosen, class_index]),
                "normalized_attention_entropy": entropy,
                "rgb": rgb,
                "heatmap": heatmap,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(class_names), figsize=(4 * len(class_names), 8))
    for column, item in enumerate(selected):
        axes[0, column].imshow(item["rgb"])
        axes[0, column].set_title(
            f"{item['class_name']}\n{item['recording']} · frame {item['source_frame']}",
            fontsize=10,
        )
        axes[0, column].axis("off")
        axes[1, column].imshow(item["rgb"])
        axes[1, column].imshow(
            item["heatmap"],
            cmap="turbo",
            alpha=0.55,
            extent=(0, 384, 384, 0),
            interpolation="bilinear",
        )
        axes[1, column].set_title(
            f"attention · p={item['confidence']:.2f}\n"
            f"entropy={item['normalized_attention_entropy']:.3f}",
            fontsize=10,
        )
        axes[1, column].axis("off")
    fig.suptitle(
        "V-JEPA 2.1 learned spatial attention\n"
        "Top: model input crop · Bottom: attention for the current temporal token",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    image_path = args.output_dir / "representative_attention.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = []
    for item in selected:
        metadata.append({key: value for key, value in item.items() if key not in {"rgb", "heatmap"}})
    (args.output_dir / "representative_attention.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"[ok] attention visualization: {image_path.resolve()}")


if __name__ == "__main__":
    main()
