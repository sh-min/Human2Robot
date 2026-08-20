"""Encode every action description into a shared frozen CLIP prompt bank."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoProcessor, CLIPModel

from skill_classifier.action_semantics import load_action_semantics


PROMPT_TEMPLATES = (
    "{action}",
    "a video showing the kitchen action: {action}",
    "a person performing this kitchen manipulation: {action}",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_action_semantics(args.semantics)
    labels = list(config["action_labels"])
    prompts = {
        label: [
            template.format(action=config["actions"][label]["en"])
            for template in PROMPT_TEMPLATES
        ]
        for label in labels
    }
    flat_prompts = [prompt for label in labels for prompt in prompts[label]]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    inputs = processor(
        text=flat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
    # transformers 4.x returns a Tensor while newer releases may return a
    # BaseModelOutputWithPooling from the same convenience method.
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    features = torch.as_tensor(features).float()
    features = torch.nn.functional.normalize(features, dim=-1)
    features = features.reshape(len(labels), len(PROMPT_TEMPLATES), -1)
    prototypes = torch.nn.functional.normalize(features.mean(dim=1), dim=-1)

    semantic_bytes = args.semantics.read_bytes()
    payload = {
        "schema_version": 1,
        "kind": "frozen_action_text_prototypes",
        "model": args.model,
        "action_labels": labels,
        "prompts": prompts,
        "templates": list(PROMPT_TEMPLATES),
        "embeddings": prototypes.cpu(),
        "semantics": {
            "path": str(args.semantics.resolve()),
            "sha256": hashlib.sha256(semantic_bytes).hexdigest(),
        },
        "conditioning_contract": {
            "all_action_prompts_available_for_every_sample": True,
            "ground_truth_selected_prompt": False,
            "text_encoder_frozen": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"Saved {len(labels)} x {prototypes.shape[-1]} action text prototypes "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
