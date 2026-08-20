"""Open-vocabulary object boxes for the interaction-object spec.

``build_object_segments_from_contact.py`` finds the object by colour: the blob
inside the grasp box whose colour departs from the support surface. That fails
in two ways this scene produces regularly. The arm segmentation spills onto
whatever the hand is holding, which removes the held object from the candidate
blobs entirely and leaves the seed to land on the largest thing still visible --
a sponge on the bench. And "not the table" is not the same as "the object", so a
blob can run from the object into a stretch of wall or table the surrounding
ring failed to characterise.

Naming the objects removes both failures. GroundingDINO is told what is on the
table and returns a box per object per frame, and the box says where the object
ends whether or not the human mask covers part of it.

Each label is run in its own forward pass. Passing them together as one
period-separated prompt makes the model return merged spans -- "sponge green
snack box" -- because adjacent phrases share tokens, and a merged label cannot
be assigned to one object.

Detections are *not* filtered for correctness here. The white bin in this scene
answers to "clear plastic container" as readily as the real one does. Which
detection is the held object is decided downstream by the contact points, which
is the only evidence that says which object the hand is on; a false positive
somewhere else in the frame is never selected and costs nothing.

Output: JSON ``{"labels": [...], "frames": {"<t>": [{"label", "box", "score"}]}}``
with boxes as ``[x0, y0, x1, y1]`` in pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEFAULT_LABELS = ["mug", "sponge", "green snack box", "red snack box",
                  "clear plastic container"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--frame_glob", default="rgb_frame*.jpg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS,
                        help="One phrase per object on the table, lower case.")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--stride", type=int, default=5,
                        help="Detect every Nth frame. The seed only has to land "
                             "near a detected frame, and an object does not "
                             "move far in five frames at 30 fps.")
    parser.add_argument("--box_threshold", type=float, default=0.25,
                        help="Kept low on purpose: contact decides which box is "
                             "the held object, so recall matters here and "
                             "precision does not.")
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frames = sorted(args.frames_dir.glob(args.frame_glob))
    if not frames:
        raise SystemExit(f"no frames matching {args.frame_glob} in {args.frames_dir}")
    wanted = list(range(0, len(frames), max(1, args.stride)))

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model = model.to(args.device).eval()
    print(f"[gdino] {args.model} on {args.device}; {len(wanted)} of "
          f"{len(frames)} frames, {len(args.labels)} labels")

    detections: dict[str, list[dict]] = {}
    for done, index in enumerate(wanted):
        image = Image.open(frames[index]).convert("RGB")
        found = []
        for label in args.labels:
            inputs = processor(images=image, text=f"{label}.",
                               return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[image.size[::-1]])[0]
            for box, score in zip(result["boxes"], result["scores"]):
                found.append({"label": label,
                              "box": [int(round(v)) for v in box.tolist()],
                              "score": round(float(score), 4)})
        detections[str(index)] = found
        if done % 20 == 0:
            print(f"[gdino] f{index}: {len(found)} boxes", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"model": args.model, "labels": args.labels, "stride": args.stride,
         "frame_count": len(frames), "frames": detections}, indent=1))
    total = sum(len(v) for v in detections.values())
    print(f"[ok] {args.output}  frames={len(detections)} boxes={total}")


if __name__ == "__main__":
    main()
