#!/usr/bin/env python3
"""Refine a coarse SAM2 object mask against same-frame RGB boundaries.

The output is a *modal* matte: only object pixels visible in the current raw
frame are selected.  A coarse SAM2 mask supplies the object seed, GrabCut
snaps it to RGB edges, and the human mask prevents expansion into visible skin
or sleeves.  No temporal RGB or generated pixels are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def component_attached_to_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), 8)
    if count <= 1:
        return candidate
    best_label = 0
    best_overlap = 0
    for label in range(1, count):
        overlap = int(((labels == label) & seed).sum())
        if overlap > best_overlap:
            best_label = label
            best_overlap = overlap
    return labels == best_label if best_label else seed.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--coarse-mask", type=Path, required=True)
    parser.add_argument("--human-mask", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-video", type=Path, default=None)
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--sure-fg-erode", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.search_radius <= 0 or args.sure_fg_erode < 0:
        parser.error("search radius must be positive and erosion non-negative")

    coarse_source = np.load(args.coarse_mask, mmap_mode="r")
    human_source = (
        np.load(args.human_mask, mmap_mode="r")
        if args.human_mask is not None
        else None
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    frame_count = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(coarse_source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    refined = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=bool, shape=(frame_count, height, width)
    )

    debug_path = args.debug_video or args.output.with_name("video_refined_object_matte.mp4")
    writer = cv2.VideoWriter(
        str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open debug writer: {debug_path}")

    search_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.search_radius + 1, 2 * args.search_radius + 1),
    )
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.sure_fg_erode + 1, 2 * args.sure_fg_erode + 1),
    )
    coarse_total = 0
    refined_total = 0
    added_total = 0
    removed_total = 0
    try:
        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video read failed at frame {index}")
            coarse = resize_mask(coarse_source[index], width, height)
            if not coarse.any():
                refined[index] = False
                writer.write(frame)
                continue
            search = cv2.dilate(coarse.astype(np.uint8), search_kernel).astype(bool)
            sure_fg = (
                cv2.erode(coarse.astype(np.uint8), erode_kernel).astype(bool)
                if args.sure_fg_erode > 0
                else coarse.copy()
            )
            if not sure_fg.any():
                sure_fg = coarse.copy()

            gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
            gc_mask[search] = cv2.GC_PR_BGD
            gc_mask[coarse] = cv2.GC_PR_FGD
            gc_mask[sure_fg] = cv2.GC_FGD
            if human_source is not None:
                human = resize_mask(human_source[index], width, height)
                # The coarse object seed wins where SAM2 already sees object;
                # only expansion into human-only pixels is forbidden.
                gc_mask[human & ~coarse] = cv2.GC_BGD

            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                frame,
                gc_mask,
                None,
                bg_model,
                fg_model,
                args.iterations,
                cv2.GC_INIT_WITH_MASK,
            )
            candidate = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)) & search
            candidate = component_attached_to_seed(candidate, sure_fg)
            refined[index] = candidate

            coarse_total += int(coarse.sum())
            refined_total += int(candidate.sum())
            added = candidate & ~coarse
            removed = coarse & ~candidate
            added_total += int(added.sum())
            removed_total += int(removed.sum())

            debug = frame.copy()
            contours, _ = cv2.findContours(
                candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(debug, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
            debug[added] = (
                0.55 * debug[added].astype(np.float32)
                + 0.45 * np.array([0, 255, 255], dtype=np.float32)
            ).astype(np.uint8)
            debug[removed] = (
                0.55 * debug[removed].astype(np.float32)
                + 0.45 * np.array([0, 0, 255], dtype=np.float32)
            ).astype(np.uint8)
            writer.write(debug)
    finally:
        capture.release()
        writer.release()
        refined.flush()

    report = {
        "schema_version": 1,
        "method": "same-frame RGB GrabCut refinement from SAM2 modal seed",
        "frame_count": frame_count,
        "coarse_pixels_total": coarse_total,
        "refined_pixels_total": refined_total,
        "added_visible_edge_pixels_total": added_total,
        "removed_nonobject_pixels_total": removed_total,
        "search_radius": args.search_radius,
        "sure_fg_erode": args.sure_fg_erode,
        "iterations": args.iterations,
        "temporal_rgb_used": False,
        "generated_pixels_used": False,
    }
    report_path = args.output.with_name("refined_object_matte_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[ok] wrote {args.output}", flush=True)
    print(f"[ok] wrote {debug_path}", flush=True)


if __name__ == "__main__":
    main()
