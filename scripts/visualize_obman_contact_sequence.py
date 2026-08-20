#!/usr/bin/env python3
"""Render a confidence-gated ObMan object-contact surface diagnostic video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def project(points, focal, width, height):
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-4)
    uv = np.full((len(points), 2), np.nan, dtype=np.float32)
    uv[valid, 0] = focal * points[valid, 0] / points[valid, 2] + width / 2
    uv[valid, 1] = focal * points[valid, 1] / points[valid, 2] + height / 2
    return uv, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_alignment_rmse_mm", type=float, default=15.0)
    parser.add_argument("--max_penetrating_vertices", type=int, default=100)
    args = parser.parse_args()

    sequence = np.load(args.sequence)
    report = json.loads(args.report.read_text())
    with np.load(args.hawor_npz) as hawor:
        focal = float(hawor["img_focal"])
    frame_indices = np.asarray(sequence["frame_indices"], dtype=np.int32)
    vertices = np.asarray(sequence["object_vertices"], dtype=np.float32)
    contacts = np.asarray(sequence["object_contact"], dtype=bool)
    reports = {item["frame_index"]: item for item in report["frames"]}

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if len(frame_indices) != frame_count:
        raise ValueError("sequence/video frame mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = 64
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width, height + header),
    )
    reliable_count = 0
    try:
        for sequence_index, frame_index in enumerate(frame_indices):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"video read failed at {frame_index}")
            item = reports[int(frame_index)]
            reliable = (
                item["alignment_rmse_mm"] <= args.max_alignment_rmse_mm
                and item["hand_penetrating_vertices"]
                <= args.max_penetrating_vertices
            )
            canvas = cv2.copyMakeBorder(
                frame, header, 0, 0, 0, cv2.BORDER_CONSTANT,
                value=(22, 22, 22),
            )
            if reliable:
                reliable_count += 1
                uv, valid = project(
                    vertices[sequence_index], focal, width, height
                )
                uv[:, 1] += header
                for point in uv[valid & ~contacts[sequence_index]][::3]:
                    cv2.circle(
                        canvas, tuple(np.rint(point).astype(int)),
                        1, (170, 170, 170), -1, cv2.LINE_AA,
                    )
                for point in uv[valid & contacts[sequence_index]]:
                    cv2.circle(
                        canvas, tuple(np.rint(point).astype(int)),
                        4, (0, 0, 255), -1, cv2.LINE_AA,
                    )
                label = "RELIABLE: object mesh gray | contact surface red"
                color = (80, 220, 80)
            else:
                label = (
                    "REJECTED: unreliable mesh alignment or penetration"
                )
                color = (0, 190, 255)
            cv2.putText(
                canvas, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        capture.release()
        writer.release()
    print(f"[ok] wrote {args.output}")
    print(f"[info] reliable frames={reliable_count}/{frame_count}")


if __name__ == "__main__":
    main()
