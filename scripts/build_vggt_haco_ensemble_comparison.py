#!/usr/bin/env python3
"""Render 2.5D, VGGT-only, dual-HaCo-only, and fused ensemble comparisons."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "8-5/mesh_sota_pilot/episode_1"
DATASET = ROOT / "data/kitchen_dataset/26.08.05_stereo_calibrated/1"
PROC = DATASET / "camera_2/inpainting/processed/view/0"
OUTPUT = PILOT / "all_objects_vggt_haco_ensemble_comparison"
sys.path.insert(0, str(ROOT / "scripts"))
from build_all_objects_full_comparison import label_for_frame, load_tracks  # noqa: E402
from compare_spar3d_xhand_occlusion_pilot import (  # noqa: E402
    _remap_image,
    _remap_labels,
    _remap_mask,
    build_static_occlusion_masks,
    undistortion_maps,
    weighted_remap_depth,
)
from compare_tracked_mesh_xhand_video import DepthPairRenderer, composite, title_panel  # noqa: E402
sys.path.insert(0, str(ROOT / "src/inpainting"))
from composite_xhand_object_barrier import resize_overlay_frame, restore_raw_object_pixels, semantic_hand_labels  # noqa: E402


class FingerEnsemble:
    """Finger-wise confidence fusion with EMA, hysteresis, and bounded filling."""

    def __init__(self) -> None:
        self.ema = np.zeros(5, np.float32)
        self.active = np.zeros(5, bool)

    def reset(self) -> None:
        self.ema.fill(0)
        self.active.fill(False)

    def fuse(
        self,
        *,
        hand: np.ndarray,
        finger_labels: np.ndarray,
        geometry: np.ndarray,
        haco: np.ndarray,
        contact_scores: np.ndarray,
        object_support: np.ndarray,
        mesh_support: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float | bool]]]:
        labels = semantic_hand_labels(hand, finger_labels)
        geometry = np.asarray(geometry, bool) & hand
        haco = np.asarray(haco, bool) & hand
        result = geometry & (labels == 6)  # Palm has no HaCo finger score.
        eligible_near_object = cv2.dilate(
            (object_support | mesh_support).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        ).astype(bool)
        records: list[dict[str, float | bool]] = []
        for index, label in enumerate(range(1, 6)):
            region = labels == label
            area = max(int(region.sum()), 1)
            g = geometry & region
            h = haco & region
            g_ratio = float(g.sum()) / area
            h_ratio = float(h.sum()) / area
            g_conf = min(1.0, g_ratio / 0.18)
            score_conf = float(np.clip((float(contact_scores[index]) - 0.68) / 0.27, 0, 1))
            # HaCo produces a narrow contact-connected subset, so its support
            # is normalized at a smaller finger-area fraction than the dense
            # VGGT volume.  Either branch can rescue the other; local agreement
            # adds confidence without being mandatory.
            h_conf = score_conf * min(1.0, h_ratio / 0.025)
            consensus = (g & cv2.dilate(h.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).any()
            fused = min(1.0, 0.46 * g_conf + 0.46 * h_conf + 0.18 * float(consensus))
            self.ema[index] = 0.65 * fused + 0.35 * self.ema[index]
            threshold = 0.28 if self.active[index] else 0.40
            self.active[index] = bool(self.ema[index] >= threshold)
            seed = (g | h) & region
            if self.active[index] and seed.any():
                closed = cv2.morphologyEx(
                    seed.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                ).astype(bool)
                result |= (closed & region & eligible_near_object) | seed
            records.append({
                "geometry_ratio": g_ratio,
                "haco_ratio": h_ratio,
                "haco_score": float(contact_scores[index]),
                "fused_confidence_ema": float(self.ema[index]),
                "active": bool(self.active[index]),
            })
        return result & hand, records


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotation = json.loads((ROOT / "8-5/data/annotations/1/gt_labels.json").read_text())
    stereo = json.loads((DATASET / "stereo_manifest.json").read_text())
    tracks = load_tracks("vggt_omega")
    mh_cal = stereo["calibration"]["intrinsics_by_view"]["MH"]
    kmh = np.asarray(mh_cal["camera_matrix"], float)
    mh_x, mh_y = undistortion_maps(kmh, np.asarray(mh_cal["distortion_k1_k2_p1_p2_k3"]), width=1280, height=720)

    overlay = PROC / "overlay_processor"
    robot_rgb_all = np.load(overlay / "robot_rgb.npy", mmap_mode="r")
    robot_depth_all = np.load(overlay / "robot_depth.npy", mmap_mode="r")
    robot_mask_all = np.load(overlay / "robot_mask.npy", mmap_mode="r")
    hand_mask_all = np.load(overlay / "robot_hand_mask.npy", mmap_mode="r")
    labels_all = np.load(overlay / "robot_finger_labels.npy", mmap_mode="r")
    support_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_amodal.npy", mmap_mode="r")
    restore_all = np.load(PROC / "object_completion_dual_haco_e2fgvi/object_mask_observed_clean.npy", mmap_mode="r")
    current_all = np.load(PROC / "overlay_best_inpaint_barrier/occluded_hand_mask.npy", mmap_mode="r")
    haco_all = np.load(PROC / "overlay_haco_dual/occluded_finger_mask.npy", mmap_mode="r")
    haco_report = json.loads((PROC / "overlay_haco_dual/report.json").read_text())
    contact_scores = np.asarray(haco_report["contact_score_fused"], np.float32)
    if contact_scores.shape != (annotation["num_frames"], 5):
        raise ValueError(f"unexpected fused HaCo score shape {contact_scores.shape}")

    background = cv2.VideoCapture(str(PROC / "object_completion_dual_haco_e2fgvi/video_object_completed.mp4"))
    output_path = OUTPUT / "episode1_vggt_omega_dual_haco_finger_ensemble_comparison.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 832))
    if not writer.isOpened():
        raise RuntimeError("failed to create ensemble comparison")
    renderer = None
    renderer_label = None
    fusion = FingerEnsemble()
    stats = {label: {"frames": 0, "vggt_px": 0, "haco_px": 0, "ensemble_px": 0, "active_frame_fingers": 0} for label in tracks}
    transition_frames = 0
    try:
        for frame in range(annotation["num_frames"]):
            label = label_for_frame(frame, annotation["segments"])
            raw = cv2.imread(str(DATASET / f"camera_2/rgb/rgb_frame{frame:06d}.jpg"))
            raw_u = _remap_image(raw, mh_x, mh_y)
            background.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bg = background.read()
            if not ok:
                raise RuntimeError(f"failed background frame {frame}")
            base = restore_raw_object_pixels(_remap_image(bg, mh_x, mh_y), raw_u, _remap_mask(restore_all[frame], mh_x, mh_y))
            rr, rd, rm, hm, fl = resize_overlay_frame(robot_rgb_all[frame], robot_depth_all[frame], robot_mask_all[frame], hand_mask_all[frame], labels_all[frame], width=1280, height=720)
            rr = _remap_image(rr, mh_x, mh_y)
            rd = weighted_remap_depth(rd, mh_x, mh_y)
            rm = _remap_mask(rm, mh_x, mh_y)
            hm = _remap_mask(hm, mh_x, mh_y)
            fl = _remap_labels(fl, mh_x, mh_y)
            current = _remap_mask(current_all[frame], mh_x, mh_y) & hm
            haco = _remap_mask(haco_all[frame], mh_x, mh_y) & hm
            existing = composite(base, rr, rm, hm, current)
            haco_view = composite(base, rr, rm, hm, haco)

            if label in tracks:
                track = tracks[label]
                local = frame - track["start"]
                if renderer_label != label:
                    if renderer is not None:
                        renderer.close()
                    renderer = DepthPairRenderer(track["mesh"], kmh, 1280, 720)
                    renderer_label = label
                    fusion.reset()
                front, back, mesh_mask, _ = renderer.render(track["poses"][local])
                support = _remap_mask(support_all[frame], mh_x, mh_y)
                shared = support & mesh_mask
                masks = build_static_occlusion_masks(
                    hand_mask=hm, finger_labels=fl, robot_depth=rd,
                    object_support_mask=support, mesh_mask=shared,
                    front_depth=np.where(shared, front, 0), back_depth=np.where(shared, back, 0),
                    current_mask=current, contact_baseline_mask=np.zeros_like(hm),
                    thumb_shell_m=.01958, finger_shell_m=.01465, palm_shell_m=.015,
                    spatial_close_radius_px=3, spatial_front_slack_m=.003,
                )
                vggt = masks["spar_volume_filter"] & hm
                ensemble, finger_records = fusion.fuse(
                    hand=hm, finger_labels=fl, geometry=vggt, haco=haco,
                    contact_scores=contact_scores[frame], object_support=support, mesh_support=mesh_mask,
                )
                vggt_view = composite(base, rr, rm, hm, vggt)
                ensemble_view = composite(base, rr, rm, hm, ensemble)
                rec = stats[label]
                rec["frames"] += 1
                rec["vggt_px"] += int(vggt.sum())
                rec["haco_px"] += int(haco.sum())
                rec["ensemble_px"] += int(ensemble.sum())
                rec["active_frame_fingers"] += sum(int(item["active"]) for item in finger_records)
                panels = [
                    title_panel(existing, "1 Existing 2.5D barrier", f"frame {frame} | {label}"),
                    title_panel(vggt_view, "2 VGGT-Omega geometry only", "dual-view hull + XHand thickness"),
                    title_panel(haco_view, "3 Dual-HaCo only", "MH geometry + SH same-finger confidence"),
                    title_panel(ensemble_view, "4 VGGT + HaCo ensemble", "finger confidence EMA + hysteresis + bounded fill"),
                ]
            else:
                transition_frames += 1
                fusion.reset()
                panels = [
                    title_panel(existing, "1 Existing 2.5D barrier", f"frame {frame} | Trans"),
                    title_panel(existing, "2 VGGT transition", "no active object surface"),
                    title_panel(haco_view, "3 Dual-HaCo", "contact output retained for inspection"),
                    title_panel(existing, "4 Ensemble transition", "object ensemble intentionally inactive"),
                ]
            writer.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
            if frame % 25 == 0:
                print(f"frame={frame}/552 label={label}", flush=True)
    finally:
        writer.release()
        background.release()
        if renderer is not None:
            renderer.close()

    report = {
        "schema_version": 1,
        "kind": "vggt_omega_dual_haco_finger_ensemble_comparison",
        "frames": 553,
        "fps": 10,
        "duration_seconds": 55.3,
        "panels": ["existing_2p5d", "vggt_geometry_only", "dual_haco_only", "vggt_haco_finger_ensemble"],
        "ensemble": {
            "unit": "five semantic XHand fingers; palm uses VGGT geometry only",
            "confidence": "0.46*VGGT hidden ratio + 0.46*HaCo score/mask support + 0.18*local agreement; either branch can rescue the other",
            "temporal": "EMA alpha=0.65; hysteresis on=0.40 off=0.28",
            "spatial": "7x7 close bounded to the semantic finger and an 11x11 dilation of object/mesh support",
            "haco_source": "dual-view per-finger maximum; SH supplies same-finger confidence only",
        },
        "object_stats": stats,
        "transition_frames": transition_frames,
        "output": str(output_path.resolve()),
        "limitations": [
            "VGGT-Omega surface is a relative-scale conservative convex proxy, not physical collision geometry.",
            "SH HaCo contributes confidence but its contact pixels are not projected into MH.",
            "The ensemble is deterministic confidence fusion, not a separately trained ensemble network.",
        ],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
