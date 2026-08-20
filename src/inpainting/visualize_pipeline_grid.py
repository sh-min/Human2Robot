"""Create a six-panel video showing the hand+arm replacement pipeline.

Panels:
    RAW | HAND+ARM MASK | INPAINTED BACKGROUND
    ROBOT HAND+ARM | AMODAL MASK | FINAL

The robot-only input is the locked RB5-850e + XHand render. The amodal-mask
panel is optional.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


PANEL_W = 270
PANEL_H = 480


def _open(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    return cap


def _read(cap):
    ok, frame = cap.read()
    return frame if ok else None


def _panel(frame, label):
    if frame is None:
        frame = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        cv2.putText(frame, "NOT GENERATED", (20, PANEL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2,
                    cv2.LINE_AA)
    else:
        frame = cv2.resize(frame, (PANEL_W, PANEL_H),
                           interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (PANEL_W, 34), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _overlay_mask(frame, mask, color):
    out = frame.copy()
    tint = np.zeros_like(out)
    tint[:] = color
    idx = mask.astype(bool)
    out[idx] = cv2.addWeighted(out, 0.35, tint, 0.65, 0)[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--robot_only", type=Path, default=None,
                    help="Override the robot-only video.")
    ap.add_argument("--final", type=Path, default=None,
                    help="Override the final composite video.")
    args = ap.parse_args()

    pd = args.processed_demo
    raw_path = pd / "video_L.mp4"
    robot_only_candidates = [
        pd / "overlay_processor" / "video_robot_only.mp4",
        pd / "overlay_processor_arm" / "video_robot_only.mkv",
    ]
    robot_only_path = (
        args.robot_only
        if args.robot_only is not None
        else next((p for p in robot_only_candidates if p.exists()),
                  robot_only_candidates[0])
    )
    overlay_path = pd / "overlay_processor" / "video_overlay_raw.mkv"
    final_candidates = [pd / "overlay_processor_layered" / "video_overlay.mp4"]
    final_path = (
        args.final
        if args.final is not None
        else next((p for p in final_candidates if p.exists()), None)
    )
    arm_path = pd / "segmentation_processor" / "masks_arm.npy"
    residual_path = pd / "overlay_processor" / "residual_mask.npy"
    bg_path = pd / "inpaint_processor" / "video_human_inpaint.mkv"

    amodal_candidates = [
        pd / "cube_layer" / "cube_mask_clean.npy",
        pd / "cube_layer" / "cube_mask_amodal.npy",
    ]
    amodal_path = next((p for p in amodal_candidates if p.exists()), None)

    required = [raw_path, arm_path, bg_path]
    if final_path is None:
        required.append(final_candidates[0])
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not robot_only_path.exists():
        for path in (overlay_path, residual_path):
            if not path.exists():
                raise FileNotFoundError(path)

    raw_cap = _open(raw_path)
    robot_cap = _open(robot_only_path) if robot_only_path.exists() else None
    overlay_cap = _open(overlay_path) if robot_cap is None else None
    final_cap = _open(final_path)
    bg_cap = _open(bg_path)

    fps = raw_cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video = int(raw_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    arm = np.load(arm_path, mmap_mode="r")
    residual = (
        np.load(residual_path, mmap_mode="r")
        if robot_cap is None else None
    )
    amodal = (
        np.load(amodal_path, mmap_mode="r")
        if amodal_path is not None else None
    )
    n = min(n_video, len(arm))
    if residual is not None:
        n = min(n, len(residual))
    if amodal is not None:
        n = min(n, len(amodal))

    out = args.out or (pd / "pipeline_components_rb5_850e_xhand.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (PANEL_W * 3, PANEL_H * 2),
    )

    for t in range(n):
        raw = _read(raw_cap)
        robot_only = _read(robot_cap) if robot_cap is not None else None
        overlay = _read(overlay_cap) if overlay_cap is not None else None
        final = _read(final_cap)
        bg = _read(bg_cap)
        if raw is None or final is None or bg is None:
            break

        arm_t = np.asarray(arm[t]).astype(bool)
        human_vis = _overlay_mask(raw, arm_t, (0, 0, 255))
        if robot_only is None:
            if overlay is None or residual is None:
                break
            residual_t = np.asarray(residual[t]).astype(bool)
            robot_t = arm_t & ~residual_t
            robot_only = np.zeros_like(raw)
            robot_only[robot_t] = overlay[robot_t]
        amodal_vis = (
            _overlay_mask(raw, np.asarray(amodal[t]).astype(bool), (0, 255, 0))
            if amodal is not None else None
        )

        top = np.hstack([
            _panel(raw, "RAW"),
            _panel(human_vis, "HAND + ARM MASK"),
            _panel(bg, "INPAINTED BACKGROUND"),
        ])
        bottom = np.hstack([
            _panel(robot_only, "ROBOT HAND + ARM"),
            _panel(amodal_vis, "AMODAL MASK"),
            _panel(final, "FINAL"),
        ])
        writer.write(np.vstack([top, bottom]))

        if (t + 1) % 100 == 0:
            print(f"{t + 1}/{n}")

    writer.release()
    raw_cap.release()
    if robot_cap is not None:
        robot_cap.release()
    if overlay_cap is not None:
        overlay_cap.release()
    final_cap.release()
    bg_cap.release()

    print(f"[ok] wrote {out}")
    if amodal_path is None:
        print("[missing optional] amodal mask: cube_layer/cube_mask_amodal.npy")


if __name__ == "__main__":
    main()
