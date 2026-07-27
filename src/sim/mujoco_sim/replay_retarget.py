"""Render a full retargeted episode (final_pose.pkl) as an mp4.

Loops every frame: solve bimanual IK + copy finger qpos via apply_frame,
render head_cam and front_view side-by-side, write an mp4.

Usage:
    MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m mujoco_sim.replay_retarget \
        --pkl data/cube_dataset/0412_val/episode_0/rgb_hawor/final_pose.pkl \
        --out output/episode_0_retarget.mp4
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import pinocchio as pin

from .ik_arm import (
    SCENE,
    apply_frame,
    cam_to_world,
    head_cam_world,
    solve_arm_ik,
    wrist_to_arm6,
)

REPO = Path(__file__).resolve().parents[3]


class ReferenceReader:
    """Sequential RGB reader for either a video or an image directory."""

    def __init__(self, source: Path | None):
        self.source = source
        self.cap = None
        self.paths: list[Path] = []
        self.index = 0
        self.sample = None
        if source is None or not source.exists():
            return
        if source.is_file():
            self.cap = cv2.VideoCapture(str(source))
            ok, bgr = self.cap.read()
            if not ok:
                self.cap.release()
                self.cap = None
                return
            self.sample = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            paths = [
                path for path in source.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]

            def numeric_key(path: Path):
                match = re.search(r"(\d+)", path.stem)
                return (int(match.group(1)) if match else -1, path.name)

            self.paths = sorted(paths, key=numeric_key)
            if self.paths:
                self.sample = imageio.imread(self.paths[0])

    @property
    def available(self) -> bool:
        return self.sample is not None

    def read(self) -> np.ndarray | None:
        if self.cap is not None:
            ok, bgr = self.cap.read()
            if not ok:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.index >= len(self.paths):
            return None
        image = imageio.imread(self.paths[self.index])
        self.index += 1
        return image

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/cube_dataset/0412_val/episode_0/rgb_hawor/final_pose.pkl")
    ap.add_argument("--rgb-dir", default="data/cube_dataset/0412_val/episode_0/rgb_hawor/extracted_images",
                    help="Legacy reference image directory; arbitrary numeric filenames supported.")
    ap.add_argument("--reference", default=None,
                    help="Reference video or image directory shown beside the simulation.")
    ap.add_argument("--out", default="output/episode_0_retarget.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--head-pitch", type=float, default=0.6)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    pkl_path = REPO / args.pkl
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pkl_path.open("rb") as f:
        pose = pickle.load(f)

    pin_model = pin.buildModelFromMJCF(str(SCENE))
    pin_data = pin_model.createData()
    muj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    muj_data = mujoco.MjData(muj_model)
    # Robot joints come first in MuJoCo's qpos/qvel; cube state follows.
    # Pinocchio sees only the robot, so its nq/nv give the slice into
    # MuJoCo's state that corresponds to the robot.
    ROBOT_NQ = pin_model.nq
    ROBOT_NV = pin_model.nv

    q_home = muj_model.qpos0.copy()
    hjid = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
    q_home[muj_model.jnt_qposadr[hjid]] = args.head_pitch
    # Use the calibration's reachable anchor configuration when available.
    # Legacy camera-frame trajectories fall back to joint-range midpoints.
    for side in ("right", "left"):
        natural = pose.get(side, {}).get("natural_arm_qpos")
        for i in range(7):
            jid = mujoco.mj_name2id(muj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_{i}")
            lo, hi = muj_model.jnt_range[jid]
            value = natural[i] if natural is not None else 0.5 * (lo + hi)
            q_home[muj_model.jnt_qposadr[jid]] = value
    muj_data.qpos[:] = q_home
    mujoco.mj_forward(muj_model, muj_data)

    T_world_cam = head_cam_world(pin_model, pin_data, q_home[:ROBOT_NQ])
    renderer = mujoco.Renderer(muj_model, height=args.height, width=args.width)

    # Drive the robot through its position actuators: each frame compute
    # the IK target via pinocchio, push to ctrl, let mj_step's PD track.
    # Cube physics and any contact reactions emerge naturally from the
    # same mj_step. n_substeps makes sim time advance ~1/fps per frame.
    n_substeps = max(1, int(round(1.0 / (args.fps * muj_model.opt.timestep))))

    # Sync actuator targets to the seeded home pose so the first mj_step
    # doesn't snap the robot toward zero.
    for ai in range(muj_model.nu):
        jid = int(muj_model.actuator_trnid[ai, 0])
        if jid >= 0:
            muj_data.ctrl[ai] = muj_data.qpos[muj_model.jnt_qposadr[jid]]

    reference_path = (
        Path(args.reference)
        if args.reference is not None
        else Path(args.rgb_dir)
    )
    if not reference_path.is_absolute():
        reference_path = REPO / reference_path
    reference = ReferenceReader(reference_path)
    if reference.available:
        # Resize reference to match render height; preserve aspect.
        sample = reference.sample
        assert sample is not None
        rgb_target_w = int(round(sample.shape[1] * args.height / sample.shape[0]))
        print(
            f"using reference from {reference_path} "
            f"({sample.shape[1]}x{sample.shape[0]} -> "
            f"{rgb_target_w}x{args.height})"
        )
    else:
        print(f"reference {reference_path} unavailable; skipping panel")

    T = int(pose["T"])
    print(f"rendering {T} frames -> {out_path}")
    tmp = Path(tempfile.mkdtemp(prefix="replay_retarget_"))
    t0 = time.time()
    ik_errors = {"right": [], "left": []}
    try:
        for t in range(T):
            errors = apply_frame(
                pin_model, pin_data, muj_model, muj_data, pose, t, T_world_cam
            )
            for side, values in errors.items():
                ik_errors[side].append(values)
            robot_q = muj_data.qpos[:ROBOT_NQ].copy()
            for _ in range(n_substeps):
                mujoco.mj_step(muj_model, muj_data)
                muj_data.qpos[:ROBOT_NQ] = robot_q
                muj_data.qvel[:ROBOT_NV] = 0
            renderer.update_scene(muj_data, camera="head_cam")
            img_head = renderer.render()
            renderer.update_scene(muj_data, camera="front_view")
            img_front = renderer.render()
            panels = []
            rgb = reference.read()
            if reference.available:
                if rgb is None:
                    assert reference.sample is not None
                    rgb = np.zeros_like(reference.sample)
                rgb = cv2.resize(
                    rgb, (rgb_target_w, args.height),
                    interpolation=cv2.INTER_AREA,
                )
                panels.append(rgb)
            panels.extend([img_head, img_front])
            imageio.imwrite(tmp / f"frame_{t:05d}.png", np.concatenate(panels, axis=1))
            if (t + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"  {t+1}/{T}  ({(t+1)/dt:.1f} fps)")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(args.fps),
            "-i", str(tmp / "frame_%05d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ], check=True)
    finally:
        renderer.close()
        reference.close()
        shutil.rmtree(tmp, ignore_errors=True)

    metrics = {"frames": T, "hands": {}}
    for side, values in ik_errors.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == 0:
            metrics["hands"][side] = {"valid_frames": 0}
            continue
        pos_mm = arr[:, 0] * 1000.0
        ori_deg = np.degrees(arr[:, 1])
        metrics["hands"][side] = {
            "valid_frames": int(len(arr)),
            "position_error_mm": {
                "mean": float(pos_mm.mean()),
                "p95": float(np.percentile(pos_mm, 95)),
                "max": float(pos_mm.max()),
            },
            "orientation_error_deg": {
                "mean": float(ori_deg.mean()),
                "p95": float(np.percentile(ori_deg, 95)),
                "max": float(ori_deg.max()),
            },
        }
        print(
            f"{side}: IK position mean/p95/max="
            f"{pos_mm.mean():.2f}/{np.percentile(pos_mm, 95):.2f}/"
            f"{pos_mm.max():.2f} mm, orientation="
            f"{ori_deg.mean():.2f}/{np.percentile(ori_deg, 95):.2f}/"
            f"{ori_deg.max():.2f} deg"
        )
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {out_path}  ({time.time()-t0:.1f}s)")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
