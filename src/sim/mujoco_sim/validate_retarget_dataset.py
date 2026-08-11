"""Validate every exported policy trajectory against RBY1 joint-limited IK.

Episodes are discovered dynamically.  The JSON report is suitable both for a
human review and as a gate before LeRobot conversion when new data is added.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from .ik_arm import SCENE, apply_frame


def _set_initial_pose(
    pose: dict,
    muj_model: mujoco.MjModel,
    muj_data: mujoco.MjData,
) -> None:
    q_home = muj_model.qpos0.copy()
    for side in ("right", "left"):
        natural = pose.get(side, {}).get("natural_arm_qpos")
        for i in range(7):
            jid = mujoco.mj_name2id(
                muj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_arm_{i}",
            )
            lo, hi = muj_model.jnt_range[jid]
            q_home[muj_model.jnt_qposadr[jid]] = (
                natural[i] if natural is not None else 0.5 * (lo + hi)
            )
    muj_data.qpos[:] = q_home
    muj_data.qvel[:] = 0
    mujoco.mj_forward(muj_model, muj_data)


def _summarize(values: list[tuple[float, float]]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"valid_frames": 0}
    pos_mm = 1000.0 * array[:, 0]
    ori_deg = np.degrees(array[:, 1])
    return {
        "valid_frames": int(len(array)),
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


def validate_episode(
    path: Path,
    pin_model: pin.Model,
    pin_data: pin.Data,
    muj_model: mujoco.MjModel,
    muj_data: mujoco.MjData,
    stride: int,
) -> dict:
    with path.open("rb") as handle:
        pose = pickle.load(handle)
    if pose.get("coordinate_frame") != "rby1_base":
        raise ValueError(
            f"{path} is in {pose.get('coordinate_frame')!r}, not 'rby1_base'"
        )
    _set_initial_pose(pose, muj_model, muj_data)

    errors: dict[str, list[tuple[float, float]]] = {
        "right": [],
        "left": [],
    }
    for t in range(0, int(pose["T"]), stride):
        frame_errors = apply_frame(
            pin_model,
            pin_data,
            muj_model,
            muj_data,
            pose,
            t,
            pin.SE3.Identity(),  # unused by rby1_base trajectories
        )
        for side, values in frame_errors.items():
            errors[side].append(values)
    return {
        "frames": int(pose["T"]),
        "sample_stride": stride,
        "hands": {side: _summarize(values) for side, values in errors.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--episode_glob", default="IMG_*")
    parser.add_argument("--max_position_p95_mm", type=float, default=10.0)
    parser.add_argument("--max_position_max_mm", type=float, default=30.0)
    parser.add_argument("--max_orientation_p95_deg", type=float, default=5.0)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    data_root = args.data_root.resolve()
    paths = sorted(
        data_root.glob(f"{args.episode_glob}/rgb_hawor/final_pose.pkl")
    )
    if not paths:
        raise FileNotFoundError(
            f"No {args.episode_glob}/rgb_hawor/final_pose.pkl "
            f"in {data_root}"
        )

    pin_model = pin.buildModelFromMJCF(str(SCENE))
    pin_data = pin_model.createData()
    muj_model = mujoco.MjModel.from_xml_path(str(SCENE))
    muj_data = mujoco.MjData(muj_model)

    report = {
        "data_root": str(data_root),
        "thresholds": {
            "position_p95_mm": args.max_position_p95_mm,
            "position_max_mm": args.max_position_max_mm,
            "orientation_p95_deg": args.max_orientation_p95_deg,
        },
        "episodes": {},
        "passed": True,
    }
    for path in paths:
        episode_id = path.parents[1].name
        result = validate_episode(
            path,
            pin_model,
            pin_data,
            muj_model,
            muj_data,
            args.stride,
        )
        episode_passed = True
        for hand in result["hands"].values():
            if hand["valid_frames"] == 0:
                continue
            episode_passed &= (
                hand["position_error_mm"]["p95"]
                <= args.max_position_p95_mm
                and hand["position_error_mm"]["max"]
                <= args.max_position_max_mm
                and hand["orientation_error_deg"]["p95"]
                <= args.max_orientation_p95_deg
            )
        result["passed"] = bool(episode_passed)
        report["passed"] &= bool(episode_passed)
        report["episodes"][episode_id] = result
        valid_hand_stats = [
            f"{side}:p95={hand['position_error_mm']['p95']:.2f}mm"
            for side, hand in result["hands"].items()
            if hand["valid_frames"]
        ]
        print(
            f"[{'PASS' if episode_passed else 'FAIL'}] {episode_id} "
            + " ".join(valid_hand_stats)
        )

    out_path = (
        args.out.resolve()
        if args.out is not None
        else data_root / "policy_trajectory_validation.json"
    )
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
