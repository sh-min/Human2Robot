"""Dynamics-based replay of a friend's retargeted episode on RBY1+XHand.

Per sim step: query PhysX for each arm's Jacobian, run one differential-IK
step (damped LS) on each arm toward the data's wrist 6D target, set the
arm + hand + head joint targets, then step physics. PD chases the targets
so contacts with the cube are physical.

Run (from repo root, in the isaac_lab env):
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/src \\
        python -m isaac_lab.replay_retarget --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--npz",
    type=str,
    default="data/cube_dataset/0412_val/episode_0/rgb_hawor/final_pose.npz",
)
parser.add_argument("--out", type=str, default="output/episode_0_isaac_replay.mp4")
parser.add_argument("--max-frames", type=int, default=0, help="0 = whole episode")
parser.add_argument("--data-fps", type=int, default=30)
parser.add_argument("--substeps", type=int, default=2, help="sim steps per data frame")
parser.add_argument("--head-pitch", type=float, default=0.6)
parser.add_argument("--settle-steps", type=int, default=30,
                    help="sim steps to let robot settle into home pose before replay")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .scene import build_scene

REPO = Path(__file__).resolve().parents[3]
USD_PATH = REPO / "src/sim/isaac_lab/assets/rby1_xhand/rby1_xhand.usd"


# --- Geometry constants (physical facts about how the scene was composed) ---
# CV/HaWoR camera frame: (+x right, +y down, +z into scene).
# MuJoCo/Isaac camera frame (this scene's head_cam): (+x right, +y up, +z out of scene).
_CV_TO_MJ = np.diag([1.0, -1.0, -1.0])

# head_cam mount on link_head_2 from the original MJCF: pos=(0.12, 0, 0.05),
# quat=(-0.5, -0.5, 0.5, 0.5). Equivalent rotation matrix:
_R_HEAD2_CAM = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
_T_HEAD2_CAM_POS = np.array([0.12, 0.0, 0.05])

# XHand mounted on link_*_arm_6 at z=-0.1261 in arm_6's local frame, so to put
# the visible wrist link at T_world_wrist we drive arm_6 to T_world_wrist
# shifted +0.1261 along its own local z.
_WRIST_ARM6_OFFSET = np.array([0.0, 0.0, 0.1261])


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w, x, y, z). Stable branch."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def world_arm6_target(
    wrist_pos_cv: np.ndarray, wrist_rot_cv: np.ndarray,
    R_world_cam: np.ndarray, t_world_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Friend's wrist pose in CV-cam frame -> world target for link_*_arm_6.

    Returns (pos_world (3,), quat_world_wxyz (4,)).
    """
    # CV-cam frame -> MJ/Isaac cam frame.
    R_camlocal_wrist = _CV_TO_MJ @ wrist_rot_cv
    p_camlocal_wrist = _CV_TO_MJ @ wrist_pos_cv
    # cam frame -> world.
    R_world_wrist = R_world_cam @ R_camlocal_wrist
    p_world_wrist = t_world_cam + R_world_cam @ p_camlocal_wrist
    # wrist link -> arm_6 (shift +0.1261 along wrist local z; orientation same).
    p_world_arm6 = p_world_wrist + R_world_wrist @ _WRIST_ARM6_OFFSET
    return p_world_arm6, quat_wxyz_from_matrix(R_world_wrist)


def main():
    # --- Friend's retargeted data ---
    data = np.load(REPO / args.npz)
    T = int(data["T"])
    if args.max_frames > 0:
        T = min(T, args.max_frames)
    print(f"[replay] {T} frames @ {args.data_fps} fps ({T/args.data_fps:.1f}s)")

    # --- Sim + scene ---
    sim_dt = 1.0 / (args.data_fps * args.substeps)
    sim = SimulationContext(SimulationCfg(dt=sim_dt))
    cube = build_scene()

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_.*", "left_arm_.*"],
                stiffness=400.0, damping=40.0,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=["rh_.*", "lh_.*"],
                stiffness=20.0, damping=1.0,
            ),
            "head": ImplicitActuatorCfg(
                joint_names_expr=["head_.*"],
                stiffness=200.0, damping=20.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    cam_cfg = CameraCfg(
        prim_path="/World/FrontCam",
        update_period=0,
        height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    camera = Camera(cfg=cam_cfg)

    sim.reset()
    device = robot.device
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[2.2, 0.0, 1.7]], device=device),
        targets=torch.tensor([[0.0, 0.0, 1.1]], device=device),
    )

    # --- Joint / body resolution ---
    right_arm_names = [f"right_arm_{i}" for i in range(7)]
    left_arm_names = [f"left_arm_{i}" for i in range(7)]
    rh_hand_names = [f"rh_{n}" for n in data["right_joint_names"]]
    lh_hand_names = [f"lh_{n}" for n in data["left_joint_names"]]

    right_arm_jids = robot.find_joints(right_arm_names, preserve_order=True)[0]
    left_arm_jids = robot.find_joints(left_arm_names, preserve_order=True)[0]
    rh_hand_jids = robot.find_joints(rh_hand_names, preserve_order=True)[0]
    lh_hand_jids = robot.find_joints(lh_hand_names, preserve_order=True)[0]
    head_jid = robot.find_joints(["head_1"], preserve_order=True)[0]

    right_arm6_bid = robot.find_bodies(["link_right_arm_6"])[0][0]
    left_arm6_bid = robot.find_bodies(["link_left_arm_6"])[0][0]
    head2_bid = robot.find_bodies(["link_head_2"])[0][0]

    # The USD is built with set_fix_base(True) so a FixedJoint pins base to
    # world; PhysX still reports is_fixed_base=False but the kinematics ARE
    # fixed-base, so the root body is excluded from get_jacobians() and we
    # subtract 1 from the body index to land on the right Jacobian slot.
    right_jacobi = right_arm6_bid - 1
    left_jacobi = left_arm6_bid - 1

    # --- Home pose: arms at mid-range (elbow-down branch), head pitched down ---
    # Joint limits from PhysX. Default joint pos for arms is mid-range of the
    # registered limits; the converter usually keeps MJCF jnt_range.
    lo = robot.data.soft_joint_pos_limits[0, :, 0].cpu().numpy()
    hi = robot.data.soft_joint_pos_limits[0, :, 1].cpu().numpy()
    home = robot.data.default_joint_pos[0].cpu().numpy().copy()
    for jid in right_arm_jids + left_arm_jids:
        home[jid] = 0.5 * (lo[jid] + hi[jid])
    home[head_jid[0]] = args.head_pitch
    # Hands stay at 0 (default).
    home_t = torch.tensor(home, device=device, dtype=torch.float32).unsqueeze(0)
    zero_v = torch.zeros_like(home_t)
    robot.write_joint_state_to_sim(home_t, zero_v)
    robot.set_joint_position_target(home_t)
    robot.write_data_to_sim()
    # Settle into home with PD before reading head pose (PhysX needs a few ticks).
    for _ in range(args.settle_steps):
        sim.step()
        robot.update(sim_dt)

    # --- head_cam world pose (head fixed throughout replay) ---
    head2_pose = robot.data.body_pose_w[0, head2_bid].cpu().numpy()
    head2_pos = head2_pose[0:3]
    head2_quat = head2_pose[3:7]  # (w, x, y, z)
    head2_R = matrix_from_quat(torch.tensor(head2_quat, dtype=torch.float64)).numpy()
    R_world_cam = head2_R @ _R_HEAD2_CAM
    t_world_cam = head2_pos + head2_R @ _T_HEAD2_CAM_POS
    print(f"[replay] head_cam world pos={t_world_cam}")

    # --- IK controllers (one per arm), damped LS, absolute pose command ---
    # Commands are root-frame (matching the tutorial pattern); controller
    # expects current EE pose in the same frame as command.
    ctrl_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                           ik_method="dls", ik_params={"lambda_val": 0.01})
    ctrl_r = DifferentialIKController(ctrl_cfg, num_envs=1, device=device)
    ctrl_l = DifferentialIKController(ctrl_cfg, num_envs=1, device=device)

    # --- Replay loop ---
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="isaac_replay_"))
    print(f"  frames -> {tmp}")

    # Pre-grab the (constant during replay) root pose to convert world targets
    # to root frame, which is what the controller expects.
    root_pose_const = robot.data.root_pose_w.clone()  # (1, 7)
    root_pos_np = root_pose_const[0, :3].cpu().numpy()
    root_quat_np = root_pose_const[0, 3:7].cpu().numpy()  # (w, x, y, z)
    root_R = matrix_from_quat(torch.tensor(root_quat_np, dtype=torch.float64)).numpy()
    print(f"[replay] root world pose: pos={root_pos_np}, quat={root_quat_np}")

    def world_to_root(p_w: np.ndarray, R_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Express a world-frame pose in the articulation-root frame."""
        R_b = root_R.T @ R_w
        p_b = root_R.T @ (p_w - root_pos_np)
        return p_b, R_b

    t0 = time.time()
    cmd_r = torch.zeros(1, 7, device=device, dtype=torch.float32)
    cmd_l = torch.zeros(1, 7, device=device, dtype=torch.float32)
    try:
        for t in range(T):
            # Compute world-frame arm6 targets, then express in root frame.
            if bool(data["right_valid"][t]):
                p_w, q_w = world_arm6_target(
                    data["right_wrist_pos"][t].astype(np.float64),
                    data["right_wrist_rot"][t].astype(np.float64),
                    R_world_cam, t_world_cam,
                )
                R_w = matrix_from_quat(torch.tensor(q_w)).numpy()
                p_b, R_b = world_to_root(p_w, R_w)
                q_b = quat_wxyz_from_matrix(R_b)
                cmd_r[0, :3] = torch.from_numpy(p_b).float()
                cmd_r[0, 3:] = torch.from_numpy(q_b).float()
                ctrl_r.set_command(cmd_r)
            if bool(data["left_valid"][t]):
                p_w, q_w = world_arm6_target(
                    data["left_wrist_pos"][t].astype(np.float64),
                    data["left_wrist_rot"][t].astype(np.float64),
                    R_world_cam, t_world_cam,
                )
                R_w = matrix_from_quat(torch.tensor(q_w)).numpy()
                p_b, R_b = world_to_root(p_w, R_w)
                q_b = quat_wxyz_from_matrix(R_b)
                cmd_l[0, :3] = torch.from_numpy(p_b).float()
                cmd_l[0, 3:] = torch.from_numpy(q_b).float()
                ctrl_l.set_command(cmd_l)

            hand_r = torch.tensor(data["right_qpos"][t].astype(np.float32), device=device).unsqueeze(0)
            hand_l = torch.tensor(data["left_qpos"][t].astype(np.float32), device=device).unsqueeze(0)
            head_tgt = torch.tensor([[args.head_pitch]], device=device, dtype=torch.float32)

            for _ in range(args.substeps):
                # Current state for IK -- convert EE world pose to root frame
                # (matches command frame; tutorial pattern). For our fixed-base
                # robot at origin this is usually a no-op but the articulation
                # root body may carry a small intrinsic offset, so do it.
                J_all = robot.root_physx_view.get_jacobians()
                J_r = J_all[:, right_jacobi, :, right_arm_jids]
                J_l = J_all[:, left_jacobi, :, left_arm_jids]
                root_pose = robot.data.root_pose_w
                ee_w_r = robot.data.body_pose_w[:, right_arm6_bid]
                ee_w_l = robot.data.body_pose_w[:, left_arm6_bid]
                ee_b_r_pos, ee_b_r_quat = subtract_frame_transforms(
                    root_pose[:, :3], root_pose[:, 3:7], ee_w_r[:, :3], ee_w_r[:, 3:7]
                )
                ee_b_l_pos, ee_b_l_quat = subtract_frame_transforms(
                    root_pose[:, :3], root_pose[:, 3:7], ee_w_l[:, :3], ee_w_l[:, 3:7]
                )
                qarm_r = robot.data.joint_pos[:, right_arm_jids]
                qarm_l = robot.data.joint_pos[:, left_arm_jids]
                arm_des_r = ctrl_r.compute(ee_b_r_pos, ee_b_r_quat, J_r, qarm_r)
                arm_des_l = ctrl_l.compute(ee_b_l_pos, ee_b_l_quat, J_l, qarm_l)
                if t < 5 and _ == 0:
                    print(
                        f"[dbg t={t}] tgt={cmd_r[0,:3].cpu().numpy()} cur={ee_b_r_pos[0].cpu().numpy()} "
                        f"qarm={qarm_r[0].cpu().numpy()[:4]} qdes={arm_des_r[0].cpu().numpy()[:4]}"
                    )

                robot.set_joint_position_target(arm_des_r, joint_ids=right_arm_jids)
                robot.set_joint_position_target(arm_des_l, joint_ids=left_arm_jids)
                robot.set_joint_position_target(hand_r, joint_ids=rh_hand_jids)
                robot.set_joint_position_target(hand_l, joint_ids=lh_hand_jids)
                robot.set_joint_position_target(head_tgt, joint_ids=head_jid)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim_dt)

            camera.update(sim_dt)
            rgb = camera.data.output["rgb"][0].cpu().numpy()[..., :3]
            imageio.imwrite(tmp / f"frame_{t:05d}.png", rgb)
            if t % 30 == 0:
                err_r = float(torch.linalg.norm(cmd_r[0, :3] - ee_b_r_pos[0]))
                err_l = float(torch.linalg.norm(cmd_l[0, :3] - ee_b_l_pos[0]))
                print(f"  t={t:>4}/{T}  ee_err R/L = {err_r*1000:5.1f}/{err_l*1000:5.1f} mm")

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(args.data_fps),
                "-i", str(tmp / "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            check=True,
        )
        print(f"wrote {out_path}  ({time.time()-t0:.1f}s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import os
    import traceback

    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    # Kit's simulation_app.close() reliably hangs on the replicator/physx
    # plugin shutdown path; skip it and let the OS reap fds/threads.
    os._exit(exit_code)
