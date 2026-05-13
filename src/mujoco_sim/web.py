"""Gradio MVP for interactive RBY1 + XHand pose tuning.

Run on a GPU/compute node:
    PYTHONPATH=$PWD/src MUJOCO_GL=osmesa python -m mujoco_sim.web

then SSH-tunnel from your laptop:
    ssh -L 7860:localhost:7860 worker-nodeN
and open http://localhost:7860 in a browser.

State is persistent across slider changes — moving a slider sets new ctrl
targets and the env's mj_step loop streams intermediate frames as the
robot transitions. Reset button forces back to home.
"""

from __future__ import annotations

import threading
from pathlib import Path

import gradio as gr
import mujoco
import numpy as np

from mujoco_sim.ik import solve_wrist_ik

REPO = Path(__file__).resolve().parent.parent.parent
SCENE = REPO / "src/mujoco_sim/scenes/rby1_xhand.xml"

# Orientation presets: (right_quat, left_quat) in MuJoCo (w, x, y, z).
ORIENT_PRESETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "palms-inward": (
        np.array([0.7071, 0.0, -0.7071, 0.0]),
        np.array([0.7071, 0.0, -0.7071, 0.0]),
    ),
    "palms-down": (
        np.array([0.7071, -0.7071, 0.0, 0.0]),
        np.array([0.7071, 0.7071, 0.0, 0.0]),
    ),
    "palms-up": (
        np.array([0.0, 0.0, 0.7071, -0.7071]),
        np.array([0.0, 0.0, 0.7071, 0.7071]),
    ),
}

DEFAULTS = {
    "rx": 0.55, "ry": -0.10, "rz": 1.10,
    "lx": 0.55, "ly":  0.10, "lz": 1.10,
    "head_pitch": 0.6,
    "hand_close": 0.0,
    "orient": "palms-inward",
}

# Stream parameters
SIM_FREQ_HZ = 500.0           # MJCF timestep is 0.002s -> 500Hz
YIELD_HZ = 30.0               # frames pushed to browser per second
_SUBSTEPS_PER_YIELD = int(round(SIM_FREQ_HZ / YIELD_HZ))

CAMERAS = ("head_cam", "front_view", "side_left", "side_right")

# Module-level persistent MuJoCo state.
_model = mujoco.MjModel.from_xml_path(str(SCENE))
_data = mujoco.MjData(_model)
# Per-camera render at 270x480 (16:9) -> 4 views ~ HD720 total pixels.
_RENDER_HW = (270, 480)
# EGL contexts are thread-bound; gradio's request workers may call us from
# different threads. Keep a thread-local Renderer instance.
_tls = threading.local()


def _sync_ctrl_to_qpos() -> None:
    """Set every position actuator's ctrl to its joint's current qpos so the
    controller doesn't snap the robot toward zero when stepping begins."""
    for ai in range(_model.nu):
        jid = int(_model.actuator_trnid[ai, 0])
        if jid >= 0:
            _data.ctrl[ai] = _data.qpos[_model.jnt_qposadr[jid]]


def _reset_to_home() -> None:
    """Reset to the default slider pose (cube-grasp ready)."""
    mujoco.mj_resetData(_model, _data)
    # Pre-set head pitch so IK starts with the head in place.
    hjid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
    _data.qpos[_model.jnt_qposadr[hjid]] = DEFAULTS["head_pitch"]
    mujoco.mj_forward(_model, _data)
    # Solve IK for the default wrist targets and apply as initial qpos.
    q_home = _compute_target_qpos(
        DEFAULTS["rx"], DEFAULTS["ry"], DEFAULTS["rz"],
        DEFAULTS["lx"], DEFAULTS["ly"], DEFAULTS["lz"],
        DEFAULTS["head_pitch"], DEFAULTS["hand_close"], DEFAULTS["orient"],
    )
    _data.qpos[:] = q_home
    _sync_ctrl_to_qpos()
    mujoco.mj_forward(_model, _data)


def _render(camera: str) -> np.ndarray:
    r = getattr(_tls, "renderer", None)
    if r is None:
        r = mujoco.Renderer(_model, height=_RENDER_HW[0], width=_RENDER_HW[1])
        _tls.renderer = r
    r.update_scene(_data, camera=camera)
    return r.render()


def _compute_target_qpos(
    rx: float, ry: float, rz: float,
    lx: float, ly: float, lz: float,
    head_pitch: float,
    hand_close: float,
    orient: str,
) -> np.ndarray:
    """Kinematic IK (doesn't mutate _data) producing a 38-DOF target qpos
    for every position-actuated joint. Other DOFs stay at current."""
    q = _data.qpos.copy()
    # Head
    hjid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
    q[_model.jnt_qposadr[hjid]] = head_pitch
    # Arm IK
    quat_r, quat_l = ORIENT_PRESETS[orient]
    q = solve_wrist_ik(_model, q, "link_right_arm_6", np.array([rx, ry, rz]), quat_r)
    q = solve_wrist_ik(_model, q, "link_left_arm_6",  np.array([lx, ly, lz]), quat_l)
    # Hand: target finger qpos = fraction along ctrl range
    for i in range(_model.nu):
        n = mujoco.mj_id2name(_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not n:
            continue
        if (n.startswith("rh_") or n.startswith("lh_")) and ("joint1" in n or "joint2" in n):
            jid = int(_model.actuator_trnid[i, 0])
            qa = _model.jnt_qposadr[jid]
            lo, hi = _model.actuator_ctrlrange[i]
            q[qa] = lo + hand_close * (hi - lo) * 0.9
    return q


def set_target(rx, ry, rz, lx, ly, lz, head_pitch, hand_close, orient):
    """Slider handler. Compute new IK target and push to ctrl. No outputs —
    images are fed by stream_loop() continuously."""
    q_target = _compute_target_qpos(rx, ry, rz, lx, ly, lz, head_pitch, hand_close, orient)
    for ai in range(_model.nu):
        jid = int(_model.actuator_trnid[ai, 0])
        if jid >= 0:
            _data.ctrl[ai] = q_target[_model.jnt_qposadr[jid]]


def stream_loop():
    """Infinite generator. Steps physics and yields a tuple of frames from
    all 4 cameras every ``_SUBSTEPS_PER_YIELD`` substeps."""
    while True:
        for _ in range(_SUBSTEPS_PER_YIELD):
            mujoco.mj_step(_model, _data)
        yield tuple(_render(cam) for cam in CAMERAS)


def reset_action():
    _reset_to_home()


# Initialize persistent state to the home pose at module load.
_reset_to_home()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RBY1 + XHand poser") as app:
        gr.Markdown("# RBY1 + XHand interactive poser (streaming, 4 views)")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Right wrist target")
                rx = gr.Slider(-0.3, 1.2, DEFAULTS["rx"], step=0.01, label="x")
                ry = gr.Slider(-1.0, 1.0, DEFAULTS["ry"], step=0.01, label="y")
                rz = gr.Slider(0.3, 2.0, DEFAULTS["rz"], step=0.01, label="z")
                gr.Markdown("### Left wrist target")
                lx = gr.Slider(-0.3, 1.2, DEFAULTS["lx"], step=0.01, label="x")
                ly = gr.Slider(-1.0, 1.0, DEFAULTS["ly"], step=0.01, label="y")
                lz = gr.Slider(0.3, 2.0, DEFAULTS["lz"], step=0.01, label="z")
                gr.Markdown("### Body / hand")
                head = gr.Slider(-0.35, 1.57, DEFAULTS["head_pitch"], step=0.01, label="head pitch")
                close = gr.Slider(0.0, 1.0, DEFAULTS["hand_close"], step=0.02, label="hand close")
                orient = gr.Dropdown(list(ORIENT_PRESETS), value=DEFAULTS["orient"], label="orientation")
                reset_btn = gr.Button("Reset to home", variant="secondary")
            with gr.Column(scale=3):
                outs: list[gr.Image] = []
                with gr.Row():
                    for cam in CAMERAS[:2]:
                        outs.append(gr.Image(label=cam, show_label=True, elem_classes=["cam-img"]))
                with gr.Row():
                    for cam in CAMERAS[2:]:
                        outs.append(gr.Image(label=cam, show_label=True, elem_classes=["cam-img"]))

        controls = [rx, ry, rz, lx, ly, lz, head, close, orient]
        for ctl in controls:
            ctl.change(set_target, inputs=controls, outputs=None)
        reset_btn.click(reset_action, inputs=None, outputs=None)
        app.load(stream_loop, inputs=None, outputs=outs)
    return app


_CAM_CSS = """
.cam-img,
.cam-img div:has(img) { width: 100% !important; max-width: 100% !important; }
.cam-img img { width: 100% !important; height: auto !important; object-fit: contain !important; }
"""

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, css=_CAM_CSS)
