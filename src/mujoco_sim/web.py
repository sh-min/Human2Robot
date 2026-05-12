"""Gradio MVP for interactive RBY1 + XHand pose tuning.

Run on a GPU/compute node:
    PYTHONPATH=$PWD/src MUJOCO_GL=osmesa python -m mujoco_sim.web

then SSH-tunnel from your laptop:
    ssh -L 7860:localhost:7860 worker-nodeN
and open http://localhost:7860 in a browser.

Each control change re-runs IK (per arm, 7-DOF only) and renders one frame
from the selected camera. No dynamics; collisions are not enforced.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import mujoco
import numpy as np

from mujoco_sim.ik import solve_wrist_ik

REPO = Path(__file__).resolve().parent.parent.parent
SCENE = REPO / "src/mujoco_sim/scenes/rby1_xhand.xml"

# Orientation presets: maps preset name -> (right_quat, left_quat)
# (w, x, y, z) in MuJoCo convention.
ORIENT_PRESETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "palms-inward": (
        np.array([0.7071, 0.0, -0.7071, 0.0]),
        np.array([0.7071, 0.0, -0.7071, 0.0]),
    ),
    "palms-down": (
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    ),
    "palms-up": (
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    ),
}

# Default pose values (used for reset).
DEFAULTS = {
    "rx": 0.55, "ry": -0.10, "rz": 1.10,
    "lx": 0.55, "ly":  0.10, "lz": 1.10,
    "head_pitch": 0.6,
    "hand_close": 0.0,
    "orient": "palms-inward",
    "camera": "front_view",
}

_model = mujoco.MjModel.from_xml_path(str(SCENE))
_data = mujoco.MjData(_model)
_renderer = mujoco.Renderer(_model, height=720, width=1280)


def render_pose(
    rx: float, ry: float, rz: float,
    lx: float, ly: float, lz: float,
    head_pitch: float,
    hand_close: float,
    orient: str,
    camera: str,
) -> np.ndarray:
    mujoco.mj_resetData(_model, _data)

    # Head pitch (head_1 joint) via qpos so head_cam follows.
    hjid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "head_1")
    _data.qpos[_model.jnt_qposadr[hjid]] = head_pitch
    mujoco.mj_forward(_model, _data)

    quat_r, quat_l = ORIENT_PRESETS[orient]
    q = _data.qpos.copy()
    q = solve_wrist_ik(_model, q, "link_right_arm_6", np.array([rx, ry, rz]), quat_r)
    q = solve_wrist_ik(_model, q, "link_left_arm_6",  np.array([lx, ly, lz]), quat_l)
    _data.qpos[:] = q

    # Hand closure: drive flexion joints (joint1/joint2) on both hands.
    for i in range(_model.nu):
        n = mujoco.mj_id2name(_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not n:
            continue
        if (n.startswith("rh_") or n.startswith("lh_")) and ("joint1" in n or "joint2" in n):
            jid = _model.actuator_trnid[i, 0]
            qa = _model.jnt_qposadr[jid]
            hi = _model.actuator_ctrlrange[i, 1]
            lo = _model.actuator_ctrlrange[i, 0]
            _data.qpos[qa] = lo + hand_close * (hi - lo) * 0.9

    mujoco.mj_forward(_model, _data)
    _renderer.update_scene(_data, camera=camera)
    return _renderer.render()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RBY1 + XHand poser") as app:
        gr.Markdown("# RBY1 + XHand interactive poser")
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
                camera = gr.Radio(["head_cam", "front_view"], value=DEFAULTS["camera"], label="camera")
            with gr.Column(scale=2):
                out = gr.Image(label="render", height=540)

        controls = [rx, ry, rz, lx, ly, lz, head, close, orient, camera]
        for ctl in controls:
            ctl.change(render_pose, inputs=controls, outputs=out)

        # Initial render on load
        app.load(render_pose, inputs=controls, outputs=out)
    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
