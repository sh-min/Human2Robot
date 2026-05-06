"""Interactive 3D playback (trimesh viewer): xhand robots + MANO skeletons +
camera frustum, all in cam frame.

Controls:
    SPACE       toggle pause/play
    LEFT/RIGHT  step ±1 frame (auto-pauses)
    HOME/END    jump to first / last frame
    drag mouse  rotate; scroll = zoom; right-drag = pan

Usage:
    conda activate RFM_retarget
    python play_sequence.py \
        --npz       <...>/cam0_hawor/retarget_input.npz \
        --right_pkl <...>/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl  <...>/cam0_hawor/qpos_xhand_left.pkl
"""
import argparse
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
import pyglet
import trimesh
from scipy.spatial.transform import Rotation as Rscipy
from trimesh.viewer import SceneViewer

from _paths import URDF_ROOT

R_MANO_XHAND = {
    "right": np.array([[0,  0,  1], [0, -1,  0], [1,  0,  0]], dtype=np.float64),
    "left":  np.array([[0,  0, -1], [0,  1,  0], [1,  0,  0]], dtype=np.float64),
}

MANO_BONES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]


def joint_color(idx):
    if idx == 0:                return [40,  40,  40, 255]
    if 1  <= idx <= 4:          return [220, 60,  60, 255]
    if 5  <= idx <= 8:          return [60, 170, 220, 255]
    if 9  <= idx <= 12:         return [80, 200,  90, 255]
    if 13 <= idx <= 16:         return [220,160,  60, 255]
    if 17 <= idx <= 20:         return [180, 80, 200, 255]
    return [128, 128, 128, 255]


def load_xhand(hand):
    urdf = os.path.join(URDF_ROOT, "xhand", f"xhand_{hand}.urdf")
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    tree = ET.parse(urdf)
    mesh_dir = Path(urdf).parent
    color = [180, 220, 230, 110] if hand == "right" else [240, 200, 195, 110]
    meshes = {}
    for link in tree.getroot().findall("link"):
        nm = link.attrib["name"]
        v = link.find("visual")
        if v is None:
            continue
        mn = v.find("geometry/mesh")
        if mn is None:
            continue
        try:
            m = trimesh.load(str(mesh_dir / mn.attrib["filename"]), force="mesh")
            m.visual.face_colors = color
            meshes[nm] = m
        except Exception:
            pass
    return model, data, meshes


def fk_world_transforms(model, data, qpos, base_T):
    pin.forwardKinematics(model, data, qpos)
    pin.updateFramePlacements(model, data)
    out = {}
    for f in range(model.nframes):
        nm = model.frames[f].name
        T = np.eye(4)
        T[:3, :3] = np.array(data.oMf[f].rotation)
        T[:3, 3] = np.array(data.oMf[f].translation)
        out[nm] = base_T @ T
    return out


def thin_line(p0, p1, color, radius=0.0012):
    if np.linalg.norm(np.asarray(p1) - np.asarray(p0)) < 1e-9:
        return None
    c = trimesh.creation.cylinder(radius=radius, segment=[p0, p1])
    c.visual.face_colors = color
    return c


def bone_transform(p0, p1):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    L = float(np.linalg.norm(diff))
    if L < 1e-9:
        T = np.eye(4); T[:3, 3] = [0, 0, 1000]; return T
    z = diff / L
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, z)
    cn = np.linalg.norm(cross)
    if cn < 1e-9:
        R = np.eye(3) if z_axis @ z > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        ang = np.arccos(np.clip(z_axis @ z, -1.0, 1.0))
        R = Rscipy.from_rotvec(cross / cn * ang).as_matrix()
    M = np.eye(4)
    M[:3, 0] = R[:, 0]
    M[:3, 1] = R[:, 1]
    M[:3, 2] = R[:, 2] * L
    M[:3, 3] = (p0 + p1) / 2.0
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--right_pkl", required=True)
    ap.add_argument("--left_pkl", required=True)
    ap.add_argument("--img_focal", type=float, default=497.77)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--img_h", type=int, default=480)
    ap.add_argument("--frustum_depth", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--bones", dest="bones", action="store_true", default=True)
    ap.add_argument("--no_bones", dest="bones", action="store_false")
    ap.add_argument("--bone_radius", type=float, default=0.0035)
    args = ap.parse_args()

    data = np.load(args.npz)
    j_left = data["joints_left"]
    j_right = data["joints_right"]
    mano_root = data["mano_global_orient"]
    valid = data["valid"]

    rd = pickle.load(open(args.right_pkl, "rb"))
    ld = pickle.load(open(args.left_pkl, "rb"))
    qr = np.asarray(rd["data"])
    ql = np.asarray(ld["data"])
    T = qr.shape[0]
    print(f"Loaded sequence: T={T} frames")

    model_r, data_r, meshes_r = load_xhand("right")
    model_l, data_l, meshes_l = load_xhand("left")

    def pin_qpos_map(model, dex_names):
        pin_names = [model.names[i] for i in range(1, len(model.names))]
        return np.array([dex_names.index(n) for n in pin_names], dtype=int)

    map_r = pin_qpos_map(model_r, rd["joint_names"])
    map_l = pin_qpos_map(model_l, ld["joint_names"])

    scene = trimesh.Scene()
    for name, m in meshes_r.items():
        scene.add_geometry(m, geom_name=f"R_{name}", node_name=f"R_{name}")
    for name, m in meshes_l.items():
        scene.add_geometry(m, geom_name=f"L_{name}", node_name=f"L_{name}")

    for hand in ("right", "left"):
        short = "R" if hand == "right" else "L"
        for i in range(21):
            r = 0.009 if i == 0 else 0.005
            s = trimesh.creation.icosphere(radius=r, subdivisions=1)
            s.visual.face_colors = joint_color(i)
            scene.add_geometry(s, geom_name=f"M{short}_{i}", node_name=f"M{short}_{i}")

    if args.bones:
        for hand in ("right", "left"):
            short = "R" if hand == "right" else "L"
            for k, (a, b) in enumerate(MANO_BONES):
                cyl = trimesh.creation.cylinder(radius=args.bone_radius, height=1.0, sections=12)
                cyl.visual.face_colors = joint_color(b)
                scene.add_geometry(cyl, geom_name=f"B{short}_{k}", node_name=f"B{short}_{k}")

    fx = args.img_focal
    cx, cy = args.img_w / 2, args.img_h / 2
    z = args.frustum_depth
    corners = []
    for u, v in [(0, 0), (args.img_w, 0), (args.img_w, args.img_h), (0, args.img_h)]:
        corners.append(np.array([(u - cx) / fx * z, (v - cy) / fx * z, z]))
    edges = [(np.zeros(3), c) for c in corners] + \
            [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    for i, (p0, p1) in enumerate(edges):
        m = thin_line(p0, p1, [180, 180, 195, 255])
        if m is not None:
            scene.add_geometry(m, geom_name=f"frustum_{i}")

    for ax_idx, c in enumerate([[230, 50, 50, 255], [50, 200, 50, 255], [50, 80, 230, 255]]):
        end = np.zeros(3); end[ax_idx] = 0.05
        m = thin_line(np.zeros(3), end, c, radius=0.0025)
        if m is not None:
            scene.add_geometry(m, geom_name=f"axis_{ax_idx}")

    state = {"frame": 0, "paused": False}

    def far_T():
        T = np.eye(4); T[:3, 3] = [0, 0, 1000]
        return T

    def apply_frame(t):
        if valid[1, t]:
            R_cm = Rscipy.from_rotvec(mano_root[1, t]).as_matrix()
            base_T = np.eye(4)
            base_T[:3, :3] = R_cm @ R_MANO_XHAND["right"]
            base_T[:3, 3] = j_right[t, 0]
            tf = fk_world_transforms(model_r, data_r, qr[t][map_r], base_T)
            for nm in meshes_r:
                if nm in tf:
                    scene.graph.update(frame_to=f"R_{nm}", frame_from="world", matrix=tf[nm])
        else:
            for nm in meshes_r:
                scene.graph.update(frame_to=f"R_{nm}", frame_from="world", matrix=far_T())

        if valid[0, t]:
            R_cm = Rscipy.from_rotvec(mano_root[0, t]).as_matrix()
            base_T = np.eye(4)
            base_T[:3, :3] = R_cm @ R_MANO_XHAND["left"]
            base_T[:3, 3] = j_left[t, 0]
            tf = fk_world_transforms(model_l, data_l, ql[t][map_l], base_T)
            for nm in meshes_l:
                if nm in tf:
                    scene.graph.update(frame_to=f"L_{nm}", frame_from="world", matrix=tf[nm])
        else:
            for nm in meshes_l:
                scene.graph.update(frame_to=f"L_{nm}", frame_from="world", matrix=far_T())

        for hand_name, h_idx, joints_arr in [("right", 1, j_right), ("left", 0, j_left)]:
            short = "R" if hand_name == "right" else "L"
            if valid[h_idx, t]:
                for i in range(21):
                    Tm = np.eye(4); Tm[:3, 3] = joints_arr[t, i]
                    scene.graph.update(frame_to=f"M{short}_{i}", frame_from="world", matrix=Tm)
                if args.bones:
                    for k, (a, b) in enumerate(MANO_BONES):
                        M = bone_transform(joints_arr[t, a], joints_arr[t, b])
                        scene.graph.update(frame_to=f"B{short}_{k}", frame_from="world", matrix=M)
            else:
                for i in range(21):
                    scene.graph.update(frame_to=f"M{short}_{i}", frame_from="world", matrix=far_T())
                if args.bones:
                    for k in range(len(MANO_BONES)):
                        scene.graph.update(frame_to=f"B{short}_{k}", frame_from="world", matrix=far_T())

    apply_frame(0)

    def callback(_scene):
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % T
            apply_frame(state["frame"])

    class PlaybackViewer(SceneViewer):
        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.SPACE:
                state["paused"] = not state["paused"]
                print(f"[{'PAUSED' if state['paused'] else 'PLAY'}] {state['frame']}/{T-1}")
            elif symbol == pyglet.window.key.LEFT:
                state["paused"] = True
                state["frame"] = max(0, state["frame"] - 1)
                apply_frame(state["frame"])
                print(f"frame {state['frame']}/{T-1}")
            elif symbol == pyglet.window.key.RIGHT:
                state["paused"] = True
                state["frame"] = min(T - 1, state["frame"] + 1)
                apply_frame(state["frame"])
                print(f"frame {state['frame']}/{T-1}")
            elif symbol == pyglet.window.key.HOME:
                state["frame"] = 0
                apply_frame(state["frame"])
                print(f"frame 0/{T-1}")
            elif symbol == pyglet.window.key.END:
                state["frame"] = T - 1
                apply_frame(state["frame"])
                print(f"frame {state['frame']}/{T-1}")
            else:
                super().on_key_press(symbol, modifiers)

    print("Controls: SPACE=pause LEFT/RIGHT=step HOME/END=jump  drag=rotate scroll=zoom")
    PlaybackViewer(scene, callback=callback, callback_period=1.0 / args.fps,
                   start_loop=True)


if __name__ == "__main__":
    main()
