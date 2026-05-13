"""Render retargeted xhand hands onto the inpainted video (pyrender + trimesh).

Replicates `src/retargeting/overlay_on_rgb.py`'s placement math (R_MANO_XHAND,
wrist at MANO joint 0) but uses pyrender instead of Sapien so it runs in the
phantom conda env without an extra dep.

Coordinate conventions:
  MANO cam space:   x=right, y=down, z=forward (OpenCV)
  pyrender world:   OpenGL — camera at identity looks along -z, y=up
  T_CV2GL = diag(1,-1,-1):
      t_pr = T_CV2GL @ t_cam              (positions)
      R_pr = T_CV2GL @ R_cam_xhand        (orientations — input frame unchanged)

Wrist placement:
  position    = joints_{left,right}[t, 0, :]   (MANO joint 0 = wrist)
  orientation = R_from_rotvec(mano_global_orient[t]) @ R_MANO_XHAND[side]

Finger angles:
  From qpos_xhand_{left,right}.pkl (12-DOF, pre-retargeted by DexPilot).

Usage:
    PYOPENGL_PLATFORM=egl python -u render_xhand_overlay.py \
        --processed_demo /result/cam0_inpaint/cam0/0 \
        --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz \
        --right_pkl /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl  /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl \
        --hand both
"""
import argparse
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mediapy as media
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

from _paths import XHAND_URDF_LEFT, XHAND_URDF_RIGHT, R_MANO_XHAND
T_CV2GL = np.diag([1., -1., -1.])


def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def parse_urdf(urdf_path: Path):
    """Return (joints, link_meshes) from a URDF. Manual parse to avoid urdfpy
    (incompatible with Python 3.10's collections.Mapping removal)."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
        axis_el = j.find("axis")
        axis = (np.array([float(v) for v in axis_el.get("xyz").split()])
                if axis_el is not None else np.array([1., 0., 0.]))
        joints[j.get("name")] = dict(
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            xyz=xyz, rpy=rpy, axis=axis,
        )

    link_meshes = {}
    for lk in root.findall("link"):
        items = []
        for vis in lk.findall("visual"):
            geom = vis.find("geometry")
            if geom is None: continue
            msh = geom.find("mesh")
            if msh is None: continue
            mesh_path = urdf_path.parent / msh.get("filename")
            if not mesh_path.exists(): continue
            m = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(list(m.geometry.values()))
            m.visual.face_colors = [200, 170, 130, 255]  # skin tone

            orig = vis.find("origin")
            if orig is not None:
                T_vis = _make_T(
                    np.array([float(v) for v in orig.get("xyz", "0 0 0").split()]),
                    np.array([float(v) for v in orig.get("rpy", "0 0 0").split()]),
                )
            else:
                T_vis = np.eye(4)
            items.append((m, T_vis))
        if items:
            link_meshes[lk.get("name")] = items

    return joints, link_meshes


def compute_fk(joints: dict, qpos_dict: dict, root_T: np.ndarray) -> dict:
    """BFS forward kinematics from URDF joint tree."""
    all_children = {j["child"] for j in joints.values()}
    all_parents  = {j["parent"] for j in joints.values()}
    root_link = next(iter(all_parents - all_children))

    link_T = {root_link: root_T}
    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        T_par = link_T[parent]
        for jname, jd in joints.items():
            if jd["parent"] != parent:
                continue
            T_origin = _make_T(jd["xyz"], jd["rpy"])
            R_j = np.eye(4)
            if jd["type"] == "revolute":
                angle = qpos_dict.get(jname, 0.0)
                R_j[:3, :3] = Rotation.from_rotvec(jd["axis"] * angle).as_matrix()
            link_T[jd["child"]] = T_par @ T_origin @ R_j
            queue.append(jd["child"])
    return link_T


def render_hands(
    bg: np.ndarray,
    hands_data: list,    # (side, wrist_pos_cam, R_mano_cam, qpos_dict, joints_tree, link_meshes)
    camera: pyrender.IntrinsicsCamera,
    renderer: pyrender.OffscreenRenderer,
) -> np.ndarray:
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0., 0., 0., 0.])
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0), pose=np.eye(4))
    pl_pose = np.eye(4); pl_pose[:3, 3] = [0.3, -0.3, -0.5]
    scene.add(pyrender.PointLight(color=[0.8, 0.8, 0.8], intensity=2.0), pose=pl_pose)

    for side, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes in hands_data:
        # R_cam_xhand maps xhand-local → MANO cam. Apply T_CV2GL to the OUTPUT
        # only (xhand local frame stays unchanged).
        R_cam_xhand = R_mano @ R_MANO_XHAND[side]
        R_pr = T_CV2GL @ R_cam_xhand
        t_pr = T_CV2GL @ wrist_pos

        T_root = np.eye(4); T_root[:3, :3] = R_pr; T_root[:3, 3] = t_pr
        link_T = compute_fk(joints_tree, qpos_dict, T_root)

        for lname, items in link_meshes.items():
            if lname not in link_T:
                continue
            for mesh, T_vis in items:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                scene.add(pr_mesh, pose=link_T[lname] @ T_vis)

    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = depth > 0
    out = bg.copy()
    out[mask] = color[mask, :3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True,
                    help="Phantom processed demo folder")
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output mp4/mkv path (default: <processed_demo>/video_overlay_xhand.mkv)")
    args = ap.parse_args()

    ri = np.load(args.hawor_npz)
    joints_left  = ri["joints_left"].astype(np.float64)
    joints_right = ri["joints_right"].astype(np.float64)
    go    = ri["mano_global_orient"]   # (2, T, 3) axis-angle
    valid = ri["valid"]
    focal = float(ri["img_focal"])

    qr = pickle.load(open(args.right_pkl, "rb"))
    ql = pickle.load(open(args.left_pkl,  "rb"))
    right_data  = np.asarray(qr["data"])
    left_data   = np.asarray(ql["data"])
    right_jname = qr["joint_names"]
    left_jname  = ql["joint_names"]

    # Background: prefer inpainted, fall back to raw
    bg_path = args.processed_demo / "inpaint_processor" / "video_human_inpaint.mkv"
    if not bg_path.exists():
        bg_path = args.processed_demo / "video_L.mp4"
    print(f"[bg] {bg_path}")
    bg_frames = media.read_video(str(bg_path))
    T_vid = bg_frames.shape[0]
    img_h, img_w = bg_frames.shape[1], bg_frames.shape[2]
    T_use = min(T_vid, joints_left.shape[0])
    print(f"[info] video T={T_vid}, npz T={joints_left.shape[0]}, using T={T_use}, hand={args.hand}")

    cx, cy = img_w / 2.0, img_h / 2.0
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy,
                                       znear=0.01, zfar=10.0)
    renderer = pyrender.OffscreenRenderer(img_w, img_h)

    side_cfg = {}
    for s, urdf in (("right", XHAND_URDF_RIGHT), ("left", XHAND_URDF_LEFT)):
        if args.hand not in (s, "both"):
            continue
        side_cfg[s] = parse_urdf(Path(urdf))
    print("URDF loaded for:", list(side_cfg.keys()))

    out_frames = list(bg_frames[:T_use])
    for t in range(T_use):
        hands_data = []
        for s in ("right", "left"):
            if s not in side_cfg:
                continue
            h_idx = 1 if s == "right" else 0
            if not valid[h_idx, t]:
                continue
            joints_tree, link_meshes = side_cfg[s]
            jnames = right_jname if s == "right" else left_jname
            qdata  = right_data[t] if s == "right" else left_data[t]
            qpos_dict = {jn: float(qdata[i]) for i, jn in enumerate(jnames)}
            wrist_pos = (joints_right if s == "right" else joints_left)[t, 0, :]
            R_mano = Rotation.from_rotvec(go[h_idx, t]).as_matrix()
            hands_data.append((s, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes))

        if hands_data:
            out_frames[t] = render_hands(bg_frames[t], hands_data, camera, renderer)
        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T_use}")

    renderer.delete()

    out = args.out or (args.processed_demo / "video_overlay_xhand.mkv")
    media.write_video(str(out), np.stack(out_frames), fps=10, codec="ffv1")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
