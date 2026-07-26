"""Shared xhand render helpers: embodiment resolution, URDF/MJCF parse, forward
kinematics, and the CV->GL frame flip. Imported by render_xhand_overlay_depth.py.

This module used to also host a standalone renderer that drew the robot over the
raw video clipped to the SAM2 arm mask (`draw = robot_mask ∧ arm_mask`) and wrote
a residual mask for a follow-up inpaint. That clip/residual path was run.py and
has been retired in favour of the depth-aware layered pipeline (run_layered.py),
which never clips the robot to the human silhouette. Only the helpers remain.

Coordinate conventions:
  MANO cam space:   x=right, y=down, z=forward (OpenCV)
  pyrender world:   OpenGL — camera at identity looks along -z, y=up
  T_CV2GL = diag(1,-1,-1):
      t_pr = T_CV2GL @ t_cam              (positions)
      R_pr = T_CV2GL @ R_cam_xhand        (orientations — input frame unchanged)
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from _paths import (DEFAULT_EMBODIMENT, EMBODIMENT_NAMES,
                    load_R_mano, load_wrist_offset, mjcf_for, urdf_for)
T_CV2GL = np.diag([1., -1., -1.])

REPO = Path(__file__).resolve().parent.parent.parent
XHAND_XML = {
    "right": REPO / "src/sim/mujoco_sim/assets/xhand_right/xhand_right.xml",
    "left":  REPO / "src/sim/mujoco_sim/assets/xhand_left/xhand_left.xml",
}


def resolve_embodiment(pkl_dict, override, side):
    """Which robot hand this pkl describes.

    Retargeting stamps `embodiment` into the pkl, so the renderer normally does
    not need telling — that stops an inspire trajectory being drawn with an
    xhand URDF. An explicit CLI override wins; pkls written before the field
    existed fall back to xhand.
    """
    if override:
        stamped = pkl_dict.get("embodiment")
        if stamped and stamped != override:
            print(f"[warn] {side}: --{side}_embodiment={override} overrides "
                  f"embodiment={stamped!r} recorded in the pkl")
        return override
    return pkl_dict.get("embodiment", DEFAULT_EMBODIMENT)


def build_side_align(embodiment_of):
    """Per-side (R_align, wrist_offset): the embodiment-specific part of placing
    the hand root, kept out of the render loops."""
    return {s: (load_R_mano(e, s), load_wrist_offset(e, s))
            for s, e in embodiment_of.items()}


def hand_root_pose(R_mano, wrist_pos, align):
    """Hand root pose in pyrender frame.

    The URDF root is put at the MANO wrist, shifted by the embodiment's wrist
    offset (expressed in the wrist frame, hence rotated into camera frame
    first). The offset is zero for xhand, whose root already sits near MANO's
    wrist; inspire's root is the arm mount flange ~54 mm further out.
    """
    R_align, offset = align
    R_cam_hand = R_mano @ R_align
    t_cam = wrist_pos + R_cam_hand @ offset
    T = np.eye(4)
    T[:3, :3] = T_CV2GL @ R_cam_hand
    T[:3, 3] = T_CV2GL @ t_cam
    return T, R_cam_hand


def load_side_urdf(embodiment, side):
    """(joints, link_meshes) for one hand, coloured from its MJCF if it has one."""
    mjcf = mjcf_for(embodiment, side)
    cmap = parse_mjcf_rgba(Path(mjcf)) if mjcf else None
    return parse_urdf(Path(urdf_for(embodiment, side)), color_map=cmap)


def parse_mjcf_rgba(mjcf_path: Path) -> dict:
    """Parse an xhand MJCF XML and return {mesh_name: [r,g,b,a] uint8}."""
    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()
    cmap = {}
    def _walk(elem):
        for g in elem.findall("geom"):
            mesh = g.get("mesh", "")
            rgba = g.get("rgba", "")
            if mesh and rgba:
                cmap[mesh] = [int(float(v) * 255) for v in rgba.split()]
        for b in elem.findall("body"):
            _walk(b)
    _walk(root.find("worldbody"))
    return cmap


def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def parse_urdf(urdf_path: Path, color_map: dict | None = None):
    """Return (joints, link_meshes) from a URDF. Manual parse to avoid urdfpy
    (incompatible with Python 3.10's collections.Mapping removal).

    If *color_map* is provided ({mesh_name: [r,g,b,a]}), mesh colors are
    looked up by the URDF link name (which matches the MJCF mesh name).
    Falls back to mesh's embedded material if no entry is found.
    """
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
        lk_name = lk.get("name")
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
            if color_map and lk_name in color_map:
                m.visual.face_colors = color_map[lk_name]

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
            link_meshes[lk_name] = items

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


