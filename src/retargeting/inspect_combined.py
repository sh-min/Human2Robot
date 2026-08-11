"""Show xhand mesh (q=0) + human MANO 21-joint skeleton (canonical frame) for
both hands in the same world frame, so the wrist-frame mismatch is visible
by eye.

Usage:
    conda activate RFM_retarget

    # Interactive (needs display via X11/VNC):
    python inspect_combined.py --npz /path/to/retarget_input.npz --frame 0

    # Headless PNG fallback:
    python inspect_combined.py --npz /path/to/retarget_input.npz --save /tmp/combined.png
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial.transform import Rotation as Rscipy

from _paths import URDF_ROOT
URDF_DIR = os.path.join(URDF_ROOT, "xhand")

# HaWoR MANO outputs joints already in mediapipe-21 layout:
#   0 = wrist
#   1-4 = thumb (1=cmc, 2=mcp, 3=ip, 4=tip)
#   5-8 = index (mcp, pip, dip, tip)
#   9-12 = middle
#   13-16 = ring
#   17-20 = pinky
MANO_BONES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),     # palm: wrist to each finger base
    (1, 2), (2, 3), (3, 4),                       # thumb chain
    (5, 6), (6, 7), (7, 8),                       # index chain
    (9, 10), (10, 11), (11, 12),                  # middle chain
    (13, 14), (14, 15), (15, 16),                 # ring chain
    (17, 18), (18, 19), (19, 20),                 # pinky chain
]
JOINT_LABELS = [
    "wrist",
    "thb_cmc", "thb_mcp", "thb_ip", "thb_tip",
    "idx_mcp", "idx_pip", "idx_dip", "idx_tip",
    "mid_mcp", "mid_pip", "mid_dip", "mid_tip",
    "rng_mcp", "rng_pip", "rng_dip", "rng_tip",
    "pky_mcp", "pky_pip", "pky_dip", "pky_tip",
]
JOINT_COLORS = {
    "thumb":  [220,  60,  60, 255],   # red
    "index":  [60,  170, 220, 255],   # blue
    "middle": [80,  200,  90, 255],   # green
    "ring":   [220, 160,  60, 255],   # orange
    "pinky":  [180,  80, 200, 255],   # purple
    "wrist":  [40,   40,  40, 255],   # black
}
def joint_color(idx):
    if idx == 0: return JOINT_COLORS["wrist"]
    if 1 <= idx <= 4:   return JOINT_COLORS["thumb"]
    if 5 <= idx <= 8:   return JOINT_COLORS["index"]
    if 9 <= idx <= 12:  return JOINT_COLORS["middle"]
    if 13 <= idx <= 16: return JOINT_COLORS["ring"]
    if 17 <= idx <= 20: return JOINT_COLORS["pinky"]
    return [128, 128, 128, 255]


def load_xhand_meshes_at_q0(urdf_path, color):
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mesh_dir = Path(urdf_path).parent

    meshes = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        visual = link.find("visual")
        if visual is None:
            continue
        m_node = visual.find("geometry/mesh")
        if m_node is None:
            continue
        mfile = mesh_dir / m_node.attrib["filename"]
        try:
            m = trimesh.load(str(mfile), force="mesh")
        except Exception:
            continue
        if model.existFrame(name):
            fid = model.getFrameId(name)
            T = np.eye(4)
            T[:3, :3] = np.array(data.oMf[fid].rotation)
            T[:3, 3] = np.array(data.oMf[fid].translation)
            m.apply_transform(T)
        m.visual.face_colors = color
        meshes.append(m)
    return meshes


def mano_joints_canonical(npz_path, hand, frame_t):
    data = np.load(npz_path)
    h_idx = 1 if hand == "right" else 0
    j = data[f"joints_{hand}"][frame_t]
    aa = data["mano_global_orient"][h_idx][frame_t]
    R_root = Rscipy.from_rotvec(aa).as_matrix()
    rel = j - j[0:1]                              # wrist-relative in cam
    canon = (R_root.T @ rel.T).T                  # in MANO canonical
    return canon


def axis_arrows(origin, R_link, length=0.10, radius=0.0025):
    geoms = []
    colors = [[255, 0, 0, 255], [0, 220, 0, 255], [40, 80, 255, 255]]
    for i, c in enumerate(colors):
        end = origin + R_link[:, i] * length
        cyl = trimesh.creation.cylinder(radius=radius, segment=[origin, end])
        cyl.visual.face_colors = c
        geoms.append(cyl)
        # Arrow tip cone
        cone = trimesh.creation.cone(radius=radius * 2.5, height=length * 0.15)
        T = trimesh.geometry.align_vectors([0, 0, 1], R_link[:, i])
        cone.apply_transform(T)
        cone.apply_translation(end)
        cone.visual.face_colors = c
        geoms.append(cone)
    return geoms


def skeleton_geoms(joints, joint_radius=0.006, bone_radius=0.002, _unused=None):
    """Per-finger colored skeleton: one color per finger so connectivity is
    legible. Pass joints (21, 3); colors picked by `joint_color(idx)` based on
    the smplx MANO ordering above."""
    geoms = []
    for k, p in enumerate(joints):
        c = joint_color(k)
        # wrist sphere bigger
        r = joint_radius * (1.7 if k == 0 else 1.0)
        s = trimesh.creation.icosphere(radius=r, subdivisions=1)
        s.apply_translation(p)
        s.visual.face_colors = c
        geoms.append(s)
    for i, j in MANO_BONES:
        a, b = joints[i], joints[j]
        if np.linalg.norm(b - a) < 1e-6:
            continue
        # color the bone with the finger's color (the child end determines)
        c = joint_color(j) if j != 0 else joint_color(i)
        cyl = trimesh.creation.cylinder(radius=bone_radius, segment=[a, b])
        cyl.visual.face_colors = c
        geoms.append(cyl)
    return geoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--frame", type=int, default=0,
                    help="Which timeframe to take MANO joints from")
    ap.add_argument("--save", default=None,
                    help="If given, save PNG (multi-view) instead of show()")
    ap.add_argument("--separate_offset", type=float, default=0.45,
                    help="x-offset between right and left hand groups (m)")
    args = ap.parse_args()

    scene = trimesh.Scene()

    # ---- Right hand group (placed at origin) ----
    # right xhand mesh = light cyan, left xhand mesh = light salmon
    right_meshes = load_xhand_meshes_at_q0(
        os.path.join(URDF_DIR, "xhand_right.urdf"), [180, 220, 230, 220]
    )
    for m in right_meshes:
        scene.add_geometry(m)
    h_right = mano_joints_canonical(args.npz, "right", args.frame)
    for g in skeleton_geoms(h_right):
        scene.add_geometry(g)
    for g in axis_arrows(np.zeros(3), np.eye(3)):
        scene.add_geometry(g)

    # ---- Left hand group (offset along +x) ----
    left_offset = np.array([args.separate_offset, 0, 0])
    left_meshes = load_xhand_meshes_at_q0(
        os.path.join(URDF_DIR, "xhand_left.urdf"), [240, 200, 195, 220]
    )
    for m in left_meshes:
        m.apply_translation(left_offset)
        scene.add_geometry(m)
    h_left = mano_joints_canonical(args.npz, "left", args.frame) + left_offset
    for g in skeleton_geoms(h_left):
        scene.add_geometry(g)
    for g in axis_arrows(left_offset, np.eye(3)):
        scene.add_geometry(g)

    # Diagnostic print: 21 MANO joints in canonical, with magnitudes
    print("\nRight hand MANO joints (canonical, wrist-relative, mm) | magnitude:")
    for k, p in enumerate(h_right):
        mag = np.linalg.norm(p) * 1000
        print(f"  [{k:2d}] {JOINT_LABELS[k]:8s}  ({p[0]*1000:+7.1f}, {p[1]*1000:+7.1f}, {p[2]*1000:+7.1f})  |{mag:6.1f}|")

    print("\nLegend:")
    print("  xhand RIGHT mesh: light CYAN  (at world origin)")
    print("  xhand LEFT  mesh: light SALMON (offset +x)")
    print("  MANO skeleton: thumb=red, index=blue, middle=green, ring=orange, pinky=purple")
    print("  big black sphere = wrist (joint 0)")
    print("  axis arrows at xhand wrist link origin: X=red Y=green Z=blue")
    print(f"  left hand offset by +{args.separate_offset}m along world X")

    if args.save:
        # Headless PNG: render via trimesh's offscreen renderer (uses pyglet/EGL).
        try:
            png = scene.save_image(resolution=(1400, 900))
            with open(args.save, "wb") as f:
                f.write(png)
            print(f"saved {args.save}")
        except Exception as e:
            print(f"Offscreen render failed ({e}); try xvfb-run or use --no-save for interactive.")
            raise
    else:
        scene.show()


if __name__ == "__main__":
    main()
