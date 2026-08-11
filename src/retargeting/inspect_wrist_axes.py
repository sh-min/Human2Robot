"""Visualize the xhand at q=0 with the wrist-link axis arrows so we can
read off the URDF's frame convention by eye.

Usage:
    conda activate RFM_retarget

    # Interactive (needs display):
    python inspect_wrist_axes.py --hand right

    # Headless multi-view PNG:
    python inspect_wrist_axes.py --hand right --save /tmp/xhand_right_axes.png
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import trimesh
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


from _paths import URDF_ROOT
URDF_DIR = os.path.join(URDF_ROOT, "xhand")


def load_meshes_at_q0(urdf_path):
    """Returns list of (verts_world, faces) for all visual meshes at q=0."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mesh_dir = Path(urdf_path).parent

    out = []
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
            mesh = trimesh.load(str(mfile), force="mesh")
        except Exception:
            continue
        if not model.existFrame(name):
            T_mat = np.eye(4)
        else:
            fid = model.getFrameId(name)
            T_mat = np.eye(4)
            T_mat[:3, :3] = np.array(data.oMf[fid].rotation)
            T_mat[:3, 3] = np.array(data.oMf[fid].translation)
        v = np.asarray(mesh.vertices)
        v_h = np.hstack([v, np.ones((len(v), 1))])
        v_w = (T_mat @ v_h.T).T[:, :3]
        out.append((v_w, np.asarray(mesh.faces)))
    return out, model, data


def get_link_pose(model, data, link_name):
    fid = model.getFrameId(link_name)
    return np.array(data.oMf[fid].rotation), np.array(data.oMf[fid].translation)


def draw_scene(ax, meshes, R_link, t_link, axis_len=0.06, mesh_step=15):
    for v, _ in meshes:
        s = max(1, len(v) // mesh_step // 100)
        ax.scatter(v[::s, 0], v[::s, 1], v[::s, 2],
                   s=1, c="lightgray", alpha=0.4)
    for i, (color, label) in enumerate([("red", "X"), ("green", "Y"), ("blue", "Z")]):
        end = t_link + R_link[:, i] * axis_len
        ax.plot([t_link[0], end[0]], [t_link[1], end[1]], [t_link[2], end[2]],
                color=color, linewidth=4)
        tip = t_link + R_link[:, i] * axis_len * 1.15
        ax.text(tip[0], tip[1], tip[2], label, color=color,
                fontsize=14, fontweight="bold")
    ax.scatter([t_link[0]], [t_link[1]], [t_link[2]], c="black", s=30)


def style(ax, title, lim=0.22):
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("world X"); ax.set_ylabel("world Y"); ax.set_zlabel("world Z")
    ax.set_title(title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--save", default=None,
                    help="If given, save 4-view PNG instead of interactive show()")
    args = ap.parse_args()

    urdf = os.path.join(URDF_DIR, f"xhand_{args.hand}.urdf")
    link = f"{args.hand}_hand_link"
    print(f"Loading {urdf}, axis frame = {link}")

    meshes, model, data = load_meshes_at_q0(urdf)
    R_link, t_link = get_link_pose(model, data, link)
    print(f"{link} pose at q=0:")
    print(f"  origin (world) = {t_link}")
    print(f"  rotation  = \n{R_link}")
    print()
    print("xhand finger TIP positions (in wrist-link frame), mm:")
    Winv = np.linalg.inv(np.block([[R_link, t_link[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
    for n in [f"{args.hand}_hand_thumb_rota_tip",
              f"{args.hand}_hand_index_rota_tip",
              f"{args.hand}_hand_mid_tip",
              f"{args.hand}_hand_ring_tip",
              f"{args.hand}_hand_pinky_tip"]:
        fid = model.getFrameId(n)
        T = np.eye(4)
        T[:3, :3] = np.array(data.oMf[fid].rotation)
        T[:3, 3] = np.array(data.oMf[fid].translation)
        rel = (Winv @ T)[:3, 3] * 1000
        print(f"  {n:35s} ({rel[0]:+7.1f}, {rel[1]:+7.1f}, {rel[2]:+7.1f})")

    if args.save:
        fig = plt.figure(figsize=(14, 10))
        views = [("front (look -X)",  0,   0),
                 ("right (look +Y)", -90,  0),
                 ("top (look -Z)",   -90, 89),
                 ("iso",              45, 30)]
        for k, (name, az, el) in enumerate(views):
            ax = fig.add_subplot(2, 2, k + 1, projection="3d")
            draw_scene(ax, meshes, R_link, t_link)
            ax.view_init(elev=el, azim=az)
            style(ax, f"{args.hand}: {name}  (azim={az}, elev={el})")
        plt.suptitle(f"xhand {args.hand} — frame at {link}  "
                     "(X=red, Y=green, Z=blue)", fontsize=14)
        plt.tight_layout()
        plt.savefig(args.save, dpi=140, bbox_inches="tight")
        print(f"\nsaved {args.save}")
    else:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_scene(ax, meshes, R_link, t_link)
        style(ax, f"xhand {args.hand} — drag to rotate")
        plt.show()


if __name__ == "__main__":
    main()
