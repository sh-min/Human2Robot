"""Static Open3D view of the first frame: MANO mesh + retargeted xhand mesh
+ optimized cube — all in cam frame.

Usage:
    conda activate RFM_retarget
    cd <repo_root>/src/simulation_tool
    python inspect_cube_pose.py \
        --npz       <seq>/rgb_hawor/retarget_input.npz \
        --pkl       <seq>/rgb_hawor/qpos_xhand_contact_left.pkl \
        --cube_pose <seq>/rgb_hawor/cube_pose.npz
"""
import argparse
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rscipy

# Borrow URDF_ROOT / R_MANO_XHAND from the retargeting module so we don't
# duplicate path constants.
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_RETARGET_DIR = os.path.abspath(os.path.join(_SIM_DIR, "..", "retargeting"))
sys.path.insert(0, _RETARGET_DIR)
from _paths import URDF_ROOT, R_MANO_XHAND as _R_DICT

ASSETS = os.path.join(_RETARGET_DIR, "assets")


def load_xhand_link_meshes(hand):
    urdf = os.path.join(URDF_ROOT, "xhand", f"xhand_{hand}.urdf")
    tree = ET.parse(urdf)
    mesh_dir = Path(urdf).parent
    out = []
    for link in tree.getroot().findall("link"):
        nm = link.attrib["name"]
        v = link.find("visual")
        if v is None: continue
        mn = v.find("geometry/mesh")
        if mn is None: continue
        m = o3d.io.read_triangle_mesh(str(mesh_dir / mn.attrib["filename"]))
        out.append((nm,
                    np.asarray(m.vertices, dtype=np.float64),
                    np.asarray(m.triangles, dtype=np.int32)))
    return out


def build_combined(link_meshes):
    Vs, Fs, ranges, names = [], [], [], []
    off = 0
    for nm, v, f in link_meshes:
        Vs.append(v)
        Fs.append(f + off)
        ranges.append((off, off + len(v)))
        names.append(nm)
        off += len(v)
    return np.concatenate(Vs, 0), np.concatenate(Fs, 0), ranges, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--cube_pose", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--hand", default="left", choices=["left", "right"])
    args = ap.parse_args()

    data = np.load(args.npz)
    verts = data[f"verts_{args.hand}"][args.frame].astype(np.float64)

    cp = np.load(args.cube_pose)
    cube_center = cp["center"]
    cube_rotvec = cp["rotvec"]
    cube_size = float(cp["size"])
    print(f"cube center={cube_center}  rotvec={cube_rotvec}  size={cube_size}  "
          f"IoU={float(cp['iou']):.3f}  contact_loss={float(cp['contact_loss']):.5f}")

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    q = np.asarray(d["data"][args.frame])
    joint_names = d["joint_names"]

    if "wrist_pos" in d and "wrist_quat" in d:
        wrist_pos = np.asarray(d["wrist_pos"][args.frame])
        qxyzw = np.asarray(d["wrist_quat"][args.frame])
        R_cam_xhand = Rscipy.from_quat(qxyzw).as_matrix()
        src = "pkl"
    else:
        mano_root = data["mano_global_orient"][0 if args.hand == "left" else 1][args.frame]
        R_cam_xhand = Rscipy.from_rotvec(mano_root).as_matrix() @ _R_DICT[args.hand]
        wrist_pos = data[f"joints_{args.hand}"][args.frame, 0]
        src = "npz"
    print(f"wrist source: {src}")

    # xhand FK
    urdf = os.path.join(URDF_ROOT, "xhand", f"xhand_{args.hand}.urdf")
    model = pin.buildModelFromUrdf(urdf)
    pdata = model.createData()
    pin_names = [model.names[i] for i in range(1, len(model.names))]
    map_pin = np.array([joint_names.index(n) for n in pin_names], dtype=int)
    pin.forwardKinematics(model, pdata, q[map_pin])
    pin.updateFramePlacements(model, pdata)

    V_loc, F, ranges, link_names = build_combined(load_xhand_link_meshes(args.hand))
    base_T = np.eye(4)
    base_T[:3, :3] = R_cam_xhand
    base_T[:3, 3] = wrist_pos

    V_world = V_loc.copy()
    for (s, e), nm in zip(ranges, link_names):
        if not model.existFrame(nm):
            continue
        fid = model.getFrameId(nm)
        T = np.eye(4)
        T[:3, :3] = np.asarray(pdata.oMf[fid].rotation)
        T[:3, 3] = np.asarray(pdata.oMf[fid].translation)
        T_cam = base_T @ T
        V_world[s:e] = V_loc[s:e] @ T_cam[:3, :3].T + T_cam[:3, 3]

    xhand_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V_world),
        o3d.utility.Vector3iVector(F.astype(np.int32)),
    )
    xhand_mesh.paint_uniform_color([0.40, 0.75, 0.92])
    xhand_mesh.compute_vertex_normals()

    mano_faces = np.load(os.path.join(ASSETS, f"mano_faces_{args.hand}.npy")).astype(np.int32)
    mano_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(mano_faces),
    )
    mano_mesh.paint_uniform_color([0.80, 0.80, 0.80])
    mano_mesh.compute_vertex_normals()

    # Cube at optimized pose
    cube = o3d.geometry.TriangleMesh.create_box(cube_size, cube_size, cube_size)
    cube.translate(np.array([-cube_size / 2] * 3))  # center on origin
    R_cube = Rscipy.from_rotvec(cube_rotvec).as_matrix()
    Tc = np.eye(4); Tc[:3, :3] = R_cube; Tc[:3, 3] = cube_center
    cube.transform(Tc)
    cube.paint_uniform_color([0.95, 0.55, 0.20])
    cube.compute_vertex_normals()

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)

    o3d.visualization.draw_geometries(
        [mano_mesh, xhand_mesh, cube, axes],
        window_name=f"cube_pose  frame={args.frame}  hand={args.hand}",
    )


if __name__ == "__main__":
    main()
