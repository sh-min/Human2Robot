"""Open3D 3D viewer that overlays three things in the same cam-frame so
stage-1 vs stage-2 retargeting can be compared:
    1) MANO mesh         (deforming surface from npz verts_right)
    2) xhand stage 1     (full mesh via FK from qpos_xhand_right.pkl)
    3) xhand stage 2     (full mesh via FK from qpos_xhand_contact_right.pkl)
    4) Contact verts     (palmar + fingertip filtered, yellow dots)

Right hand only.

Controls (focus the Open3D window):
    SPACE   pause / play
    [ / ]   step -1 / +1 frame (auto-pauses)
    , / .   jump -10 / +10
    H       jump first ; E jump last
    M       toggle MANO mesh visibility
    1       toggle xhand stage-1 visibility
    2       toggle xhand stage-2 visibility
    C       toggle contact dots visibility

Usage:
    conda activate RFM_retarget
    cd <repo_root>/src/retargeting
    python compare_stages.py \
        --npz       /path/to/<seq>_hawor/retarget_input.npz \
        --pkl1      /path/to/<seq>_hawor/qpos_xhand_right.pkl \
        --pkl2      /path/to/<seq>_hawor/qpos_xhand_contact_right.pkl \
        --contact_dir /path/to/<seq>/contact
"""
import argparse
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rscipy

from _paths import URDF_ROOT, R_MANO_XHAND as _R_DICT

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
HAND = "right"
NV = 778

R_MANO_XHAND = _R_DICT[HAND].astype(np.float64)


def load_xhand_link_meshes(hand):
    """Load every visual mesh in the xhand URDF; return list of
    (link_name, vertices_local (Nv,3), faces (Nf,3))."""
    urdf = os.path.join(URDF_ROOT, "xhand", f"xhand_{hand}.urdf")
    tree = ET.parse(urdf)
    mesh_dir = Path(urdf).parent
    out = []
    for link in tree.getroot().findall("link"):
        name = link.attrib["name"]
        v = link.find("visual")
        if v is None:
            continue
        mn = v.find("geometry/mesh")
        if mn is None:
            continue
        m = o3d.io.read_triangle_mesh(str(mesh_dir / mn.attrib["filename"]))
        verts = np.asarray(m.vertices, dtype=np.float64)
        faces = np.asarray(m.triangles, dtype=np.int32)
        out.append((name, verts, faces))
    return out


def build_combined_xhand(link_meshes):
    """Pre-build a single TriangleMesh that concatenates all links. Returns
    (mesh, link_vert_ranges, link_names) where link_vert_ranges[i] = (start, end)
    so we can update each link's vertex slice via FK."""
    all_v, all_f, ranges, names = [], [], [], []
    offset = 0
    for name, v, f in link_meshes:
        all_v.append(v)
        all_f.append(f + offset)
        ranges.append((offset, offset + len(v)))
        names.append(name)
        offset += len(v)
    V = np.concatenate(all_v, axis=0)
    F = np.concatenate(all_f, axis=0)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V),
        o3d.utility.Vector3iVector(F),
    )
    return mesh, ranges, names, V


def fk_link_transforms(model, data, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T = {}
    for fid in range(model.nframes):
        nm = model.frames[fid].name
        Tm = np.eye(4)
        Tm[:3, :3] = np.asarray(data.oMf[fid].rotation)
        Tm[:3, 3] = np.asarray(data.oMf[fid].translation)
        T[nm] = Tm
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pkl1", required=True, help="xhand stage-1 qpos pkl (right)")
    ap.add_argument("--pkl2", required=True, help="xhand stage-2 qpos pkl (right)")
    ap.add_argument("--contact_dir", default=None,
                    help="Default: <npz parent>/../contact")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.npz)
    joints_world = data[f"joints_{HAND}"]    # (T, 21, 3)
    verts_world = data[f"verts_{HAND}"]      # (T, 778, 3)
    mano_root = data["mano_global_orient"][1]      # right
    valid = data["valid"][1]
    start_idx_data = int(data["start_idx"])
    T = verts_world.shape[0]

    npz_dir = os.path.dirname(os.path.abspath(args.npz))
    contact_dir = args.contact_dir or os.path.join(os.path.dirname(npz_dir), "contact")
    print(f"T={T}  start_idx={start_idx_data}  contact_dir={contact_dir}")

    with open(args.pkl1, "rb") as f:
        d1 = pickle.load(f)
    with open(args.pkl2, "rb") as f:
        d2 = pickle.load(f)
    q1 = np.asarray(d1["data"])           # (T, 12) dex-retargeting joint order
    q2 = np.asarray(d2["data"])

    # MANO faces + palmar/tip masks + finger_part for filtering contact dots
    mano_faces = np.load(os.path.join(ASSETS, f"mano_faces_{HAND}.npy")).astype(np.int32)
    palmar = np.load(os.path.join(ASSETS, f"palmar_mask_{HAND}.npy")).astype(bool)
    tip_mask = np.load(os.path.join(ASSETS, f"fingertip_mask_{HAND}.npy")).astype(bool)

    # pinocchio for xhand FK
    urdf = os.path.join(URDF_ROOT, "xhand", f"xhand_{HAND}.urdf")
    model = pin.buildModelFromUrdf(urdf)
    pdata = model.createData()
    pin_names = [model.names[i] for i in range(1, len(model.names))]
    map_pin = np.array([d1["joint_names"].index(n) for n in pin_names], dtype=int)

    link_meshes = load_xhand_link_meshes(HAND)
    xhand_mesh_1, ranges, link_names, V_local = build_combined_xhand(link_meshes)
    xhand_mesh_2 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V_local.copy()),
        o3d.utility.Vector3iVector(np.asarray(xhand_mesh_1.triangles)),
    )
    xhand_mesh_1.paint_uniform_color([0.40, 0.78, 0.92])    # cyan
    xhand_mesh_2.paint_uniform_color([0.95, 0.45, 0.45])    # red

    mano_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_world[0].copy()),
        o3d.utility.Vector3iVector(mano_faces),
    )
    mano_mesh.paint_uniform_color([0.78, 0.78, 0.78])

    contact_pc = o3d.geometry.PointCloud()
    contact_pc.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    contact_pc.paint_uniform_color([1.0, 0.85, 0.10])

    # Coordinate axes at camera origin
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # ----- Update logic -----
    def world_to_cam_xhand_base(t):
        """Returns 4x4: xhand wrist link pose in cam frame."""
        R_root = Rscipy.from_rotvec(mano_root[t]).as_matrix()
        R_cam_xh = R_root @ R_MANO_XHAND
        Tb = np.eye(4)
        Tb[:3, :3] = R_cam_xh
        Tb[:3, 3] = joints_world[t, 0]    # actual wrist position
        return Tb

    def update_xhand_mesh(mesh, qpos_dex, base_T):
        q_pin = qpos_dex[map_pin]
        Tdict = fk_link_transforms(model, pdata, q_pin)
        new_V = V_local.copy()
        for (s, e), name in zip(ranges, link_names):
            if name not in Tdict:
                continue
            T_cam = base_T @ Tdict[name]
            v_loc = V_local[s:e]
            new_V[s:e] = (v_loc @ T_cam[:3, :3].T) + T_cam[:3, 3]
        mesh.vertices = o3d.utility.Vector3dVector(new_V)
        mesh.compute_vertex_normals()

    def load_contact_mask(t):
        path = os.path.join(contact_dir,
                            f"rgb_frame{start_idx_data + t:05d}.npz")
        if not os.path.exists(path):
            return np.zeros(NV, bool)
        d = np.load(path)
        k = f"{HAND}_contact_mask"
        if k not in d.files:
            return np.zeros(NV, bool)
        return d[k].astype(bool)

    state = {
        "frame": max(0, min(T - 1, args.start)),
        "paused": False,
        "show_mano": True, "show_1": True, "show_2": True, "show_contact": True,
    }

    def apply(t):
        if valid[t]:
            mano_mesh.vertices = o3d.utility.Vector3dVector(verts_world[t])
            mano_mesh.compute_vertex_normals()
            base_T = world_to_cam_xhand_base(t)
            update_xhand_mesh(xhand_mesh_1, q1[t], base_T)
            update_xhand_mesh(xhand_mesh_2, q2[t], base_T)
            # Contact dots
            cm = load_contact_mask(t) & palmar & tip_mask
            if cm.any():
                pts = verts_world[t][cm]
            else:
                pts = np.zeros((0, 3))
            contact_pc.points = o3d.utility.Vector3dVector(pts)
        else:
            mano_mesh.vertices = o3d.utility.Vector3dVector(np.zeros((NV, 3)))
            contact_pc.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    apply(state["frame"])

    # ----- Viewer + key callbacks -----
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"compare stages (right) — T={T}",
                      width=1200, height=900)
    for g in (mano_mesh, xhand_mesh_1, xhand_mesh_2, contact_pc, axes):
        vis.add_geometry(g)

    ro = vis.get_render_option()
    ro.mesh_show_back_face = True
    ro.point_size = 8.0
    ro.background_color = np.array([0.07, 0.08, 0.10])
    ro.light_on = True

    visible = {"mano": True, "1": True, "2": True, "contact": True}

    def set_visible(name, mesh, on):
        if visible[name] == on:
            return
        if on:
            vis.add_geometry(mesh, reset_bounding_box=False)
        else:
            vis.remove_geometry(mesh, reset_bounding_box=False)
        visible[name] = on

    def refresh():
        if visible["mano"]:    vis.update_geometry(mano_mesh)
        if visible["1"]:       vis.update_geometry(xhand_mesh_1)
        if visible["2"]:       vis.update_geometry(xhand_mesh_2)
        if visible["contact"]: vis.update_geometry(contact_pc)

    def status():
        return (f"frame {state['frame']}/{T-1}  "
                f"M={'on' if visible['mano'] else 'off'}  "
                f"1={'on' if visible['1'] else 'off'}  "
                f"2={'on' if visible['2'] else 'off'}  "
                f"C={'on' if visible['contact'] else 'off'}  "
                f"{'PAUSED' if state['paused'] else 'PLAY'}")

    def step(delta, set_pause=True):
        if set_pause:
            state["paused"] = True
        state["frame"] = max(0, min(T - 1, state["frame"] + delta))
        apply(state["frame"])
        refresh()
        print(status())
        return False

    def on_space(_):
        state["paused"] = not state["paused"]
        print(status()); return False

    def on_left(_):  return step(-1)
    def on_right(_): return step(+1)
    def on_back10(_): return step(-10)
    def on_fwd10(_):  return step(+10)
    def on_home(_):
        state["frame"] = 0; apply(0); refresh(); print(status()); return False
    def on_end(_):
        state["frame"] = T - 1; apply(T - 1); refresh(); print(status()); return False
    def on_m(_): set_visible("mano",   mano_mesh,    not visible["mano"]); print(status()); return False
    def on_1(_): set_visible("1",      xhand_mesh_1, not visible["1"]);    print(status()); return False
    def on_2(_): set_visible("2",      xhand_mesh_2, not visible["2"]);    print(status()); return False
    def on_c(_): set_visible("contact", contact_pc,  not visible["contact"]); print(status()); return False

    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("["), on_left)
    vis.register_key_callback(ord("]"), on_right)
    vis.register_key_callback(ord(","), on_back10)
    vis.register_key_callback(ord("."), on_fwd10)
    vis.register_key_callback(ord("H"), on_home)
    vis.register_key_callback(ord("E"), on_end)
    vis.register_key_callback(ord("M"), on_m)
    vis.register_key_callback(ord("1"), on_1)
    vis.register_key_callback(ord("2"), on_2)
    vis.register_key_callback(ord("C"), on_c)

    # Animation loop via register_animation_callback
    period = 1.0 / args.fps
    import time
    last = [time.time()]

    def animate(_):
        now = time.time()
        if not state["paused"] and (now - last[0]) >= period:
            state["frame"] = (state["frame"] + 1) % T
            apply(state["frame"])
            refresh()
            last[0] = now
        return False

    vis.register_animation_callback(animate)

    print("Controls: SPACE=pause [/]=±1 frame ,/.=±10  H/E=first/last  "
          "M/1/2/C=toggle visibility")
    print(status())
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
