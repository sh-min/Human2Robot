"""Render retargeted xhand hands over the raw video, clipped to the arm mask.

The robot hand is rendered on top of the RAW frames (not on an inpainted bg)
and clipped to the SAM2 arm+hand segmentation: `draw = robot_mask ∧ arm_mask`.
Any robot pixels that fall outside the human silhouette are discarded so the
robot stays inside the same screen-space region the human arm/hand occupied.
The leftover human pixels inside `arm_mask` that the robot did not cover are
written to `residual_mask.npy` for a follow-up E2FGVI pass.

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

Outputs:
    <processed_demo>/overlay_processor/video_overlay_raw.mkv   raw+robot composite
    <processed_demo>/overlay_processor/residual_mask.npy        (T,H,W) bool

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

import cv2
import mediapy as media
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

from _paths import XHAND_URDF_LEFT, XHAND_URDF_RIGHT, R_MANO_XHAND
T_CV2GL = np.diag([1., -1., -1.])

REPO = Path(__file__).resolve().parent.parent.parent
XHAND_XML = {
    "right": REPO / "src/sim/mujoco_sim/assets/xhand_right/xhand_right.xml",
    "left":  REPO / "src/sim/mujoco_sim/assets/xhand_left/xhand_left.xml",
}


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


def render_robot(
    hands_data: list,    # (side, wrist_pos_cam, R_mano_cam, qpos_dict, joints_tree, link_meshes)
    camera: pyrender.IntrinsicsCamera,
    renderer: pyrender.OffscreenRenderer,
):
    """Render the robot hand(s) and return (rgb (H,W,3) uint8, mask (H,W) bool)."""
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0., 0., 0., 0.])
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0), pose=np.eye(4))
    pl_pose = np.eye(4); pl_pose[:3, 3] = [0.3, -0.3, -0.5]
    scene.add(pyrender.PointLight(color=[0.8, 0.8, 0.8], intensity=2.0), pose=pl_pose)

    for side, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes in hands_data:
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
    return color[..., :3], mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True,
                    help="Phantom processed demo folder")
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--mask_dilate", type=int, default=0,
                    help="Iterations of 3x3 cross dilation applied to the arm mask "
                         "before clipping the robot. Larger = more robot pixels "
                         "survive outside the human silhouette. Residual mask is "
                         "still computed against the un-dilated arm mask.")
    ap.add_argument("--debug", action="store_true",
                    help="Also write debug videos: arm-masked raw, robot-only, "
                         "robot-after-arm-clip.")
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

    # Background is the raw video — the residual inpaint will fill leftover human pixels later.
    bg_path = args.processed_demo / "video_L.mp4"
    print(f"[bg] {bg_path}")
    bg_frames = media.read_video(str(bg_path))
    T_vid = bg_frames.shape[0]
    img_h, img_w = bg_frames.shape[1], bg_frames.shape[2]

    # Arm mask (combined arm+hand from SAM2) — robot is clipped to this region.
    mask_path = args.processed_demo / "segmentation_processor" / "masks_arm.npy"
    arm_masks = np.load(mask_path)
    T_use = min(T_vid, joints_left.shape[0], arm_masks.shape[0])
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    print(f"[info] video T={T_vid}, npz T={joints_left.shape[0]}, "
          f"mask T={arm_masks.shape[0]}, using T={T_use}, hand={args.hand}, "
          f"mask_dilate={args.mask_dilate}")

    cx, cy = img_w / 2.0, img_h / 2.0
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy,
                                       znear=0.01, zfar=10.0)
    renderer = pyrender.OffscreenRenderer(img_w, img_h)

    side_cfg = {}
    for s, urdf in (("right", XHAND_URDF_RIGHT), ("left", XHAND_URDF_LEFT)):
        if args.hand not in (s, "both"):
            continue
        cmap = parse_mjcf_rgba(XHAND_XML[s])
        side_cfg[s] = parse_urdf(Path(urdf), color_map=cmap)
    print("URDF loaded for:", list(side_cfg.keys()))

    out_frames = list(bg_frames[:T_use])
    residual = np.zeros((T_use, img_h, img_w), dtype=bool)

    dbg_arm_masked = [] if args.debug else None
    dbg_robot_only = [] if args.debug else None
    dbg_robot_cut  = [] if args.debug else None

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

        arm_t = arm_masks[t].astype(bool)
        if args.mask_dilate > 0:
            arm_t_clip = cv2.dilate(arm_t.astype(np.uint8), dilate_kernel,
                                    iterations=args.mask_dilate).astype(bool)
        else:
            arm_t_clip = arm_t
        if hands_data:
            rgb, robot_mask = render_robot(hands_data, camera, renderer)
            # Clip robot to the (optionally dilated) arm mask.
            draw = robot_mask & arm_t_clip
            out = bg_frames[t].copy()
            out[draw] = rgb[draw]
            out_frames[t] = out
            # Residual = un-dilated arm pixels that the robot didn't cover.
            # We don't inpaint the margin region — those pixels were original BG.
            residual[t] = arm_t & ~draw
        else:
            rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            robot_mask = np.zeros((img_h, img_w), dtype=bool)
            draw = np.zeros((img_h, img_w), dtype=bool)
            residual[t] = arm_t

        if args.debug:
            # 1) raw with arm mask zeroed out (so you can see exactly what SAM2 covered)
            am = bg_frames[t].copy()
            am[arm_t] = 0
            dbg_arm_masked.append(am)

            # 2) robot-only on black: full pyrender output, no arm clip
            ronly = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            ronly[robot_mask] = rgb[robot_mask]
            dbg_robot_only.append(ronly)

            # 3) robot after arm clip: only the pixels that actually got composited
            # (= robot ∧ arm_mask)
            rcut = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            rcut[draw] = rgb[draw]
            dbg_robot_cut.append(rcut)

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T_use}")

    renderer.delete()

    out_dir = args.processed_demo / "overlay_processor"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_mkv = out_dir / "video_overlay_raw.mkv"
    residual_npy = out_dir / "residual_mask.npy"
    media.write_video(str(overlay_mkv), np.stack(out_frames), fps=args.fps, codec="ffv1")
    np.save(residual_npy, residual)
    per_frame = residual.sum(axis=(1, 2))
    print(f"[ok] wrote {overlay_mkv}")
    print(f"[ok] wrote {residual_npy} "
          f"(residual avg {per_frame.mean():.0f} px / max {per_frame.max()} px)")

    if args.debug:
        dbg = out_dir / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        for name, frames in (
            ("video_arm_masked.mkv",    dbg_arm_masked),
            ("video_robot_only.mkv",    dbg_robot_only),
            ("video_robot_arm_cut.mkv", dbg_robot_cut),
        ):
            path = dbg / name
            media.write_video(str(path), np.stack(frames), fps=args.fps, codec="ffv1")
            print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()
