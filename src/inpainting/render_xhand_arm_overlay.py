"""Render retargeted XHand hands + RBY1 arms over the raw video.

Same pipeline as render_xhand_overlay.py but additionally renders the 7-DOF
RBY1 arm links (shoulder→wrist) for each side. IK is solved per-frame with
pinocchio to find the arm joint angles that place the wrist at the HaWoR
target, then FK gives arm-link world poses which are projected into camera
space for pyrender.

Outputs:
    <processed_demo>/overlay_processor/video_overlay_raw.mkv   raw+robot composite
    <processed_demo>/overlay_processor/residual_mask.npy        (T,H,W) bool

Usage:
    PYOPENGL_PLATFORM=egl python -u render_xhand_arm_overlay.py \
        --processed_demo /result/skill2policy/processed/cam0/0 \
        --hawor_npz /data/skill2policy/cam0_hawor/retarget_input.npz \
        --right_pkl /data/skill2policy/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl  /data/skill2policy/cam0_hawor/qpos_xhand_left.pkl \
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
import pinocchio as pin
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

from _paths import XHAND_URDF_LEFT, XHAND_URDF_RIGHT, R_MANO_XHAND

REPO = Path(__file__).resolve().parent.parent.parent
SCENE = REPO / "src/mujoco_sim/scenes/rby1_xhand.xml"
RBY1_ASSETS = REPO / "third_party/mujoco_menagerie/rainbow_robotics_rby1/assets"

T_CV2GL = np.diag([1., -1., -1.])

# head_cam mount on link_head_2 (from compose_rby1_xhand.py)
_T_HEAD2_CAM = pin.SE3(
    np.array([[0.0, 0.0, -1.0],
              [-1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0]]),
    np.array([0.12, 0.0, 0.05]),
)
_CV_TO_MJ = pin.SE3(np.diag([1.0, -1.0, -1.0]), np.zeros(3))
_T_WRIST_ARM6 = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.1261]))

# Arm link → mesh file mapping (from the composed MJCF).
ARM_LINK_MESHES = {
    "right": {
        "link_right_arm_0": ["LINK_7.obj"],
        "link_right_arm_1": ["LINK_8.obj"],
        "link_right_arm_2": ["LINK_9_0.obj", "LINK_9_1.obj"],
        "link_right_arm_3": ["LINK_10_0.obj", "LINK_10_1.obj"],
        "link_right_arm_4": ["LINK_11.obj"],
        "link_right_arm_5": ["LINK_12_0_V1.1.obj", "LINK_12_1_V1.1.obj"],
        "link_right_arm_6": ["LINK_13_0_V1.1.obj", "LINK_13_1_V1.1.obj"],
    },
    "left": {
        "link_left_arm_0": ["LINK_14.obj"],
        "link_left_arm_1": ["LINK_15.obj"],
        "link_left_arm_2": ["LINK_16_0.obj", "LINK_16_1.obj"],
        "link_left_arm_3": ["LINK_17_0.obj", "LINK_17_1.obj"],
        "link_left_arm_4": ["LINK_18.obj"],
        "link_left_arm_5": ["LINK_19_0_V1.1.obj", "LINK_19_1_V1.1.obj"],
        "link_left_arm_6": ["LINK_20_0_V1.1.obj", "LINK_20_1_V1.1.obj"],
    },
}

ARM_COLOR = [180, 180, 185, 255]  # light grey, robot-like


# ---------------------------------------------------------------------------
# URDF parsing (reused from render_xhand_overlay.py)
# ---------------------------------------------------------------------------

def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def parse_urdf(urdf_path: Path):
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
            m.visual.face_colors = [200, 170, 130, 255]
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


# ---------------------------------------------------------------------------
# Pinocchio IK helpers (adapted from ik_arm.py, no mujoco dependency)
# ---------------------------------------------------------------------------

def _arm_dof_idx(model: pin.Model, side: str) -> np.ndarray:
    idx = []
    for i in range(7):
        jname = f"{side}_arm_{i}"
        jid = model.getJointId(jname)
        idx.append(model.idx_vs[jid])
    return np.array(idx, dtype=int)


def head_cam_world(model: pin.Model, data: pin.Data, q: np.ndarray) -> pin.SE3:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fid = model.getFrameId("link_head_2")
    return data.oMf[fid] * _T_HEAD2_CAM


def cam_to_world(p_cv, R_cv, T_world_mjcam):
    T_cvcam_wrist = pin.SE3(R_cv, p_cv)
    return T_world_mjcam * _CV_TO_MJ * T_cvcam_wrist


def solve_arm_ik(model, data, q0, side, target,
                 max_iters=200, damping=1e-2, step_scale=0.5,
                 tol_pos=1e-3, tol_ori=1e-2):
    q = q0.copy()
    fid = model.getFrameId(f"link_{side}_arm_6")
    dof = _arm_dof_idx(model, side)
    eye6 = np.eye(6)
    for _ in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        err6 = pin.log6(data.oMf[fid].inverse() * target).vector
        if np.linalg.norm(err6[:3]) < tol_pos and np.linalg.norm(err6[3:]) < tol_ori:
            break
        J_full = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)
        J = J_full[:, dof]
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * eye6, err6)
        v = np.zeros(model.nv)
        v[dof] = step_scale * dq
        q = pin.integrate(model, q, v)
    return q


def get_arm_link_poses_world(model, data, q, side):
    """Return {link_name: SE3} for all 7 arm links after FK at config q."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    poses = {}
    for i in range(7):
        lname = f"link_{side}_arm_{i}"
        fid = model.getFrameId(lname)
        poses[lname] = data.oMf[fid].copy()
    return poses


# ---------------------------------------------------------------------------
# Arm mesh loading
# ---------------------------------------------------------------------------

def load_arm_meshes(side: str):
    """Load OBJ meshes for the arm links. Returns {link_name: [trimesh.Trimesh]}."""
    meshes = {}
    for link_name, obj_files in ARM_LINK_MESHES[side].items():
        parts = []
        for fname in obj_files:
            fpath = RBY1_ASSETS / fname
            if not fpath.exists():
                print(f"[warn] missing mesh {fpath}")
                continue
            m = trimesh.load(str(fpath), force="mesh")
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(list(m.geometry.values()))
            m.visual.face_colors = ARM_COLOR
            parts.append(m)
        if parts:
            meshes[link_name] = parts
    return meshes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_robot(
    hands_data,       # hand meshes in camera space (from URDF FK)
    arm_meshes_data,  # (link_poses_camspace, arm_meshes) per side
    camera, renderer,
):
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0., 0., 0., 0.])
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0), pose=np.eye(4))
    pl_pose = np.eye(4); pl_pose[:3, 3] = [0.3, -0.3, -0.5]
    scene.add(pyrender.PointLight(color=[0.8, 0.8, 0.8], intensity=2.0), pose=pl_pose)

    # Hands (same as render_xhand_overlay.py)
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

    # Arms: link poses are already in MuJoCo camera frame (= OpenGL convention)
    # from T_cam_world * T_world_link, so use them directly — no T_CV2GL.
    for link_poses_gl, arm_meshes in arm_meshes_data:
        for lname, parts in arm_meshes.items():
            if lname not in link_poses_gl:
                continue
            T_pr = link_poses_gl[lname]
            for mesh in parts:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                scene.add(pr_mesh, pose=T_pr)

    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = depth > 0
    return color[..., :3], mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--mask_dilate", type=int, default=0)
    ap.add_argument("--head_pitch", type=float, default=0.6)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # --- Load HaWoR data ---
    ri = np.load(args.hawor_npz)
    joints_left  = ri["joints_left"].astype(np.float64)
    joints_right = ri["joints_right"].astype(np.float64)
    go    = ri["mano_global_orient"]
    valid = ri["valid"]
    focal = float(ri["img_focal"])

    qr = pickle.load(open(args.right_pkl, "rb"))
    ql = pickle.load(open(args.left_pkl,  "rb"))
    right_data  = np.asarray(qr["data"])
    left_data   = np.asarray(ql["data"])
    right_jname = qr["joint_names"]
    left_jname  = ql["joint_names"]

    # --- Load background video + arm masks ---
    bg_path = args.processed_demo / "video_L.mp4"
    print(f"[bg] {bg_path}")
    bg_frames = media.read_video(str(bg_path))
    T_vid = bg_frames.shape[0]
    img_h, img_w = bg_frames.shape[1], bg_frames.shape[2]

    mask_path = args.processed_demo / "segmentation_processor" / "masks_arm.npy"
    arm_masks = np.load(mask_path)
    T_use = min(T_vid, joints_left.shape[0], arm_masks.shape[0])
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    print(f"[info] T={T_use}, hand={args.hand}, mask_dilate={args.mask_dilate}")

    # --- Pyrender camera ---
    cx, cy = img_w / 2.0, img_h / 2.0
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy,
                                       znear=0.01, zfar=10.0)
    pr_renderer = pyrender.OffscreenRenderer(img_w, img_h)

    # --- Load hand URDFs ---
    side_cfg = {}
    for s, urdf in (("right", XHAND_URDF_RIGHT), ("left", XHAND_URDF_LEFT)):
        if args.hand not in (s, "both"):
            continue
        side_cfg[s] = parse_urdf(Path(urdf))
    print("URDF loaded for:", list(side_cfg.keys()))

    # --- Load arm meshes ---
    arm_mesh_cfg = {}
    for s in side_cfg:
        arm_mesh_cfg[s] = load_arm_meshes(s)
    print("Arm meshes loaded for:", list(arm_mesh_cfg.keys()))

    # --- Pinocchio model for IK ---
    pin_model = pin.buildModelFromMJCF(str(SCENE))
    pin_data = pin_model.createData()
    q_home = np.zeros(pin_model.nq)
    # Set head_1 pitch
    for i in range(pin_model.njoints):
        if pin_model.names[i] == "head_1":
            q_home[pin_model.idx_qs[i]] = args.head_pitch
            break
    T_world_cam = head_cam_world(pin_model, pin_data, q_home)
    T_cam_world = T_world_cam.inverse()
    print(f"[info] head_cam world pos: {T_world_cam.translation}")

    # --- Main loop ---
    out_frames = list(bg_frames[:T_use])
    residual = np.zeros((T_use, img_h, img_w), dtype=bool)
    q_prev = q_home.copy()

    dbg_robot_only = [] if args.debug else None

    for t in range(T_use):
        hands_data = []
        arm_meshes_data = []
        q_ik = q_prev.copy()

        for s in ("right", "left"):
            if s not in side_cfg:
                continue
            h_idx = 1 if s == "right" else 0
            if not valid[h_idx, t]:
                continue

            # Hand data (same as render_xhand_overlay.py)
            joints_tree, link_meshes = side_cfg[s]
            jnames = right_jname if s == "right" else left_jname
            qdata  = right_data[t] if s == "right" else left_data[t]
            qpos_dict = {jn: float(qdata[i]) for i, jn in enumerate(jnames)}
            wrist_pos = (joints_right if s == "right" else joints_left)[t, 0, :]
            R_mano = Rotation.from_rotvec(go[h_idx, t]).as_matrix()
            hands_data.append((s, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes))

            # Arm IK: wrist pose in camera space → world space → IK → FK
            T_world_wrist = cam_to_world(
                wrist_pos.astype(np.float64),
                (R_mano @ R_MANO_XHAND[s]).astype(np.float64),
                T_world_cam,
            )
            T_world_arm6 = T_world_wrist * _T_WRIST_ARM6
            q_ik = solve_arm_ik(pin_model, pin_data, q_ik, s, T_world_arm6)

            # Get arm link poses in world space, convert to camera space
            link_poses_world = get_arm_link_poses_world(pin_model, pin_data, q_ik, s)
            link_poses_cam = {}
            for lname, T_w in link_poses_world.items():
                T_c = T_cam_world * T_w
                T_c_44 = np.eye(4)
                T_c_44[:3, :3] = T_c.rotation
                T_c_44[:3, 3] = T_c.translation
                link_poses_cam[lname] = T_c_44

            arm_meshes_data.append((link_poses_cam, arm_mesh_cfg[s]))

        q_prev = q_ik

        arm_t = arm_masks[t].astype(bool)
        if args.mask_dilate > 0:
            arm_t_clip = cv2.dilate(arm_t.astype(np.uint8), dilate_kernel,
                                    iterations=args.mask_dilate).astype(bool)
        else:
            arm_t_clip = arm_t

        if hands_data:
            rgb, robot_mask = render_robot(hands_data, arm_meshes_data,
                                           camera, pr_renderer)
            draw = robot_mask & arm_t_clip
            out = bg_frames[t].copy()
            out[draw] = rgb[draw]
            out_frames[t] = out
            residual[t] = arm_t & ~draw
        else:
            residual[t] = arm_t

        if args.debug and hands_data:
            ronly = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            ronly[robot_mask] = rgb[robot_mask]
            dbg_robot_only.append(ronly)
        elif args.debug:
            dbg_robot_only.append(np.zeros((img_h, img_w, 3), dtype=np.uint8))

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T_use}")

    pr_renderer.delete()

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
        dbg_dir = out_dir / "debug"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        path = dbg_dir / "video_robot_only.mkv"
        media.write_video(str(path), np.stack(dbg_robot_only), fps=args.fps, codec="ffv1")
        print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()
