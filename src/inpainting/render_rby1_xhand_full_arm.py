"""Render retargeted XHand hands + full RBY1 arms over an inpainted background.

Local extension to the upstream layered renderer that renders all 7-DOF RBY1
arm links (shoulder→wrist) for each side. IK is solved per-frame with
pinocchio to find the arm joint angles that place the wrist at the HaWoR
target, then FK gives arm-link world poses which are projected into camera
space for pyrender.

The human-removal mask is deliberately NOT used during compositing.  It only
belongs to the earlier background-inpainting stage.  The complete robot
silhouette is drawn afterwards, so fingers and arm links cannot be amputated
by the human mask.

Outputs:
    <processed_demo>/video_overlay_rby1_xhand.mkv
    <processed_demo>/overlay_processor_arm/video_robot_only.mkv
    <processed_demo>/overlay_processor_arm/robot_mask.npz

Usage:
    PYOPENGL_PLATFORM=egl python -u render_rby1_xhand_full_arm.py \
        --processed_demo /result/skill2policy/processed/cam0/0 \
        --hawor_npz /data/skill2policy/cam0_hawor/retarget_input.npz \
        --right_pkl /data/skill2policy/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl  /data/skill2policy/cam0_hawor/qpos_xhand_left.pkl \
        --hand both
"""
import argparse
import json
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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from _paths import XHAND_URDF_LEFT, XHAND_URDF_RIGHT

REPO = Path(__file__).resolve().parent.parent.parent
RBY1_ROOT = REPO / "third_party/mujoco_menagerie/rainbow_robotics_rby1"
SCENE = RBY1_ROOT / "rby1m_1.2_no_gripper.xml"
RBY1_ASSETS = RBY1_ROOT / "assets"

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

# Reachable elbow-down seeds.  The arm has one redundant DoF, so a neutral
# zero seed can converge to an elbow-inverted branch even when the wrist pose
# itself is correct.  These seeds only select the natural branch; subsequent
# frames are warm-started from the previous solution.
_SAFE_ARM_QPOS = {
    "left": np.array([-1.2, 0.4, -1.2, -1.0, 0.0, 1.2, 0.8]),
    "right": np.array([-1.2, -0.4, 1.2, -1.0, 0.0, 1.2, -0.8]),
}

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
# Standalone URDF parsing for the local full-arm extension
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
                 max_iters=100, damping=1e-2, step_scale=0.5,
                 tol_pos=1e-3, tol_ori=1e-2):
    """Solve one arm without allowing multi-turn or branch-flip poses.

    The previous DLS loop integrated unconstrained joint updates.  It could
    therefore match the wrist with physically impossible angles and visibly
    twist the elbow/forearm.  A bounded solve keeps every joint inside the
    robot limits.  A small previous-frame term resolves the redundant arm DoF
    continuously while leaving wrist position/orientation as the main target.
    """
    q = q0.copy()
    fid = model.getFrameId(f"link_{side}_arm_6")
    dof = _arm_dof_idx(model, side)
    lower = model.lowerPositionLimit[dof].astype(np.float64) + 1e-7
    upper = model.upperPositionLimit[dof].astype(np.float64) - 1e-7
    previous = np.clip(q[dof], lower, upper)
    orientation_weight = 0.12
    continuity_weight = 0.004

    def residual(x):
        q[dof] = x
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current = data.oMf[fid]
        position = current.translation - target.translation
        orientation = pin.log3(current.rotation.T @ target.rotation)
        continuity = continuity_weight * (x - previous)
        return np.concatenate([
            position,
            orientation_weight * orientation,
            continuity,
        ])

    def run(start, evaluations):
        return least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            max_nfev=evaluations,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )

    solutions = [run(previous, max_iters)]
    target_error = residual(solutions[0].x)
    if (
        np.linalg.norm(target_error[:3]) > max(tol_pos, 5e-3)
        or np.linalg.norm(target_error[3:6]) / orientation_weight
        > max(tol_ori, np.radians(3.0))
    ):
        # Retry alternate starts only if the warm-started branch cannot reach
        # the target.  Continuity remains part of each candidate's score.
        solutions.extend([
            run(_SAFE_ARM_QPOS[side], max_iters * 2),
            run(0.5 * (lower + upper), max_iters * 2),
        ])

    solution = min(solutions, key=lambda item: np.linalg.norm(residual(item.x)))
    q[dof] = solution.x
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
    seg_node_map = {}
    scene.add(camera, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0), pose=np.eye(4))
    pl_pose = np.eye(4); pl_pose[:3, 3] = [0.3, -0.3, -0.5]
    scene.add(pyrender.PointLight(color=[0.8, 0.8, 0.8], intensity=2.0), pose=pl_pose)

    # Hands
    for side, wrist_pos, R_cam_xhand, qpos_dict, joints_tree, link_meshes in hands_data:
        R_pr = T_CV2GL @ R_cam_xhand
        t_pr = T_CV2GL @ wrist_pos
        T_root = np.eye(4); T_root[:3, :3] = R_pr; T_root[:3, 3] = t_pr
        link_T = compute_fk(joints_tree, qpos_dict, T_root)
        for lname, items in link_meshes.items():
            if lname not in link_T:
                continue
            for mesh, T_vis in items:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                node = scene.add(pr_mesh, pose=link_T[lname] @ T_vis)
                # Semantic IDs match composite_rb5_contact_occlusion.py:
                # thumb, index, middle, ring, pinky = 1..5.  Palm/wrist and
                # arm remain zero so HaCo can only hide finger geometry.
                lower_name = lname.lower()
                label = 0
                for finger_id, token in enumerate(
                    ("thumb", "index", "mid", "ring", "pinky"), start=1
                ):
                    if token in lower_name:
                        label = finger_id
                        break
                seg_node_map[node] = np.array([label, 0, 0], dtype=np.uint8)

    # Arms: link poses are already in MuJoCo camera frame (= OpenGL convention)
    # from T_cam_world * T_world_link, so use them directly — no T_CV2GL.
    for link_poses_gl, arm_meshes in arm_meshes_data:
        for lname, parts in arm_meshes.items():
            if lname not in link_poses_gl:
                continue
            T_pr = link_poses_gl[lname]
            for mesh in parts:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                node = scene.add(pr_mesh, pose=T_pr)
                seg_node_map[node] = np.array([0, 0, 0], dtype=np.uint8)

    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = depth > 0
    semantic, _ = renderer.render(
        scene,
        flags=pyrender.RenderFlags.SEG,
        seg_node_map=seg_node_map,
    )
    finger_labels = semantic[..., 0].astype(np.uint8)
    finger_labels[~mask] = 0
    return color[..., :3], mask, depth.astype(np.float32), finger_labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--background", type=Path, default=None,
                    help="Hand+arm-removed background video. Default: "
                         "<processed_demo>/inpaint_processor/video_human_inpaint.mkv")
    ap.add_argument("--output", type=Path, default=None,
                    help="Composite output. Default: "
                         "<processed_demo>/video_overlay_rby1_xhand.mkv")
    ap.add_argument("--aux_output_dir", type=Path, default=None,
                    help="Robot-only video, mask, and render metadata directory. "
                         "Default: <processed_demo>/overlay_processor_arm")
    ap.add_argument(
        "--compositor_output_dir",
        type=Path,
        default=None,
        help=(
            "Optional compact RGB-D/semantic array bundle for the HaCo "
            "contact-occlusion compositor."
        ),
    )
    ap.add_argument(
        "--compositor_scale",
        type=float,
        default=0.25,
        help="Resolution scale for --compositor_output_dir (default: 0.25).",
    )
    ap.add_argument("--require_smoothed", action="store_true",
                    help="Fail unless both PKLs contain smoothed finger qpos, "
                         "wrist position, and wrist orientation metadata.")
    ap.add_argument("--head_pitch", type=float, default=0.6)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if not 0.0 < args.compositor_scale <= 1.0:
        ap.error("--compositor_scale must be in (0, 1]")

    # --- Load HaWoR data ---
    ri = np.load(args.hawor_npz)
    focal_source = float(ri["img_focal"])

    qr = pickle.load(open(args.right_pkl, "rb"))
    ql = pickle.load(open(args.left_pkl,  "rb"))
    for side, pkl_path, trajectory in (
        ("right", args.right_pkl, qr),
        ("left", args.left_pkl, ql),
    ):
        required = {"data", "wrist_pos", "wrist_quat", "valid", "joint_names"}
        missing = sorted(required - set(trajectory))
        if missing:
            raise KeyError(f"{side} trajectory {pkl_path} missing keys: {missing}")
        if args.require_smoothed and "smoothing" not in trajectory:
            raise ValueError(
                f"{side} trajectory is not marked as smoothed: {pkl_path}"
            )

    right_data  = np.asarray(qr["data"])
    left_data   = np.asarray(ql["data"])
    right_wrist_pos = np.asarray(qr["wrist_pos"], dtype=np.float64)
    left_wrist_pos = np.asarray(ql["wrist_pos"], dtype=np.float64)
    right_wrist_rot = Rotation.from_quat(
        np.asarray(qr["wrist_quat"], dtype=np.float64)
    ).as_matrix()
    left_wrist_rot = Rotation.from_quat(
        np.asarray(ql["wrist_quat"], dtype=np.float64)
    ).as_matrix()
    right_valid = np.asarray(qr["valid"], dtype=bool)
    left_valid = np.asarray(ql["valid"], dtype=bool)
    right_jname = qr["joint_names"]
    left_jname  = ql["joint_names"]
    print(
        "[trajectory] right smoothing="
        f"{qr.get('smoothing', 'none')}, left smoothing={ql.get('smoothing', 'none')}"
    )

    # --- Load the already inpainted background.  The robot is composited only
    # after human hand+arm removal, and is never clipped by that removal mask.
    raw_path = args.processed_demo / "video_L.mp4"
    bg_path = (
        args.background
        if args.background is not None
        else args.processed_demo / "inpaint_processor" / "video_human_inpaint.mkv"
    )
    if not bg_path.exists():
        raise FileNotFoundError(
            f"inpainted background missing: {bg_path}\n"
            "Run inpaint_hands.py --mode legacy first."
        )
    print(f"[bg] {bg_path}")
    bg_frames = media.read_video(str(bg_path))
    T_vid = bg_frames.shape[0]
    img_h, img_w = bg_frames.shape[1], bg_frames.shape[2]
    raw_cap = cv2.VideoCapture(str(raw_path))
    raw_w = int(raw_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(raw_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_cap.release()
    if raw_w <= 0 or raw_h <= 0:
        raw_w, raw_h = img_w, img_h
    scale_x = img_w / raw_w
    scale_y = img_h / raw_h
    if abs(scale_x - scale_y) > 0.01:
        print(f"[warn] non-uniform resize from {raw_w}x{raw_h} to {img_w}x{img_h}")
    focal = focal_source * (scale_x + scale_y) / 2.0
    T_use = min(
        T_vid,
        left_data.shape[0],
        right_data.shape[0],
        left_wrist_pos.shape[0],
        right_wrist_pos.shape[0],
        left_wrist_rot.shape[0],
        right_wrist_rot.shape[0],
        left_valid.shape[0],
        right_valid.shape[0],
    )
    print(f"[info] T={T_use}, hand={args.hand}, render={img_w}x{img_h}, "
          f"focal={focal:.1f} (source={focal_source:.1f})")

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
    # Start each redundant arm on its reachable elbow-down branch.  This does
    # not alter the tracked wrist/hand trajectory; it only selects the arm
    # configuration used to reach that trajectory.
    for side in ("right", "left"):
        arm_dof = _arm_dof_idx(pin_model, side)
        q_home[arm_dof] = np.clip(
            _SAFE_ARM_QPOS[side],
            pin_model.lowerPositionLimit[arm_dof],
            pin_model.upperPositionLimit[arm_dof],
        )
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
    q_prev = q_home.copy()
    robot_only_frames = []
    robot_masks = []
    compositor_arrays = None
    if args.compositor_output_dir is not None:
        compositor_dir = args.compositor_output_dir.resolve()
        compositor_dir.mkdir(parents=True, exist_ok=True)
        compositor_width = max(1, int(round(img_w * args.compositor_scale)))
        compositor_height = max(1, int(round(img_h * args.compositor_scale)))
        compositor_shape = (T_use, compositor_height, compositor_width)
        compositor_arrays = {
            "rgb": np.lib.format.open_memmap(
                compositor_dir / "robot_rgb.npy",
                mode="w+", dtype=np.uint8,
                shape=(*compositor_shape, 3),
            ),
            "depth": np.lib.format.open_memmap(
                compositor_dir / "robot_depth.npy",
                mode="w+", dtype=np.float32,
                shape=compositor_shape,
            ),
            "mask": np.lib.format.open_memmap(
                compositor_dir / "robot_mask.npy",
                mode="w+", dtype=bool,
                shape=compositor_shape,
            ),
            "finger_mask": np.lib.format.open_memmap(
                compositor_dir / "robot_finger_mask.npy",
                mode="w+", dtype=bool,
                shape=compositor_shape,
            ),
            "labels": np.lib.format.open_memmap(
                compositor_dir / "robot_finger_labels.npy",
                mode="w+", dtype=np.uint8,
                shape=compositor_shape,
            ),
        }

    for t in range(T_use):
        hands_data = []
        arm_meshes_data = []
        q_ik = q_prev.copy()

        for s in ("right", "left"):
            if s not in side_cfg:
                continue
            side_valid = right_valid if s == "right" else left_valid
            if not side_valid[t]:
                continue

            # Hand data.  The smooth PKL is the single source of truth for
            # finger qpos, wrist translation, and robot-wrist orientation.
            joints_tree, link_meshes = side_cfg[s]
            jnames = right_jname if s == "right" else left_jname
            qdata  = right_data[t] if s == "right" else left_data[t]
            qpos_dict = {jn: float(qdata[i]) for i, jn in enumerate(jnames)}
            wrist_pos = (
                right_wrist_pos[t] if s == "right" else left_wrist_pos[t]
            )
            R_cam_xhand = (
                right_wrist_rot[t] if s == "right" else left_wrist_rot[t]
            )
            hands_data.append(
                (s, wrist_pos, R_cam_xhand, qpos_dict, joints_tree, link_meshes)
            )

            # Arm IK: wrist pose in camera space → world space → IK → FK
            T_world_wrist = cam_to_world(
                wrist_pos.astype(np.float64),
                R_cam_xhand.astype(np.float64),
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

        if hands_data:
            rgb, robot_mask, robot_depth, finger_labels = render_robot(
                hands_data, arm_meshes_data, camera, pr_renderer
            )
            out = bg_frames[t].copy()
            # Do not intersect with the human hand/arm mask.  That mask was
            # consumed by E2FGVI and must not constrain the robot silhouette.
            out[robot_mask] = rgb[robot_mask]
            out_frames[t] = out
        else:
            rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            robot_mask = np.zeros((img_h, img_w), dtype=bool)
            robot_depth = np.zeros((img_h, img_w), dtype=np.float32)
            finger_labels = np.zeros((img_h, img_w), dtype=np.uint8)

        if compositor_arrays is not None:
            size = (compositor_width, compositor_height)
            small_mask = cv2.resize(
                robot_mask.astype(np.uint8), size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            small_labels = cv2.resize(
                finger_labels, size, interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            small_mask |= small_labels > 0
            small_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            small_rgb[~small_mask] = 0
            small_depth = cv2.resize(
                robot_depth, size, interpolation=cv2.INTER_NEAREST
            ).astype(np.float32)
            small_depth[~small_mask] = 0.0
            compositor_arrays["rgb"][t] = small_rgb
            compositor_arrays["depth"][t] = small_depth
            compositor_arrays["mask"][t] = small_mask
            compositor_arrays["finger_mask"][t] = small_labels > 0
            compositor_arrays["labels"][t] = small_labels

        ronly = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        ronly[robot_mask] = rgb[robot_mask]
        robot_only_frames.append(ronly)
        robot_masks.append(robot_mask)

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T_use}")

    pr_renderer.delete()

    if compositor_arrays is not None:
        for array in compositor_arrays.values():
            array.flush()
        dominant_side = (
            "left" if int(left_valid[:T_use].sum()) >= int(right_valid[:T_use].sum())
            else "right"
        )
        (compositor_dir / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "renderer": "pyrender_rby1_xhand_arm_stabilized",
            "side": dominant_side,
            "frames": int(T_use),
            "width": int(compositor_width),
            "height": int(compositor_height),
            "source_width": int(img_w),
            "source_height": int(img_h),
            "fps": int(args.fps),
            "finger_labels": {
                "thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5,
            },
            "arm_ik_joint_limits": True,
            "arm_ik_temporal_continuity": True,
        }, indent=2))
        print(f"[ok] wrote HaCo compositor arrays: {compositor_dir}")

    out_dir = (
        args.aux_output_dir
        if args.aux_output_dir is not None
        else args.processed_demo / "overlay_processor_arm"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or (args.processed_demo / "video_overlay_rby1_xhand.mkv")
    robot_only_path = out_dir / "video_robot_only.mkv"
    robot_mask_path = out_dir / "robot_mask.npz"
    media.write_video(str(output), np.stack(out_frames), fps=args.fps, codec="ffv1")
    media.write_video(str(robot_only_path), np.stack(robot_only_frames),
                      fps=args.fps, codec="ffv1")
    np.savez_compressed(robot_mask_path, mask=np.stack(robot_masks))
    metadata_path = out_dir / "render_metadata.json"
    metadata_path.write_text(json.dumps({
        "right_pkl": str(args.right_pkl.resolve()),
        "left_pkl": str(args.left_pkl.resolve()),
        "right_smoothing": qr.get("smoothing"),
        "left_smoothing": ql.get("smoothing"),
        "smoothed_finger_qpos": (
            "smoothing" in qr and "smoothing" in ql
        ),
        "smoothed_wrist_position": (
            "smoothing" in qr and "smoothing" in ql
        ),
        "smoothed_wrist_orientation": (
            "smoothing" in qr and "smoothing" in ql
        ),
        "arm_ik_uses_smoothed_wrist_pose": True,
        "arm_ik_joint_limits": True,
        "arm_ik_temporal_continuity": True,
        "arm_ik_natural_elbow_seed": True,
        "frames": int(T_use),
        "fps": args.fps,
    }, indent=2))
    coverage = np.stack(robot_masks).sum(axis=(1, 2))
    print(f"[ok] wrote {output}")
    print(f"[ok] wrote {robot_only_path}")
    print(f"[ok] wrote {robot_mask_path} "
          f"(robot avg {coverage.mean():.0f} px / max {coverage.max()} px)")
    print(f"[ok] wrote {metadata_path}")


if __name__ == "__main__":
    main()
