"""Stage 5 — render xhand hands + RBY1 lower arm, emit RGB + depth + mask.

This is the depth-aware variant of `render_xhand_overlay.py`. It does NOT
composite the robot into the raw video and does NOT clip to the arm mask —
that's the depth-aware compositor's job (`composite_layered.py`).

In addition to the XHand finger meshes, the lower arm links (arm3→arm6,
i.e. forearm + wrist) are rendered. Their poses are derived geometrically
from the wrist orientation: the forearm extends along the +z axis of the
xhand/arm6 frame, using the RBY1 kinematic offsets.

Outputs (all length-T arrays, T = min(video, hawor)):
    <processed_demo>/overlay_processor/robot_rgb.npy    (T,H,W,3) uint8
    <processed_demo>/overlay_processor/robot_depth.npy  (T,H,W)   float16
                                                        meters in MANO cam frame.
                                                        +inf where the robot is not drawn.
    <processed_demo>/overlay_processor/robot_mask.npy   (T,H,W)   bool

Coordinate sanity:
    pyrender camera at identity (looks along -Z in OpenGL world)
    T_CV2GL = diag(1, -1, -1) maps MANO cam (OpenCV) → pyrender world.
    With the camera at origin, pyrender's returned depth value equals the
    original MANO cam Z (forward, meters), so no extra rescale.

Usage:
    PYOPENGL_PLATFORM=egl python -u render_xhand_overlay_depth.py \
        --processed_demo /result/cam0_inpaint/cam0/0 \
        --hawor_npz /data/RFM_proj/cam0_hawor/retarget_input.npz \
        --right_pkl /data/RFM_proj/cam0_hawor/qpos_xhand_right.pkl \
        --left_pkl  /data/RFM_proj/cam0_hawor/qpos_xhand_left.pkl
"""
import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mediapy as media
import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

from _paths import (XHAND_URDF_LEFT, XHAND_URDF_RIGHT, R_MANO_XHAND,
                    EMBODIMENT_NAMES, forearm_sign)
from render_xhand_overlay import (compute_fk, parse_urdf, parse_mjcf_rgba,
                                  T_CV2GL, XHAND_XML, build_side_align,
                                  hand_root_pose, load_side_urdf,
                                  resolve_embodiment)
from scene_lighting import (directional_light_pose, estimate_illumination,
                            light_dir_cam)
from traj_smooth import smooth_channels, smooth_rotvec


def _build_light(frames, dir_cam=None):
    """Scene-matched light rig for _render_robot_rgbd, from the video frames.

    One global tint (room colour is ~constant) drives a scene-coloured key
    light aimed along the shared LIGHT_DIR_CAM, plus warm ambient and a weak
    fill. Returns the dict _render_robot_rgbd expects.
    """
    tint = estimate_illumination(frames)
    return dict(
        key_color=tint,
        key_intensity=3.5,
        key_pose=directional_light_pose(dir_cam, T_CV2GL),
        ambient=(tint * 0.35).tolist(),
        fill_color=(tint * 0.8),
        fill_intensity=1.2,
        dir_cam=light_dir_cam(dir_cam),
    )

REPO = Path(__file__).resolve().parent.parent.parent
RBY1_ASSETS = REPO / "third_party/mujoco_menagerie/rainbow_robotics_rby1/assets"

# XHand root is at arm6 + (0, 0, -0.1261) in arm6 frame, so
# arm6 = wrist + 0.1261 along forearm (+z of xhand frame).
EE_OFFSET_Z = 0.1261

# Lower arm link meshes (arm3=elbow → arm6=wrist end).
# arm4/5/6 share the same body-frame origin; arm3 is 0.256m further up the arm.
_LOWER_ARM_MESHES = {
    "right": {
        "link_right_arm_3": ["LINK_10_0.obj", "LINK_10_1.obj"],
        "link_right_arm_4": ["LINK_11.obj"],
        "link_right_arm_5": ["LINK_12_0_V1.1.obj", "LINK_12_1_V1.1.obj"],
        "link_right_arm_6": ["LINK_13_0_V1.1.obj", "LINK_13_1_V1.1.obj"],
    },
    "left": {
        "link_left_arm_3": ["LINK_17_0.obj", "LINK_17_1.obj"],
        "link_left_arm_4": ["LINK_18.obj"],
        "link_left_arm_5": ["LINK_19_0_V1.1.obj", "LINK_19_1_V1.1.obj"],
        "link_left_arm_6": ["LINK_20_0_V1.1.obj", "LINK_20_1_V1.1.obj"],
    },
}
# Body-frame offsets from arm4 (MJCF): arm3 is at arm4 + (0.031, 0, 0.256) in arm4 frame.
_ARM3_OFFSET = np.array([0.031, 0.0, 0.256])
def _load_lower_arm_meshes(side: str) -> dict:
    """Load OBJ meshes for lower arm links. Returns {link_name: [trimesh]}.

    Meshes keep their embedded OBJ material (the RBY1 arm geoms have no
    explicit rgba in the composed MJCF, so MuJoCo uses the mesh's own
    material — a neutral grey).
    """
    out = {}
    for link_name, fnames in _LOWER_ARM_MESHES[side].items():
        parts = []
        for fn in fnames:
            p = RBY1_ASSETS / fn
            if not p.exists():
                print(f"[warn] missing {p}")
                continue
            m = trimesh.load(str(p), force="mesh")
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(list(m.geometry.values()))
            parts.append(m)
        if parts:
            out[link_name] = parts
    return out


# Hand frame -> arm frame. The arm code assumes the wrist frame's +z runs
# wrist->elbow, which holds for xhand. A hand whose +z runs the other way
# (inspire) needs a 180 deg roll about x to bring it into that convention;
# applying it to the whole rotation — not just the forearm direction — keeps
# _ARM3_OFFSET and the arm mesh orientation consistent with it.
_FLIP_X = np.diag([1.0, -1.0, -1.0])


def _arm_link_poses_pyrender(side, wrist_pos_cv, R_cam_xhand, sign=+1):
    """Compute lower-arm link poses in pyrender (OpenGL) space.

    All coordinates start in OpenCV camera space (x-right, y-down, z-forward),
    then are converted to pyrender (T_CV2GL).

    *sign* is the embodiment's forearm_sign. EE_OFFSET_Z is a property of the
    RBY1 flange rather than of the hand, so it is shared across embodiments;
    only the frame convention differs.

    Returns {link_name: 4x4 pose matrix} for arm3–arm6.
    """
    R_cam_arm = R_cam_xhand if sign > 0 else R_cam_xhand @ _FLIP_X
    forearm_dir_cv = R_cam_arm[:, 2]  # wrist -> elbow

    # arm6/5/4 share the same position: wrist + EE_OFFSET along forearm
    pos_arm456_cv = wrist_pos_cv + EE_OFFSET_Z * forearm_dir_cv
    # arm3 (elbow) is further up: arm4 + _ARM3_OFFSET in arm4's local frame
    pos_arm3_cv = pos_arm456_cv + R_cam_arm @ _ARM3_OFFSET

    R_pr = T_CV2GL @ R_cam_arm
    poses = {}
    for lname, pos_cv in [
        (f"link_{side}_arm_3", pos_arm3_cv),
        (f"link_{side}_arm_4", pos_arm456_cv),
        (f"link_{side}_arm_5", pos_arm456_cv),
        (f"link_{side}_arm_6", pos_arm456_cv),
    ]:
        T = np.eye(4)
        T[:3, :3] = R_pr
        T[:3, 3] = T_CV2GL @ pos_cv
        poses[lname] = T
    return poses


def _render_robot_rgbd(hands_data, arm_scene_data, camera, renderer, side_align,
                       light=None):
    """Return (rgb (H,W,3) uint8, depth (H,W) float32 meters, mask (H,W) bool).

    *light*: None keeps the legacy neutral rig; otherwise a dict from
    _build_light() with the scene-matched tint and direction.
    """
    if light is None:
        # Legacy rig — kept so --relight none reproduces prior output exactly.
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0., 0., 0., 0.])
        scene.add(camera, pose=np.eye(4))
        scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0),
                  pose=np.eye(4))
        pl_pose = np.eye(4); pl_pose[:3, 3] = [0.3, -0.3, -0.5]
        scene.add(pyrender.PointLight(color=[0.8, 0.8, 0.8], intensity=2.0),
                  pose=pl_pose)
    else:
        scene = pyrender.Scene(ambient_light=light["ambient"],
                               bg_color=[0., 0., 0., 0.])
        scene.add(camera, pose=np.eye(4))
        # Key light: scene-tinted, aimed along the shared LIGHT_DIR_CAM so its
        # highlights agree with the contact shadow cast in the compositor.
        scene.add(pyrender.DirectionalLight(color=light["key_color"],
                                            intensity=light["key_intensity"]),
                  pose=light["key_pose"])
        # Weak tinted fill from the camera so shadowed sides don't go pure black.
        pl_pose = np.eye(4); pl_pose[:3, 3] = [0.3, -0.3, -0.5]
        scene.add(pyrender.PointLight(color=light["fill_color"],
                                      intensity=light["fill_intensity"]),
                  pose=pl_pose)

    # Robot hand fingers
    for side, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes in hands_data:
        T_root, _ = hand_root_pose(R_mano, wrist_pos, side_align[side])
        link_T = compute_fk(joints_tree, qpos_dict, T_root)

        for lname, items in link_meshes.items():
            if lname not in link_T:
                continue
            for mesh, T_vis in items:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                scene.add(pr_mesh, pose=link_T[lname] @ T_vis)

    # Lower arm links
    for link_poses, arm_meshes in arm_scene_data:
        for lname, parts in arm_meshes.items():
            if lname not in link_poses:
                continue
            for mesh in parts:
                pr_mesh = pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
                scene.add(pr_mesh, pose=link_poses[lname])

    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = depth > 0
    return color[..., :3], depth.astype(np.float32), mask


def _render_thumb_depths(hands_data, camera, renderer, side_align):
    """Render each XHand thumb alone for visible-pixel identification.

    Visibility against the complete robot is resolved by the caller using the
    full-scene depth.  Separate side buffers let contact compositors promote
    only the thumb associated with a tracked interaction object.
    """
    outputs = {}
    for target_side in ("right", "left"):
        scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3],
            bg_color=[0.0, 0.0, 0.0, 0.0],
        )
        scene.add(camera, pose=np.eye(4))
        added = False
        for side, wrist_pos, R_mano, qpos_dict, joints_tree, link_meshes in hands_data:
            if side != target_side:
                continue
            T_root, _ = hand_root_pose(R_mano, wrist_pos, side_align[side])
            link_T = compute_fk(joints_tree, qpos_dict, T_root)
            thumb_links = {
                f"{side}_hand_thumb_bend_link",
                f"{side}_hand_thumb_rota_link1",
                f"{side}_hand_thumb_rota_link2",
                f"{side}_hand_thumb_rota_tip",
            }
            for lname in thumb_links:
                if lname not in link_T or lname not in link_meshes:
                    continue
                for mesh, T_vis in link_meshes[lname]:
                    scene.add(
                        pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False),
                        pose=link_T[lname] @ T_vis,
                    )
                    added = True
        if added:
            depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            outputs[target_side] = depth.astype(np.float32)
        else:
            outputs[target_side] = None
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl",  type=Path, required=True)
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    ap.add_argument("--output_subdir", default="overlay_processor",
                    help="Output directory under --processed_demo.")
    ap.add_argument("--right_embodiment", choices=EMBODIMENT_NAMES, default=None,
                    help="Override the robot hand model for the right hand. "
                         "Default: read from the pkl, which retargeting stamps.")
    ap.add_argument("--left_embodiment", choices=EMBODIMENT_NAMES, default=None,
                    help="Override the robot hand model for the left hand.")
    ap.add_argument("--relight", choices=["auto", "none"], default="auto",
                    help="auto: tint the robot lighting to match the scene "
                         "(default). none: legacy neutral rig, reproduces prior "
                         "output.")
    ap.add_argument("--light_dir", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Override light travel direction in the camera frame "
                         "(x-right, y-down, z-forward). Default: see "
                         "scene_lighting.LIGHT_DIR_CAM.")
    ap.add_argument("--smooth", dest="smooth", action="store_true", default=True,
                    help="Temporally smooth the trajectory (fingers, wrist "
                         "position, global orientation) before rendering "
                         "(default on).")
    ap.add_argument("--no_smooth", dest="smooth", action="store_false",
                    help="Disable trajectory smoothing.")
    ap.add_argument("--smooth_win", type=int, default=15,
                    help="Savgol window (frames) for finger qpos. Odd.")
    ap.add_argument("--smooth_wrist_win", type=int, default=21,
                    help="Savgol window (frames) for wrist pos + orientation. "
                         "Odd, larger because wrist estimation is jitterier.")
    args = ap.parse_args()

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

    bg_path = args.processed_demo / "video_L.mp4"
    bg_frames = media.read_video(str(bg_path))
    T_vid = bg_frames.shape[0]
    img_h, img_w = bg_frames.shape[1], bg_frames.shape[2]

    T_use = min(T_vid, joints_left.shape[0])
    print(f"[info] video T={T_vid}, npz T={joints_left.shape[0]}, "
          f"using T={T_use}, hand={args.hand}")

    if args.smooth:
        # Smooth exactly what the render loop consumes: finger qpos (pkl) plus
        # wrist position joints_*[:,0] and global orientation go[h] (npz). Done
        # per hand over the time axis, interpolating across invalid frames.
        go = np.asarray(go, dtype=np.float64).copy()
        joints_left = joints_left.copy()
        joints_right = joints_right.copy()
        for h_idx, s in ((0, "left"), (1, "right")):
            v = np.asarray(valid[h_idx]).astype(bool)
            data = left_data if s == "left" else right_data
            n = min(len(data), len(v), len(go[h_idx]))
            vv = v[:n]
            if s == "left":
                left_data[:n]  = smooth_channels(data[:n], vv, win=args.smooth_win)
                joints_left[:n, 0]  = smooth_channels(joints_left[:n, 0], vv,
                                                      win=args.smooth_wrist_win)
            else:
                right_data[:n] = smooth_channels(data[:n], vv, win=args.smooth_win)
                joints_right[:n, 0] = smooth_channels(joints_right[:n, 0], vv,
                                                      win=args.smooth_wrist_win)
            go[h_idx, :n] = smooth_rotvec(go[h_idx, :n], vv,
                                          win=args.smooth_wrist_win)
        print(f"[smooth] on: finger_win={args.smooth_win}, "
              f"wrist_win={args.smooth_wrist_win}")

    cx, cy = img_w / 2.0, img_h / 2.0
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy,
                                       znear=0.01, zfar=10.0)
    renderer = pyrender.OffscreenRenderer(img_w, img_h)

    embodiment_of = {
        "right": resolve_embodiment(qr, args.right_embodiment, "right"),
        "left":  resolve_embodiment(ql, args.left_embodiment, "left"),
    }
    side_cfg = {}
    arm_mesh_cfg = {}
    for s in ("right", "left"):
        if args.hand not in (s, "both"):
            continue
        side_cfg[s] = load_side_urdf(embodiment_of[s], s)
        arm_mesh_cfg[s] = _load_lower_arm_meshes(s)
    side_align = build_side_align(embodiment_of)
    print("URDF loaded for:", {s: embodiment_of[s] for s in side_cfg})
    print("Arm meshes loaded for:", list(arm_mesh_cfg.keys()))

    light = None
    if args.relight == "auto":
        light = _build_light(bg_frames[:T_use], dir_cam=args.light_dir)
        print(f"[relight] scene tint = {np.round(light['key_color'], 3).tolist()}, "
              f"light_dir_cam = {np.round(light['dir_cam'], 3).tolist()}")

    rgb_buf   = np.zeros((T_use, img_h, img_w, 3), dtype=np.uint8)
    depth_buf = np.full((T_use, img_h, img_w), np.inf, dtype=np.float32)
    mask_buf  = np.zeros((T_use, img_h, img_w), dtype=bool)
    thumb_mask_buf = {
        "right": np.zeros((T_use, img_h, img_w), dtype=bool),
        "left": np.zeros((T_use, img_h, img_w), dtype=bool),
    }

    for t in range(T_use):
        hands_data = []
        arm_scene_data = []
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

            # Lower arm poses from wrist orientation
            _, R_cam_hand = hand_root_pose(R_mano, wrist_pos, side_align[s])
            link_poses = _arm_link_poses_pyrender(
                s, wrist_pos, R_cam_hand, sign=forearm_sign(embodiment_of[s]))
            arm_scene_data.append((link_poses, arm_mesh_cfg[s]))

        if not hands_data:
            continue

        rgb, depth, mask = _render_robot_rgbd(hands_data, arm_scene_data,
                                              camera, renderer, side_align, light)
        rgb_buf[t] = rgb
        mask_buf[t] = mask
        depth_buf[t, mask] = depth[mask]   # leave non-robot pixels at +inf
        thumb_depths = _render_thumb_depths(
            hands_data, camera, renderer, side_align
        )
        for side, thumb_depth in thumb_depths.items():
            if thumb_depth is None:
                continue
            # Retain only thumb surfaces that are frontmost in the complete
            # robot render; hidden thumb triangles must not be promoted later.
            thumb_mask_buf[side][t] = (
                (thumb_depth > 0)
                & mask
                & (np.abs(thumb_depth - depth) <= 2.0e-4)
            )

        if (t + 1) % 100 == 0:
            print(f"  {t+1}/{T_use}")

    renderer.delete()

    out_dir = args.processed_demo / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "robot_rgb.npy",   rgb_buf)
    np.save(out_dir / "robot_depth.npy", depth_buf.astype(np.float16))
    np.save(out_dir / "robot_mask.npy",  mask_buf)
    np.save(out_dir / "robot_thumb_mask_right.npy", thumb_mask_buf["right"])
    np.save(out_dir / "robot_thumb_mask_left.npy", thumb_mask_buf["left"])
    per_frame = mask_buf.sum(axis=(1, 2))
    print(f"[ok] wrote robot_rgb/depth/mask to {out_dir}  "
          f"(robot avg {per_frame.mean():.0f} px / max {per_frame.max()} px)")


if __name__ == "__main__":
    main()
