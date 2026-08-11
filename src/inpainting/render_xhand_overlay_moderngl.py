"""Headless xhand RGB/depth renderer using a standalone ModernGL context.

This is a drop-in output-compatible fallback for
``render_xhand_overlay_depth.py`` on machines where pyrender cannot initialize
EGL.  It writes ``robot_rgb.npy``, ``robot_depth.npy`` and ``robot_mask.npy``
incrementally, so long 720p clips do not require several gigabytes of RAM.

The renderer consumes the same Skill2Policy NPZ and xhand PKL files.  It also
writes ``robot_preview.mp4`` by placing the render over the source video, which
is useful for checking camera-space alignment before running inpainting.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import moderngl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from _paths import REPO_ROOT, forearm_sign
from render_xhand_overlay import (
    T_CV2GL,
    build_side_align,
    compute_fk,
    hand_root_pose,
    load_side_urdf,
    resolve_embodiment,
)
from traj_smooth import smooth_channels, smooth_rotvec


RBY1_ASSETS = (
    Path(REPO_ROOT) / "third_party" / "mujoco_menagerie" /
    "rainbow_robotics_rby1" / "assets"
)
EE_OFFSET_Z = 0.1261
_ARM3_OFFSET = np.array([0.031, 0.0, 0.256])
_FLIP_X = np.diag([1.0, -1.0, -1.0])
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


def _load_arm_meshes(side: str) -> dict[str, list[trimesh.Trimesh]]:
    result = {}
    for link, names in _LOWER_ARM_MESHES[side].items():
        parts = []
        for name in names:
            path = RBY1_ASSETS / name
            if not path.exists():
                continue
            mesh = trimesh.load(str(path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            parts.append(mesh)
        if parts:
            result[link] = parts
    return result


def _arm_link_poses(side: str, wrist_pos_cv: np.ndarray,
                    r_cam_hand: np.ndarray, sign: int) -> dict[str, np.ndarray]:
    r_cam_arm = r_cam_hand if sign > 0 else r_cam_hand @ _FLIP_X
    pos_456 = wrist_pos_cv + EE_OFFSET_Z * r_cam_arm[:, 2]
    pos_3 = pos_456 + r_cam_arm @ _ARM3_OFFSET
    r_gl = T_CV2GL @ r_cam_arm
    poses = {}
    for link, pos in (
        (f"link_{side}_arm_3", pos_3),
        (f"link_{side}_arm_4", pos_456),
        (f"link_{side}_arm_5", pos_456),
        (f"link_{side}_arm_6", pos_456),
    ):
        transform = np.eye(4)
        transform[:3, :3] = r_gl
        transform[:3, 3] = T_CV2GL @ pos
        poses[link] = transform
    return poses


def _mesh_color(mesh: trimesh.Trimesh, fallback=(165, 172, 184)) -> np.ndarray:
    try:
        color = np.asarray(mesh.visual.main_color, dtype=np.float32)[:3]
    except Exception:
        color = np.asarray(fallback, dtype=np.float32)
    if color.max(initial=0) <= 1.0:
        color *= 255.0
    return np.clip(color / 255.0, 0.0, 1.0).astype(np.float32)


@dataclass
class GLMesh:
    vao: moderngl.VertexArray
    color: np.ndarray
    visual_transform: np.ndarray


class Renderer:
    def __init__(self, width: int, height: int, focal: float,
                 near: float = 0.01, far: float = 10.0):
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 projection;
                uniform mat4 model;
                in vec3 in_position;
                in vec3 in_normal;
                out vec3 normal_world;
                void main() {
                    gl_Position = projection * model * vec4(in_position, 1.0);
                    normal_world = mat3(transpose(inverse(model))) * in_normal;
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 base_color;
                in vec3 normal_world;
                out vec4 frag_color;
                void main() {
                    vec3 n = normalize(normal_world);
                    vec3 light_dir = normalize(vec3(-0.35, 0.55, 0.75));
                    float diffuse = abs(dot(n, light_dir));
                    float shade = 0.42 + 0.58 * diffuse;
                    vec3 color = base_color * shade + vec3(0.06) * (1.0 - shade);
                    frag_color = vec4(color, 1.0);
                }
            """,
        )
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = 2.0 * focal / width
        projection[1, 1] = 2.0 * focal / height
        projection[2, 2] = -(far + near) / (far - near)
        projection[2, 3] = -(2.0 * far * near) / (far - near)
        projection[3, 2] = -1.0
        self.program["projection"].write(projection.T.tobytes())
        self.color = self.ctx.texture((width, height), 4, dtype="f1")
        self.depth = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(self.color, self.depth)

    def upload(self, mesh: trimesh.Trimesh,
               visual_transform: np.ndarray | None = None) -> GLMesh:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
        interleaved = np.concatenate([vertices, normals], axis=1)
        vbo = self.ctx.buffer(interleaved.tobytes())
        ibo = self.ctx.buffer(faces.tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 3f", "in_position", "in_normal")],
            index_buffer=ibo,
            index_element_size=4,
        )
        return GLMesh(vao=vao, color=_mesh_color(mesh),
                      visual_transform=(np.eye(4) if visual_transform is None
                                        else np.asarray(visual_transform)))

    def render(self, drawables: list[tuple[GLMesh, np.ndarray]]):
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        for item, model in drawables:
            self.program["model"].write(
                np.asarray(model @ item.visual_transform,
                           dtype=np.float32).T.tobytes()
            )
            self.program["base_color"].value = tuple(float(v) for v in item.color)
            item.vao.render(moderngl.TRIANGLES)
        rgba = np.frombuffer(
            self.fbo.read(components=4, dtype="f1", alignment=1), dtype=np.uint8
        ).reshape(self.height, self.width, 4)
        zbuf = np.frombuffer(self.depth.read(), dtype=np.float32).reshape(
            self.height, self.width
        )
        rgba = np.flipud(rgba)
        zbuf = np.flipud(zbuf)
        mask = zbuf < (1.0 - 1e-7)
        z_ndc = zbuf * 2.0 - 1.0
        linear = (2.0 * self.near * self.far) / np.clip(
            self.far + self.near - z_ndc * (self.far - self.near), 1e-8, None
        )
        linear[~mask] = np.inf
        return rgba[..., :3].copy(), linear.astype(np.float32), mask


def _upload_link_meshes(renderer: Renderer, link_meshes: dict):
    result = {}
    for link, parts in link_meshes.items():
        uploaded = []
        for part in parts:
            if isinstance(part, tuple):
                mesh, visual_transform = part
            else:
                mesh, visual_transform = part, np.eye(4)
            uploaded.append(renderer.upload(mesh, visual_transform))
        result[link] = uploaded
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_demo", type=Path, required=True)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--right_pkl", type=Path, required=True)
    parser.add_argument("--left_pkl", type=Path, required=True)
    parser.add_argument("--hand", choices=("right", "left", "both"), default="both")
    parser.add_argument("--output_subdir", default="overlay_processor")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_arm", action="store_true")
    parser.add_argument("--no_smooth", action="store_true")
    args = parser.parse_args()

    video_path = args.processed_demo / "video_L.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    pose = np.load(args.hawor_npz)
    joints = {
        "left": pose["joints_left"].astype(np.float64),
        "right": pose["joints_right"].astype(np.float64),
    }
    orient = pose["mano_global_orient"].astype(np.float64).copy()
    valid = pose["valid"].astype(bool)
    focal = float(pose["img_focal"])
    pkl = {
        "right": pickle.load(open(args.right_pkl, "rb")),
        "left": pickle.load(open(args.left_pkl, "rb")),
    }
    qdata = {side: np.asarray(pkl[side]["data"], dtype=np.float64).copy()
             for side in ("right", "left")}
    if not args.no_smooth:
        for side, hand_idx in (("left", 0), ("right", 1)):
            n = min(len(qdata[side]), len(joints[side]), len(valid[hand_idx]))
            qdata[side][:n] = smooth_channels(
                qdata[side][:n], valid[hand_idx, :n], win=15
            )
            joints[side][:n, 0] = smooth_channels(
                joints[side][:n, 0], valid[hand_idx, :n], win=21
            )
            orient[hand_idx, :n] = smooth_rotvec(
                orient[hand_idx, :n], valid[hand_idx, :n], win=21
            )

    enabled = [side for side in ("right", "left")
               if args.hand in (side, "both")]
    embodiment = {side: resolve_embodiment(pkl[side], None, side) for side in enabled}
    side_align = build_side_align(embodiment)
    side_urdf = {side: load_side_urdf(embodiment[side], side) for side in enabled}
    arm_meshes = {side: _load_arm_meshes(side) for side in enabled}

    renderer = Renderer(width, height, focal)
    hand_gl = {}
    arm_gl = {}
    for side in enabled:
        tree, meshes = side_urdf[side]
        hand_gl[side] = (tree, _upload_link_meshes(renderer, meshes))
        arm_gl[side] = _upload_link_meshes(renderer, arm_meshes[side])
    print(f"[info] GL renderer={renderer.ctx.info.get('GL_RENDERER')}, "
          f"video={width}x{height}, focal={focal:.1f}, enabled={enabled}")

    start = max(0, args.start_frame)
    total = min(video_frames, joints["left"].shape[0], joints["right"].shape[0],
                len(qdata["right"]), len(qdata["left"]))
    end = total if args.max_frames is None else min(total, start + args.max_frames)
    count = max(0, end - start)
    if not count:
        raise ValueError(f"empty frame range {start}:{end}")

    output = args.processed_demo / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    rgb_out = np.lib.format.open_memmap(
        output / "robot_rgb.npy", mode="w+", dtype=np.uint8,
        shape=(count, height, width, 3),
    )
    depth_out = np.lib.format.open_memmap(
        output / "robot_depth.npy", mode="w+", dtype=np.float16,
        shape=(count, height, width),
    )
    depth_out[:] = np.inf
    mask_out = np.lib.format.open_memmap(
        output / "robot_mask.npy", mode="w+", dtype=bool,
        shape=(count, height, width),
    )
    preview_path = output / "robot_preview.mp4"
    preview = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (width, height))
    source = cv2.VideoCapture(str(video_path))
    source.set(cv2.CAP_PROP_POS_FRAMES, start)

    for out_idx, frame_idx in enumerate(range(start, end)):
        drawables = []
        for side in enabled:
            hand_idx = 1 if side == "right" else 0
            if not valid[hand_idx, frame_idx]:
                continue
            tree, meshes = hand_gl[side]
            qnames = pkl[side]["joint_names"]
            qdict = {name: float(qdata[side][frame_idx, idx])
                     for idx, name in enumerate(qnames)}
            wrist = joints[side][frame_idx, 0]
            r_mano = Rotation.from_rotvec(orient[hand_idx, frame_idx]).as_matrix()
            root, r_cam_hand = hand_root_pose(r_mano, wrist, side_align[side])
            link_poses = compute_fk(tree, qdict, root)
            for link, parts in meshes.items():
                if link in link_poses:
                    drawables.extend((part, link_poses[link]) for part in parts)
            if not args.no_arm:
                arm_poses = _arm_link_poses(
                    side, wrist, r_cam_hand, forearm_sign(embodiment[side])
                )
                for link, parts in arm_gl[side].items():
                    if link in arm_poses:
                        drawables.extend((part, arm_poses[link]) for part in parts)

        rgb, depth, mask = renderer.render(drawables)
        rgb_out[out_idx] = rgb
        depth_out[out_idx] = depth.astype(np.float16)
        mask_out[out_idx] = mask
        ok, background = source.read()
        if ok:
            composed = background.copy()
            composed[mask] = rgb[mask]
            preview.write(composed)
        if (out_idx + 1) % 50 == 0 or out_idx + 1 == count:
            print(f"[frame] {out_idx + 1}/{count} source={frame_idx} "
                  f"robot_px={int(mask.sum())}", flush=True)

    source.release()
    preview.release()
    rgb_out.flush(); depth_out.flush(); mask_out.flush()
    print(f"[ok] {output / 'robot_rgb.npy'}")
    print(f"[ok] {output / 'robot_depth.npy'}")
    print(f"[ok] {output / 'robot_mask.npy'}")
    print(f"[ok] {preview_path}")


if __name__ == "__main__":
    main()
