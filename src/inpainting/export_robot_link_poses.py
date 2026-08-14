"""Export per-frame robot link meshes + poses for an external renderer.

The overlay renderer keeps the robot as URDF links posed by FK, never as a mesh
file, so a Blender/Cycles pass has nothing to import. This writes the two things
such a renderer needs and nothing more:

    meshes      one entry per link: absolute mesh path
    poses       (T, L, 4, 4) link poses in the OpenCV camera frame

RB5-850e arm poses come from rb5_build_overlay_input.py; hand poses are the same
FK the pyrender overlay uses, so the exported robot lines up with it frame for
frame.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from _paths import forearm_sign, urdf_for  # noqa: F401  (mirrors renderer paths)
from urdf_fk import (build_side_align, compute_fk, hand_root_pose,  # type: ignore
                     load_side_urdf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--left_pkl", type=Path, required=True)
    parser.add_argument("--rb5_npz", type=Path, required=True,
                        help="rb5_build_overlay_input.py output (link_poses).")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pose = np.load(args.hawor_npz)
    joints = pose[f"joints_{args.side}"].astype(np.float64)
    global_orient = pose["mano_global_orient"]
    hand_index = 0 if args.side == "left" else 1
    valid = pose["valid"][hand_index].astype(bool)

    with open(args.left_pkl, "rb") as handle:
        retarget = pickle.load(handle)
    qpos = np.asarray(retarget["data"])
    joint_names = list(retarget["joint_names"])

    rb5 = np.load(args.rb5_npz)
    arm_poses = rb5["link_poses"]                    # (T, 7, 4, 4), camera frame
    arm_names = [str(name) for name in rb5["link_names"]]

    side_cfg = load_side_urdf("xhand", args.side)
    side_align = build_side_align({args.side: "xhand"})
    link_meshes = side_cfg["link_meshes"]
    joints_tree = side_cfg["joints"]

    hand_links = sorted(link_meshes)
    mesh_paths, mesh_owner = [], []
    for name in arm_names:
        mesh_paths.append(str(Path(__file__).resolve().parents[2]
                              / "third_party" / "rb5_850e" / "meshes" / "visual"
                              / f"{name}.dae"))
        mesh_owner.append(name)
    hand_mesh_index = {}
    for name in hand_links:
        for order, (mesh, _) in enumerate(link_meshes[name]):
            hand_mesh_index[(name, order)] = len(mesh_paths)
            mesh_paths.append(str(getattr(mesh, "metadata", {}).get("file_path", "")))
            mesh_owner.append(f"{name}#{order}")

    frame_count = min(len(arm_poses), len(qpos), len(joints))
    poses = np.tile(np.eye(4), (frame_count, len(mesh_paths), 1, 1))
    visible = np.zeros((frame_count, len(mesh_paths)), dtype=bool)
    for frame in range(frame_count):
        poses[frame, :len(arm_names)] = arm_poses[frame]
        visible[frame, :len(arm_names)] = True
        if not valid[frame]:
            continue
        qpos_dict = {name: float(qpos[frame, i])
                     for i, name in enumerate(joint_names)}
        rotation = Rotation.from_rotvec(global_orient[hand_index, frame]).as_matrix()
        root, _ = hand_root_pose(rotation, joints[frame, 0, :],
                                 side_align[args.side])
        link_transforms = compute_fk(joints_tree, qpos_dict, root)
        for name in hand_links:
            if name not in link_transforms:
                continue
            for order, (_, visual) in enumerate(link_meshes[name]):
                index = hand_mesh_index[(name, order)]
                poses[frame, index] = link_transforms[name] @ visual
                visible[frame, index] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, poses=poses.astype(np.float32), visible=visible,
             mesh_paths=np.asarray(mesh_paths), mesh_owner=np.asarray(mesh_owner))
    print(f"[ok] wrote {args.output}: {frame_count} frames, "
          f"{len(mesh_paths)} meshes ({len(arm_names)} arm links)")
    print(json.dumps({"arm_links": arm_names, "hand_links": hand_links}, indent=1))


if __name__ == "__main__":
    main()
