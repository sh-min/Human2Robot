"""Procrustes-fit R_MANO (the rotation that takes a vector in MANO canonical
wrist frame to a robot hand's wrist link frame).

For each hand:
  A = wrist-relative MCP knuckle positions of MANO at zero pose (5, 3)
  B = wrist-relative MCP link origins of the robot hand at q=0 (5, 3)
  R = argmin_R || A @ R - B ||_F   (SVD, proper rotation)

Saves to:
  src/retargeting/assets/R_mano_{embodiment}_{hand}.npy

Run in hawor env (needs MANO via HaWoR):
    conda activate hawor
    cd <repo_root>/src/retargeting
    python compute_R_mano_xhand.py --embodiment xhand
    python compute_R_mano_xhand.py --embodiment inspire --hand left
"""
import argparse
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
HAWOR_DIR = os.path.join(REPO_ROOT, "third_party", "HaWoR")
sys.path.insert(0, THIS_DIR)      # for _paths, before the chdir below
sys.path.insert(0, HAWOR_DIR)
os.chdir(HAWOR_DIR)

import xml.etree.ElementTree as ET

import torch
from hawor.utils.process import run_mano, run_mano_left

from _paths import EMBODIMENT_NAMES, uses_offset, urdf_path

ASSETS = os.path.join(REPO_ROOT, "src", "retargeting", "assets")

# MANO joints (mediapipe-21 layout as HaWoR outputs):
#   0=wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
# Proximal joint of each finger:
MANO_MCP_IDX = [1, 5, 9, 13, 17]  # thumb_cmc, idx_mcp, mid_mcp, ring_mcp, pinky_mcp

# Wrist link and the five first-finger links (closest to wrist), thumb->pinky,
# matching MANO_MCP_IDX order. "{hand}" is substituted with right/left.
MCP_SPEC = {
    "xhand": dict(
        wrist_link="{hand}_hand_link",
        mcp_links=[
            "{hand}_hand_thumb_bend_link",
            "{hand}_hand_index_bend_link",
            "{hand}_hand_mid_link1",
            "{hand}_hand_ring_link1",
            "{hand}_hand_pinky_link1",
        ],
    ),
    "inspire": dict(
        wrist_link="base",   # note: no chirality prefix; left/right names coincide
        mcp_links=[
            "thumb_proximal_base",
            "index_proximal",
            "middle_proximal",
            "ring_proximal",
            "pinky_proximal",
        ],
    ),
}


def get_mano_mcp_canonical(hand):
    """MANO at zero pose (T-pose) — return wrist-relative MCP knuckles (5, 3)."""
    B, T = 1, 1
    trans = torch.zeros(B, T, 3)
    root_orient = torch.zeros(B, T, 3)
    hand_pose = torch.zeros(B, T, 15, 3)
    betas = torch.zeros(B, T, 10)
    fn = run_mano if hand == "right" else run_mano_left
    out = fn(trans, root_orient, hand_pose, betas=betas)
    joints = out["joints"][0, 0].cpu().numpy()  # (21, 3)
    mcp = joints[MANO_MCP_IDX]                  # (5, 3)
    wrist = joints[0]                           # (3,)
    return (mcp - wrist).astype(np.float64)


def _rpy_to_matrix(rpy):
    """URDF rpy (extrinsic x-y-z) -> rotation matrix, i.e. Rz @ Ry @ Rx."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _joint_origin_T(j):
    o = j.find("origin")
    xyz_s = o.attrib.get("xyz", "0 0 0") if o is not None else "0 0 0"
    rpy_s = o.attrib.get("rpy", "0 0 0") if o is not None else "0 0 0"
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix([float(s) for s in rpy_s.split()])
    T[:3, 3] = [float(s) for s in xyz_s.split()]
    return T


def get_robot_mcp(embodiment, hand):
    """Robot hand at q=0: MCP link origins expressed in the wrist link frame,
    (5, 3), thumb->pinky.

    Full FK from the URDF root rather than reading each MCP joint's origin
    directly: that shortcut only holds when every MCP joint's parent *is* the
    wrist link (true for xhand, false for inspire, whose proximals hang off
    hand_base_link). At q=0 every joint rotation is identity, so this is just
    a composition of link origin transforms.
    """
    spec = MCP_SPEC[embodiment]
    wrist_link = spec["wrist_link"].format(hand=hand)
    mcp_links = [ln.format(hand=hand) for ln in spec["mcp_links"]]

    root = ET.parse(urdf_path(embodiment, hand)).getroot()
    joints = {}
    for j in root.findall("joint"):
        parent, child = j.find("parent"), j.find("child")
        if parent is None or child is None:
            continue
        joints[child.attrib["link"]] = (parent.attrib["link"], _joint_origin_T(j))

    children = set(joints)
    parents = {p for p, _ in joints.values()}
    root_link = next(iter(parents - children))

    link_T = {root_link: np.eye(4)}
    pending = dict(joints)
    while pending:
        progressed = False
        for ch, (par, T) in list(pending.items()):
            if par in link_T:
                link_T[ch] = link_T[par] @ T
                del pending[ch]
                progressed = True
        if not progressed:
            raise RuntimeError(f"disconnected URDF links: {sorted(pending)}")

    missing = [ln for ln in [wrist_link] + mcp_links if ln not in link_T]
    if missing:
        raise KeyError(f"{embodiment}/{hand}: links not in URDF: {missing}")

    T_wrist_inv = np.linalg.inv(link_T[wrist_link])
    return np.stack(
        [(T_wrist_inv @ link_T[ln])[:3, 3] for ln in mcp_links], axis=0
    ).astype(np.float64)


def procrustes_rotation(A, B):
    """Find R minimizing ||A @ R - B||_F, ensuring proper rotation det=+1."""
    M = A.T @ B
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", default="xhand", choices=EMBODIMENT_NAMES)
    ap.add_argument("--hand", default="both", choices=["right", "left", "both"])
    args = ap.parse_args()

    emb = args.embodiment
    hands = ("right", "left") if args.hand == "both" else (args.hand,)
    want_offset = uses_offset(emb)

    os.makedirs(ASSETS, exist_ok=True)
    for hand in hands:
        A = get_mano_mcp_canonical(hand)
        B = get_robot_mcp(emb, hand)

        # Uncentred fit — see EMBODIMENTS in _paths.py for why this beats
        # fitting on centred knuckles even when an offset is also used.
        R = procrustes_rotation(A, B)
        # Residual translation between the rotated human knuckles and the
        # robot's. Left at zero for embodiments whose root already sits near the
        # MANO wrist, so their behaviour is untouched.
        offset = (A @ R).mean(0) - B.mean(0) if want_offset else np.zeros(3)

        residual = np.linalg.norm((A @ R) - offset - B, axis=1)
        print(f"\n=== {emb} / {hand}  (offset={'fitted' if want_offset else 'none'}) ===")
        print(f"MANO canonical MCP (mm):")
        for k, p in zip(("thb", "idx", "mid", "rng", "pky"), A * 1000):
            print(f"  {k}: ({p[0]:+7.2f}, {p[1]:+7.2f}, {p[2]:+7.2f})")
        print(f"{emb} MCP (mm):")
        for k, p in zip(("thb", "idx", "mid", "rng", "pky"), B * 1000):
            print(f"  {k}: ({p[0]:+7.2f}, {p[1]:+7.2f}, {p[2]:+7.2f})")
        print(f"R_mano_{emb}_{hand} =")
        print(np.array2string(R, precision=4, suppress_small=True))
        print(f"per-knuckle residual (mm): {(residual * 1000).round(2).tolist()}")
        print(f"mean residual: {residual.mean() * 1000:.2f} mm")

        out = os.path.join(ASSETS, f"R_mano_{emb}_{hand}.npy")
        np.save(out, R)
        print(f"saved {out}")

        if want_offset:
            out_off = os.path.join(ASSETS, f"wrist_offset_{emb}_{hand}.npy")
            np.save(out_off, offset)
            print(f"wrist offset (mm): {(offset * 1000).round(2).tolist()}"
                  f"  |d| = {np.linalg.norm(offset) * 1000:.2f}")
            print(f"saved {out_off}")


if __name__ == "__main__":
    main()
