"""Procrustes-fit R_MANO_XHAND (the rotation that takes a vector in MANO
canonical wrist frame to xhand wrist link frame).

For each hand:
  A = wrist-relative MCP knuckle positions of MANO at zero pose (5, 3)
  B = wrist-relative MCP joint origins of xhand at q=0 (5, 3)
  R = argmin_R || A @ R - B ||_F   (SVD, proper rotation)

Saves to:
  src/retargeting/assets/R_mano_xhand_{right,left}.npy

Run in hawor env (needs MANO via HaWoR):
    conda activate hawor
    cd <repo_root>/src/retargeting
    python compute_R_mano_xhand.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HAWOR_DIR = os.path.join(REPO_ROOT, "third_party", "HaWoR")
sys.path.insert(0, HAWOR_DIR)
os.chdir(HAWOR_DIR)

import xml.etree.ElementTree as ET

import torch
from hawor.utils.process import run_mano, run_mano_left

ASSETS = os.path.join(REPO_ROOT, "src", "retargeting", "assets")

# MANO joints (mediapipe-21 layout as HaWoR outputs):
#   0=wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
# Proximal joint of each finger:
MANO_MCP_IDX = [1, 5, 9, 13, 17]  # thumb_cmc, idx_mcp, mid_mcp, ring_mcp, pinky_mcp

# Corresponding xhand first finger links (closest to wrist), same order:
XHAND_MCP_LINKS = lambda h: [
    f"{h}_hand_thumb_bend_link",
    f"{h}_hand_index_bend_link",
    f"{h}_hand_mid_link1",
    f"{h}_hand_ring_link1",
    f"{h}_hand_pinky_link1",
]


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


def get_xhand_mcp(hand):
    """xhand at q=0: MCP link origins in wrist link frame (5, 3).

    Each MCP link's joint origin is given in URDF relative to its parent
    (right_hand_link == wrist link). Since all MCP joints have parent =
    wrist link and we are at q=0, the joint origin XYZ is already the
    point in the wrist frame.
    """
    urdf = os.path.join(REPO_ROOT, "src", "retargeting", "assets", "xhand",
                        f"xhand_{hand}.urdf")
    tree = ET.parse(urdf)
    root = tree.getroot()

    wanted_children = set(XHAND_MCP_LINKS(hand))
    name_to_origin = {}
    for j in root.findall("joint"):
        child = j.find("child")
        if child is None:
            continue
        ch = child.attrib.get("link")
        if ch not in wanted_children:
            continue
        o = j.find("origin")
        xyz = o.attrib.get("xyz", "0 0 0") if o is not None else "0 0 0"
        name_to_origin[ch] = np.array([float(s) for s in xyz.split()],
                                      dtype=np.float64)

    return np.stack([name_to_origin[ln] for ln in XHAND_MCP_LINKS(hand)], axis=0)


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
    os.makedirs(ASSETS, exist_ok=True)
    for hand in ("right", "left"):
        A = get_mano_mcp_canonical(hand)
        B = get_xhand_mcp(hand)
        R = procrustes_rotation(A, B)
        residual = np.linalg.norm(A @ R - B, axis=1)
        print(f"\n=== {hand} ===")
        print(f"MANO canonical MCP (mm):")
        for k, p in zip(("thb", "idx", "mid", "rng", "pky"), A * 1000):
            print(f"  {k}: ({p[0]:+7.2f}, {p[1]:+7.2f}, {p[2]:+7.2f})")
        print(f"xhand MCP (mm):")
        for k, p in zip(("thb", "idx", "mid", "rng", "pky"), B * 1000):
            print(f"  {k}: ({p[0]:+7.2f}, {p[1]:+7.2f}, {p[2]:+7.2f})")
        print(f"R_MANO_XHAND ({hand}) =")
        print(np.array2string(R, precision=4, suppress_small=True))
        print(f"per-knuckle residual (mm): {(residual * 1000).round(2).tolist()}")
        print(f"mean residual: {residual.mean() * 1000:.2f} mm")

        out = os.path.join(ASSETS, f"R_mano_xhand_{hand}.npy")
        np.save(out, R)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
