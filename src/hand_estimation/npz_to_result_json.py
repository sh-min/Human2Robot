"""Convert HaWoR retarget_input.npz -> per-frame result.json.

`extract_for_retarget.py` saves MANO params in axis-angle form inside a single
.npz. The skill_classifier pipeline (`data_preprocess/preprocess.py` and
`skill_classifier/infer_long_horizon.py`) expects a per-frame result.json with
MANO params as rotation matrices. This script bridges the two.

Usage:
    python npz_to_result_json.py \
        --npz     <recording>/rgb_hawor/retarget_input.npz \
        --rgb_dir <recording>/rgb \
        --img_glob 'rgb_frame*.png' \
        --out     <recording>/result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def aa_to_rotmat(aa):
    """Axis-angle (3,) -> rotation matrix (3, 3) via Rodrigues."""
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = aa / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float64)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="retarget_input.npz")
    p.add_argument("--rgb_dir", required=True, help="RGB frame directory")
    p.add_argument("--img_glob", default="*.png", help="Glob inside --rgb_dir")
    p.add_argument("--out", required=True, help="Output result.json path")
    args = p.parse_args()

    d = np.load(args.npz)
    joints = [d["joints_left"], d["joints_right"]]   # [T, 21, 3] each
    gos = d["mano_global_orient"]                    # [2, T, 3]
    hps = d["mano_hand_pose"]                        # [2, T, 15, 3]
    valid = d["valid"]                               # [2, T] bool/0-1
    T_npz = int(joints[0].shape[0])

    frame_files = sorted(Path(args.rgb_dir).glob(args.img_glob))
    if not frame_files:
        raise RuntimeError(f"No frames matched {args.img_glob!r} in {args.rgb_dir}")
    frame_names = [f.stem for f in frame_files]
    T = min(T_npz, len(frame_names))

    out = {}
    for t in range(T):
        hands = []
        for hi in (0, 1):  # 0 = left, 1 = right
            if not bool(valid[hi, t]):
                continue
            go = aa_to_rotmat(gos[hi, t])                              # (3, 3)
            hp = np.stack([aa_to_rotmat(hps[hi, t, j]) for j in range(15)])  # (15, 3, 3)
            hands.append({
                "is_right": int(hi),
                "kpts_3d": joints[hi][t].astype(np.float32).tolist(),
                "mano_params": {
                    "global_orient": [go.tolist()],  # [1, 3, 3] to match consumer
                    "hand_pose": hp.tolist(),        # [15, 3, 3]
                },
            })
        out[frame_names[t]] = hands

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    n_valid = sum(1 for v in out.values() if v)
    print(f"[done] {args.out}  (frames={T}, frames_with_hand={n_valid})")


if __name__ == "__main__":
    main()
