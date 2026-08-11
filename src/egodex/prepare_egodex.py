"""Adapt an EgoDex episode to the skill2policy pipeline input format.

EgoDex ships each episode as a paired `N.mp4` + `N.hdf5` (1080p@30Hz).
The pipeline (run_pipeline.sh / run_layered.py) instead expects a directory
of `frame_*.jpg` images plus a camera focal length in pixels. This script
bridges the two:

    1. Decode `N.mp4` to `<out>/frame_%05d.jpg`.
    2. Read `camera/intrinsic` (3x3) from `N.hdf5` and report fx (== fy).

The focal length is constant across all EgoDex files at a given resolution,
but we read it per-episode so the value is always correct.

Usage:
    python prepare_egodex.py --episode /data/.../test/add_remove_lid/0 \
        --out /data/.../egodex_prepared/add_remove_lid/0
    # prints: IMG_FOCAL=736.6339
"""
import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np


def read_focal(hdf5_path: Path) -> float:
    with h5py.File(hdf5_path, "r") as f:
        K = np.asarray(f["camera"]["intrinsic"], dtype=np.float64)  # (3, 3)
    fx, fy = K[0, 0], K[1, 1]
    if abs(fx - fy) / max(fx, fy) > 1e-3:
        print(f"[warn] fx={fx:.3f} != fy={fy:.3f}; using fx")
    return float(fx)


# Arm chain to seed SAM2 along the limb (elbow -> forearm -> wrist -> fingers).
_FINGERS = ["Index", "Middle", "Ring", "Little", "Thumb"]
_FINGER_PARTS = ["Knuckle", "Tip"]


def _arm_joint_names(side: str, available: set) -> list:
    names = [side + "Arm", side + "Forearm", side + "Hand"]
    for fg in _FINGERS:
        for part in _FINGER_PARTS:
            names.append(f"{side}{fg}Finger{part}")
    return [n for n in names if n in available]


def export_arm_kpts(hdf5_path: Path, out_dir: Path, scale: float) -> None:
    """Project the EgoDex arm-chain joints to 2D (at the output frame scale) and
    save <out_dir>/arm_kpts_2d.npz with per-side (T,J,2) + (T,J) validity.

    Projection: world -> camera via inv(camera extrinsic); EgoDex camera frame is
    +Z-forward / +Y-down (OpenCV-style, no ARKit flip), then pinhole with the
    hdf5 intrinsic. Pixels are scaled by `scale` (= out_height / orig_height).
    """
    with h5py.File(hdf5_path, "r") as f:
        K = np.asarray(f["camera"]["intrinsic"], dtype=np.float64)
        Tcam = np.asarray(f["transforms"]["camera"], dtype=np.float64)  # (T,4,4)
        tg = f["transforms"]
        avail = set(tg.keys())
        T = Tcam.shape[0]
        W = int(round(2 * K[0, 2] * scale))
        H = int(round(2 * K[1, 2] * scale))
        out = {}
        for side in ("left", "right"):
            names = _arm_joint_names(side, avail)
            Jw = np.stack([np.asarray(tg[n], dtype=np.float64) for n in names], axis=1)
            A = np.full((T, len(names), 2), np.nan)
            for t in range(T):
                Cinv = np.linalg.inv(Tcam[t])
                for k in range(len(names)):
                    x, y, z = (Cinv @ np.append(Jw[t, k][:3, 3], 1.0))[:3]
                    if z <= 1e-4:
                        continue
                    A[t, k] = [(K[0, 0] * x / z + K[0, 2]) * scale,
                               (K[1, 1] * y / z + K[1, 2]) * scale]
            valid = (~np.isnan(A[..., 0]) & (A[..., 0] >= 0) & (A[..., 0] < W)
                     & (A[..., 1] >= 0) & (A[..., 1] < H))
            out[side] = A.astype(np.float32)
            out[side + "_valid"] = valid
            out[side + "_joints"] = np.array(names)
    np.savez(out_dir / "arm_kpts_2d.npz", **out)


def video_height(mp4_path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of", "csv=p=0", str(mp4_path)],
        check=True, capture_output=True, text=True,
    )
    return int(out.stdout.strip())


def extract_frames(mp4_path: Path, out_dir: Path, height: int | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # -qscale:v 2 ~ visually lossless JPEG; %05d matches the frame_*.jpg glob.
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path)]
    if height is not None:
        cmd += ["-vf", f"scale=-2:{height}"]  # -2 keeps aspect, forces even width
    cmd += ["-qscale:v", "2", str(out_dir / "frame_%05d.jpg")]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("frame_*.jpg")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True,
                    help="episode path WITHOUT extension (e.g. .../add_remove_lid/0)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory for frame_*.jpg")
    ap.add_argument("--height", type=int, default=None,
                    help="downscale frames to this height (keeps aspect). "
                         "Focal is scaled by height/orig_height to stay consistent.")
    args = ap.parse_args()

    mp4 = args.episode.with_suffix(".mp4")
    hdf5 = args.episode.with_suffix(".hdf5")
    for p in (mp4, hdf5):
        if not p.exists():
            raise FileNotFoundError(p)

    focal = read_focal(hdf5)
    scale = 1.0
    if args.height is not None:
        orig_h = video_height(mp4)
        scale = args.height / orig_h
        focal *= scale
    n = extract_frames(mp4, args.out, height=args.height)
    export_arm_kpts(hdf5, args.out, scale)
    print(f"[ok] {n} frames -> {args.out}")
    print(f"[ok] arm keypoints -> {args.out / 'arm_kpts_2d.npz'}")
    print(f"IMG_FOCAL={focal:.4f}")


if __name__ == "__main__":
    main()
