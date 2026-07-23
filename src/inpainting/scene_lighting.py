"""Shared scene-lighting model for the robot overlay.

Two stages need to agree on the light: the render stage (stage 5) shades the
robot with it, and the compositor (stage 10) casts the contact shadow with it.
If they disagree, highlights and shadow point different ways and the illusion
breaks — so the light direction lives here, imported by both.

Frames are OpenCV camera convention (x-right, y-down, z-forward), metres.
"""
import numpy as np

# Light *travel* direction in the camera (CV) frame: where photons go, not where
# the lamp is. Default models a head-mounted egocentric camera in a room lit
# from above and slightly ahead of the wearer, so light travels down (+y, since
# y is down), forward (+z), and a touch to the right (+x). Tunable per demo via
# the CLI, but this is a sane, stable default that needs no per-frame estimation
# (single-image light-direction estimation is too unreliable to trust).
LIGHT_DIR_CAM = np.array([0.25, 0.85, 0.45], dtype=np.float64)
LIGHT_DIR_CAM /= np.linalg.norm(LIGHT_DIR_CAM)


def light_dir_cam(vec=None) -> np.ndarray:
    """Normalised light travel direction in the CV camera frame."""
    v = np.asarray(LIGHT_DIR_CAM if vec is None else vec, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def estimate_illumination(frames, sample_stride: int = 30,
                          bright_pct: float = 60.0) -> np.ndarray:
    """White-balance tint (r,g,b in [0,1], max channel = 1) of the room light.

    Robust and temporally stable by design: pooled over frames sampled across
    the clip (room colour is ~constant, so a global tint avoids per-frame
    flicker), using only the brighter pixels — those are lit surfaces that
    carry the illuminant colour, whereas dark pixels are mostly albedo.

    *frames* is (T,H,W,3) or a single (H,W,3), uint8 or float.
    """
    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[None]
    arr = arr[::max(1, sample_stride)].astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    px = arr.reshape(-1, 3)
    lum = px @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    keep = px[lum >= np.percentile(lum, bright_pct)]
    tint = keep.mean(0) if len(keep) else px.mean(0)
    return (tint / (tint.max() + 1e-6)).astype(np.float64)


def directional_light_pose(dir_cam=None, T_cv2gl=None) -> np.ndarray:
    """4x4 pose for a pyrender DirectionalLight travelling along *dir_cam*.

    pyrender emits a DirectionalLight along its node's local -z, and the render
    scene is in OpenGL frame, so the CV travel direction is converted with the
    same T_CV2GL the meshes use, then a rotation is built mapping -z onto it.
    """
    if T_cv2gl is None:
        T_cv2gl = np.diag([1.0, -1.0, -1.0])
    d_gl = T_cv2gl @ light_dir_cam(dir_cam)
    d_gl = d_gl / (np.linalg.norm(d_gl) + 1e-12)

    # Rotation taking local -z to d_gl (Rodrigues from the -z / d_gl pair).
    z_neg = np.array([0.0, 0.0, -1.0])
    v = np.cross(z_neg, d_gl)
    s = np.linalg.norm(v)
    c = float(np.dot(z_neg, d_gl))
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    pose = np.eye(4)
    pose[:3, :3] = R
    return pose
