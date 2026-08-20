"""Render an RB5-850e + XHand overlay without Isaac Sim.

The adapter input is produced by :mod:`rb5_build_overlay_input`.  Arm FK is
evaluated directly from the RB5 URDF and the hand FK reuses the existing
XHand URDF helpers.  Pyrender runs two passes per frame:

* a lit RGB/depth pass for the robot overlay;
* a flat segmentation pass for exact, visible per-finger and anatomical
  surface labels.

The RB5 visual meshes are Collada files, but the project environments do not
ship ``pycollada``.  This renderer therefore uses the matching URDF collision
STLs and gives them a neutral robot material.  The XHand keeps its visual
meshes and colours.

Full render outputs (under ``<out>/overlay_processor``)::

    robot_rgb.npy             uint8   (T,H,W,3), RGB
    robot_depth.npy           float16 (T,H,W), metres, +inf off robot
    robot_mask.npy            bool    (T,H,W)
    robot_finger_labels.npy   uint8   (T,H,W), 0=other, 1..5=fingers
    robot_finger_surface_labels.npy
                              uint8   (T,H,W), packed 0=other, 1..15
    robot_finger_mask.npy     bool    (T,H,W)
    manifest.json

The packed anatomical label is ``(finger_id - 1) * 3 + surface_id``, where
``surface_id`` is 1=palmar/front, 2=lateral/side, 3=dorsal/back.  Face normals
are classified in each XHand link frame, so the labels describe the hand and
do not change when the camera moves.

For a cheap placement check, ``--preview`` respects ``--start``/``--n`` and
writes ``<out>/rb5_preview.png`` over ``--background`` without allocating the
full arrays.  A non-preview render deliberately requires the complete frame
range so an incomplete array cannot be mistaken for a valid compositor input.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Prefer a regular GL context when an X display is available (for example
# under xvfb-run).  Force EGL only for genuinely headless execution.
if "PYOPENGL_PLATFORM" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh

from atomic_directory_publish import publish_directory
from rb5_finger_semantics import FINGER_LABEL_IDS, xhand_finger_link_names
from render_xhand_overlay import compute_fk, load_side_urdf


REPO = Path(__file__).resolve().parents[2]
RB5_URDF = REPO / "third_party" / "rb5_850e" / "rb5_850e.urdf"
RB5_COLLISION_DIR = REPO / "third_party" / "rb5_850e" / "meshes" / "collision"
RB5_JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
RB5_LINK_NAMES = tuple(f"link{index}" for index in range(7))
RB5_ARM_COLOR = (0.72, 0.72, 0.72)
RB5_LIGHT_COLOR = (1.0, 1.0, 1.0)
SURFACE_LABEL_IDS = {
    "palmar": 1,
    "lateral": 2,
    "dorsal": 3,
}
SURFACE_NORMAL_DOT_THRESHOLD = 0.5
PALMAR_NORMAL_AXES = {
    "right": {
        "thumb": (0.0, 0.0, -1.0),
        "other": (0.0, 1.0, 0.0),
    },
    "left": {
        "thumb": (0.0, 0.0, -1.0),
        "other": (0.0, -1.0, 0.0),
    },
}
MAX_PACKED_SURFACE_LABEL = (
    len(FINGER_LABEL_IDS) * len(SURFACE_LABEL_IDS)
)
T_CV2GL = np.diag([1.0, -1.0, -1.0])
OFFSCREEN_POSE = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 5.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _numbers(value: str | None, count: int, default: tuple[float, ...]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    parsed = np.asarray([float(part) for part in value.split()], dtype=np.float64)
    if parsed.shape != (count,):
        raise ValueError(f"expected {count} values, got {value!r}")
    return parsed


def _make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    # Kept local so importing the renderer does not require scipy.
    rx, ry, rz = rpy
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    rotation = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def load_rb5_urdf() -> tuple[dict[str, dict], dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]]]:
    """Load RB5 joints and collision STL geometry in URDF link frames."""
    if not RB5_URDF.is_file():
        raise FileNotFoundError(RB5_URDF)
    root = ET.parse(RB5_URDF).getroot()

    joints: dict[str, dict] = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        xyz = _numbers(origin.get("xyz") if origin is not None else None, 3, (0, 0, 0))
        rpy = _numbers(origin.get("rpy") if origin is not None else None, 3, (0, 0, 0))
        axis = element.find("axis")
        joints[element.get("name")] = {
            "type": element.get("type"),
            "parent": element.find("parent").get("link"),
            "child": element.find("child").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": _numbers(
                axis.get("xyz") if axis is not None else None,
                3,
                (1, 0, 0),
            ),
        }

    link_meshes: dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]] = {}
    for link in root.findall("link"):
        link_name = link.get("name")
        stl_path = RB5_COLLISION_DIR / f"{link_name}.stl"
        if not stl_path.is_file():
            continue
        mesh = trimesh.load(str(stl_path), force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"unsupported RB5 mesh at {stl_path}: {type(mesh)}")

        collision = link.find("collision")
        origin = collision.find("origin") if collision is not None else None
        visual_transform = _make_transform(
            _numbers(origin.get("xyz") if origin is not None else None, 3, (0, 0, 0)),
            _numbers(origin.get("rpy") if origin is not None else None, 3, (0, 0, 0)),
        )
        mesh_element = (
            collision.find("geometry/mesh") if collision is not None else None
        )
        if mesh_element is not None and mesh_element.get("scale"):
            mesh.apply_scale(_numbers(mesh_element.get("scale"), 3, (1, 1, 1)))
        link_meshes[link_name] = [(mesh, visual_transform)]

    missing_joints = sorted(set(RB5_JOINT_NAMES) - set(joints))
    missing_links = sorted({f"link{i}" for i in range(7)} - set(link_meshes))
    if missing_joints or missing_links:
        raise RuntimeError(
            f"incomplete RB5 URDF assets: joints={missing_joints}, links={missing_links}"
        )
    return joints, link_meshes


def finger_link_fingers(side: str) -> dict[str, str]:
    """Map every visible finger link, including fixed tips, to its finger."""
    groups = {name: list(links) for name, links in xhand_finger_link_names(side).items()}
    prefix = f"{side}_hand_"
    groups["thumb"].append(prefix + "thumb_rota_tip")
    groups["index"].append(prefix + "index_rota_tip")
    groups["middle"].append(prefix + "mid_tip")
    groups["ring"].append(prefix + "ring_tip")
    groups["pinky"].append(prefix + "pinky_tip")
    return {
        link_name: finger
        for finger, link_names in groups.items()
        for link_name in link_names
    }


def finger_link_labels(side: str) -> dict[str, int]:
    """Map every visible finger link, including fixed tips, to label 1..5."""
    return {
        link_name: FINGER_LABEL_IDS[finger]
        for link_name, finger in finger_link_fingers(side).items()
    }


def palmar_normal_axis(side: str, finger: str) -> np.ndarray:
    """Return the anatomical palmar normal in an XHand link frame.

    The four non-thumb fingers extend along local ``-Z``.  Their pad normal is
    local ``+Y`` for the right hand and ``-Y`` for the left hand.  The thumb
    extends along local ``+X`` and has local ``-Z`` as its pad normal on both
    sides.  Keeping this definition link-local makes the classification
    independent of FK and camera pose.
    """
    if side not in PALMAR_NORMAL_AXES:
        raise ValueError(f"unsupported XHand side: {side!r}")
    if finger not in FINGER_LABEL_IDS:
        raise ValueError(f"unsupported XHand finger: {finger!r}")
    key = "thumb" if finger == "thumb" else "other"
    return np.asarray(PALMAR_NORMAL_AXES[side][key], dtype=np.float64)


def classify_finger_face_surfaces(
    mesh: trimesh.Trimesh,
    visual_transform: np.ndarray,
    side: str,
    finger: str,
    dot_threshold: float = SURFACE_NORMAL_DOT_THRESHOLD,
) -> np.ndarray:
    """Classify mesh faces as palmar, lateral, or dorsal.

    ``mesh.face_normals`` are expressed in the visual-mesh frame.  A URDF
    visual origin may rotate that frame relative to the link, so normals are
    transformed with the inverse-transpose of its linear transform before the
    anatomical dot product is evaluated.  Degenerate/non-finite faces are
    conservatively labelled lateral.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected trimesh.Trimesh, got {type(mesh)}")
    transform = np.asarray(visual_transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("visual_transform must be a finite 4x4 matrix")
    if not np.isfinite(dot_threshold) or not 0.0 < dot_threshold <= 1.0:
        raise ValueError("dot_threshold must be in (0, 1]")

    normals_mesh = np.asarray(mesh.face_normals, dtype=np.float64)
    if normals_mesh.shape != (len(mesh.faces), 3):
        raise ValueError(
            f"mesh face-normal shape {normals_mesh.shape}, "
            f"expected {(len(mesh.faces), 3)}"
        )
    labels = np.full(
        len(mesh.faces),
        SURFACE_LABEL_IDS["lateral"],
        dtype=np.uint8,
    )
    if len(labels) == 0:
        return labels

    try:
        # For row-vector normals this is equivalent to
        # (inverse(A).T @ normal_column).T.
        normals_link = normals_mesh @ np.linalg.inv(transform[:3, :3])
    except np.linalg.LinAlgError as exc:
        raise ValueError("visual_transform has a singular linear part") from exc
    lengths = np.linalg.norm(normals_link, axis=1)
    valid = np.isfinite(normals_link).all(axis=1) & np.isfinite(lengths) & (lengths > 0)
    normals_link[valid] /= lengths[valid, None]
    dots = np.full(len(labels), np.nan, dtype=np.float64)
    dots[valid] = normals_link[valid] @ palmar_normal_axis(side, finger)
    labels[dots >= dot_threshold] = SURFACE_LABEL_IDS["palmar"]
    labels[dots <= -dot_threshold] = SURFACE_LABEL_IDS["dorsal"]
    return labels


def pack_finger_surface_label(finger_label: int, surface_label: int) -> int:
    """Pack a finger ID (1..5) and anatomical surface ID (1..3)."""
    if finger_label not in FINGER_LABEL_IDS.values():
        raise ValueError(f"invalid finger label: {finger_label}")
    if surface_label not in SURFACE_LABEL_IDS.values():
        raise ValueError(f"invalid surface label: {surface_label}")
    return (finger_label - 1) * len(SURFACE_LABEL_IDS) + surface_label


def decode_packed_finger_labels(packed: np.ndarray) -> np.ndarray:
    """Decode packed anatomical IDs to the existing 0..5 finger contract."""
    values = np.asarray(packed)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"packed surface labels must be integer, got {values.dtype}")
    if values.size and (
        int(values.min()) < 0 or int(values.max()) > MAX_PACKED_SURFACE_LABEL
    ):
        raise ValueError(
            f"packed surface labels must be in 0..{MAX_PACKED_SURFACE_LABEL}"
        )
    decoded = np.zeros(values.shape, dtype=np.uint8)
    foreground = values > 0
    decoded[foreground] = (
        (values[foreground].astype(np.int64) - 1) // len(SURFACE_LABEL_IDS) + 1
    ).astype(np.uint8)
    return decoded


def validate_packed_surface_fingers(
    packed: np.ndarray,
    finger_labels: np.ndarray,
    context: str = "packed surface labels",
) -> None:
    """Require packed IDs to decode exactly to a finger-label image/array."""
    expected = np.asarray(finger_labels)
    decoded = decode_packed_finger_labels(packed)
    if decoded.shape != expected.shape:
        raise ValueError(
            f"{context} shape {decoded.shape}, expected {expected.shape}"
        )
    if not np.issubdtype(expected.dtype, np.integer):
        raise TypeError(f"finger labels must be integer, got {expected.dtype}")
    if not np.array_equal(decoded, expected):
        mismatch = int(np.count_nonzero(decoded != expected))
        raise RuntimeError(
            f"{context} do not decode to finger labels "
            f"({mismatch} mismatched pixels)"
        )


def align_packed_surfaces_to_finger_labels(
    packed: np.ndarray,
    finger_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Repack rendered surfaces onto an authoritative existing finger map.

    Splitting a triangle mesh into three draw primitives can change a handful
    of raster boundary pixels relative to the old unsplit segmentation pass.
    Surface-only backfill must not silently change the established finger map,
    so it reuses the rendered anatomical surface only where the rendered and
    existing finger IDs agree.  Any newly exposed or finger-disagreeing boundary
    pixel is conservatively assigned lateral/side.
    """
    values = np.asarray(packed)
    expected = np.asarray(finger_labels)
    generated = decode_packed_finger_labels(values)
    if generated.shape != expected.shape:
        raise ValueError(
            f"packed surface shape {generated.shape}, expected {expected.shape}"
        )
    if not np.issubdtype(expected.dtype, np.integer):
        raise TypeError(f"finger labels must be integer, got {expected.dtype}")
    if expected.size and (
        int(expected.min()) < 0
        or int(expected.max()) > max(FINGER_LABEL_IDS.values())
    ):
        raise ValueError("finger labels contain values outside 0..5")

    foreground = expected > 0
    rendered_surface = np.zeros(values.shape, dtype=np.uint8)
    rendered = values > 0
    rendered_surface[rendered] = (
        (values[rendered].astype(np.int64) - 1) % len(SURFACE_LABEL_IDS) + 1
    ).astype(np.uint8)
    missing_surface = foreground & ~rendered
    finger_mismatch = foreground & rendered & (generated != expected)
    fallback_surface = missing_surface | finger_mismatch
    rendered_surface[fallback_surface] = SURFACE_LABEL_IDS["lateral"]

    aligned = np.zeros(values.shape, dtype=np.uint8)
    aligned[foreground] = (
        (expected[foreground].astype(np.int64) - 1) * len(SURFACE_LABEL_IDS)
        + rendered_surface[foreground]
    ).astype(np.uint8)
    validate_packed_surface_fingers(
        aligned,
        expected,
        context="aligned packed surface labels",
    )
    return aligned, {
        "raster_mismatch_pixels": int(np.count_nonzero(generated != expected)),
        "missing_surface_fallback_pixels": int(np.count_nonzero(missing_surface)),
        "finger_mismatch_fallback_pixels": int(np.count_nonzero(finger_mismatch)),
        "lateral_fallback_pixels": int(np.count_nonzero(fallback_surface)),
    }


def finger_surface_manifest_contract() -> dict:
    """Return the JSON-serialisable packed anatomical label contract."""
    return {
        "filename": "robot_finger_surface_labels.npy",
        "dtype": "uint8",
        "background": 0,
        "valid_range": [0, MAX_PACKED_SURFACE_LABEL],
        "surface_ids": dict(SURFACE_LABEL_IDS),
        "packing": "(finger_id - 1) * 3 + surface_id",
        "decode": {
            "finger_id": "((packed_id - 1) // 3) + 1",
            "surface_id": "((packed_id - 1) % 3) + 1",
            "condition": "packed_id > 0",
        },
        "normal_frame": "xhand_link",
        "face_normal_dot_threshold": SURFACE_NORMAL_DOT_THRESHOLD,
        "palmar_normal_axes": PALMAR_NORMAL_AXES,
        "surface_only_alignment": {
            "finger_authority": "robot_finger_labels.npy",
            "finger_zero": "packed_id = 0",
            "finger_positive": (
                "reuse rendered surface_id only when rendered finger_id matches; "
                "otherwise use lateral surface_id = 2"
            ),
            "missing_rendered_surface": "lateral surface_id = 2",
        },
    }


def split_finger_mesh_surfaces(
    mesh: trimesh.Trimesh,
    visual_transform: np.ndarray,
    side: str,
    finger: str,
) -> list[tuple[trimesh.Trimesh, np.ndarray, int]]:
    """Split one visual mesh into material-preserving anatomical submeshes."""
    face_labels = classify_finger_face_surfaces(
        mesh,
        visual_transform,
        side,
        finger,
    )
    output: list[tuple[trimesh.Trimesh, np.ndarray, int]] = []
    for surface_label in SURFACE_LABEL_IDS.values():
        face_indices = np.flatnonzero(face_labels == surface_label)
        if not len(face_indices):
            continue
        submesh = mesh.submesh(
            [face_indices],
            append=True,
            repair=False,
        )
        if not isinstance(submesh, trimesh.Trimesh):
            raise TypeError(
                f"surface split returned {type(submesh)} for {finger}"
            )
        output.append((submesh, visual_transform, surface_label))
    if sum(len(item[0].faces) for item in output) != len(mesh.faces):
        raise RuntimeError("surface split did not preserve every mesh face")
    return output


def resolve_frame_range(total: int, start: int, n: int, preview: bool) -> range:
    if total <= 0:
        raise ValueError("overlay input has no frames")
    if start < 0 or start >= total:
        raise ValueError(f"invalid --start={start} for T={total}")
    if n < 0:
        raise ValueError("--n must be non-negative")
    if not preview:
        if start != 0 or n not in (0, total):
            raise ValueError(
                "partial array renders are intentionally disabled; use "
                "--preview with --start/--n, or omit both for the full contract"
            )
        return range(total)
    count = 1 if n == 0 else n
    return range(start, min(start + count, total))


def resolve_image_size(
    data: np.lib.npyio.NpzFile,
    width: int | None,
    height: int | None,
    background: Path | None,
    render_scale: float,
) -> tuple[int, int, int, int, float]:
    """Return source W/H, render W/H, and scaled focal length."""
    if (width is None) != (height is None):
        raise ValueError("--width and --height must be provided together")
    if width is None and "img_width" in data and "img_height" in data:
        width, height = int(data["img_width"]), int(data["img_height"])
    if width is None and background is not None:
        import cv2

        capture = cv2.VideoCapture(str(background))
        if capture.isOpened():
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError(
            "image size is unavailable; pass --width and --height (case01: 1280 720)"
        )
    if not np.isfinite(render_scale) or render_scale <= 0 or render_scale > 1:
        raise ValueError("--render_scale must be in (0, 1]")
    render_width = max(1, int(round(width * render_scale)))
    render_height = max(1, int(round(height * render_scale)))
    focal = float(data["img_focal"]) * (render_width / width)
    return width, height, render_width, render_height, focal


def validate_input(data: np.lib.npyio.NpzFile, joint_meta: dict) -> tuple[int, str, list[str]]:
    required = {
        "rb5_q": (None, 6),
        "wrist_pos": (None, 3),
        "wrist_rot": (None, 3, 3),
        "qpos": (None, 12),
        "valid": (None,),
        "T_cam_base": (4, 4),
    }
    if "rb5_q" not in data:
        raise KeyError("overlay input is missing rb5_q")
    total = int(data["rb5_q"].shape[0])
    for key, expected in required.items():
        actual = None if key not in data else data[key].shape
        wanted = tuple(total if value is None else value for value in expected)
        if actual != wanted:
            raise ValueError(f"overlay input {key} shape {actual}, expected {wanted}")
    for key in ("rb5_q", "wrist_pos", "wrist_rot", "qpos", "T_cam_base"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"overlay input {key} contains non-finite values")
    if "safety_constraints_enforced" not in data or not bool(
        np.asarray(data["safety_constraints_enforced"]).item()
    ):
        raise ValueError(
            "overlay input is not execution-constrained; regenerate it with "
            "rb5_build_overlay_input.py"
        )

    raw_side = np.asarray(data["side"])
    side = str(raw_side.item() if raw_side.ndim == 0 else raw_side)
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported hand side: {side!r}")
    if joint_meta.get("side", side) != side:
        raise ValueError("joint-name sidecar does not match overlay input side")
    if joint_meta.get("embodiment") != "xhand":
        raise ValueError("joint-name sidecar must declare embodiment='xhand'")
    constraints = joint_meta.get("trajectory_constraints", {})
    if not constraints.get("enforced", False):
        raise ValueError("joint-name sidecar lacks an enforced trajectory contract")
    joint_names = list(joint_meta.get("joint_names", []))
    if len(joint_names) != 12 or len(set(joint_names)) != 12:
        raise ValueError("joint-name sidecar must contain 12 unique XHand joints")
    return total, side, joint_names


def _pyrender_geometry(
    link_meshes: dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]],
    material: pyrender.Material | None = None,
) -> dict[str, list[tuple[pyrender.Mesh, np.ndarray]]]:
    output: dict[str, list[tuple[pyrender.Mesh, np.ndarray]]] = {}
    for link_name, items in link_meshes.items():
        output[link_name] = [
            (
                pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material,
                    smooth=False,
                ),
                visual_transform,
            )
            for mesh, visual_transform in items
        ]
    return output


def _pyrender_xhand_surface_geometry(
    link_meshes: dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]],
    side: str,
) -> dict[str, list[tuple[pyrender.Mesh, np.ndarray, int]]]:
    """Convert XHand visuals, splitting finger faces by anatomical surface.

    Non-finger links remain one mesh with surface ID zero.  ``submesh`` carries
    the original visual/material data into each finger surface, so the normal
    lit RGB pass remains visually equivalent to the unsplit hand.
    """
    link_fingers = finger_link_fingers(side)
    output: dict[str, list[tuple[pyrender.Mesh, np.ndarray, int]]] = {}
    for link_name, items in link_meshes.items():
        finger = link_fingers.get(link_name)
        converted: list[tuple[pyrender.Mesh, np.ndarray, int]] = []
        for mesh, visual_transform in items:
            if finger is None:
                split_items = [(mesh, visual_transform, 0)]
            else:
                split_items = split_finger_mesh_surfaces(
                    mesh,
                    visual_transform,
                    side,
                    finger,
                )
            converted.extend(
                (
                    pyrender.Mesh.from_trimesh(submesh, smooth=False),
                    transform,
                    surface_label,
                )
                for submesh, transform, surface_label in split_items
            )
        output[link_name] = converted
    return output


def _root_pose_cv_to_gl(transform: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = T_CV2GL @ transform[:3, :3]
    pose[:3, 3] = T_CV2GL @ transform[:3, 3]
    return pose


def hand_segmentation_color(
    finger_label: int,
    packed_surface_label: int,
) -> np.ndarray:
    """Encode finger/surface IDs plus an all-XHand ownership bit.

    The first two channels retain the established finger contracts.  The
    third channel is one for every XHand visual, including the palm and hand
    base, and zero for RB5 links/background.  This keeps the palm separable
    from the arm without changing any existing finger IDs.
    """
    if not 0 <= int(finger_label) <= max(FINGER_LABEL_IDS.values()):
        raise ValueError("finger label is outside 0..5")
    if not 0 <= int(packed_surface_label) <= MAX_PACKED_SURFACE_LABEL:
        raise ValueError("packed surface label is outside 0..15")
    # Preserve the renderer's historical third channel exactly for fingers;
    # only non-finger XHand links (palm/base) need a new non-zero ownership
    # value.  Keeping finger colours bit-identical avoids introducing a new
    # raster-boundary disagreement between the finger and packed-surface
    # channels.
    ownership_or_surface = (
        ((int(packed_surface_label) - 1) % 3) + 1
        if int(finger_label) > 0
        else 1
    )
    return np.asarray(
        (
            int(finger_label),
            int(packed_surface_label),
            ownership_or_surface,
        ),
        dtype=np.uint8,
    )


class RobotScene:
    def __init__(
        self,
        width: int,
        height: int,
        focal: float,
        side: str,
    ) -> None:
        arm_color = RB5_ARM_COLOR
        light_color = RB5_LIGHT_COLOR
        self.scene = pyrender.Scene(
            ambient_light=np.asarray(light_color) * 0.32,
            bg_color=(0.0, 0.0, 0.0, 0.0),
        )
        self.scene.add(
            pyrender.IntrinsicsCamera(
                fx=focal,
                fy=focal,
                cx=width / 2.0,
                cy=height / 2.0,
                znear=0.01,
                zfar=10.0,
            ),
            pose=np.eye(4),
        )
        self.scene.add(
            pyrender.DirectionalLight(color=light_color, intensity=3.5),
            pose=np.eye(4),
        )
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = (0.4, -0.4, -0.4)
        self.scene.add(
            pyrender.PointLight(color=light_color, intensity=2.0),
            pose=fill_pose,
        )

        arm_joints, arm_geometry_raw = load_rb5_urdf()
        hand_joints, hand_geometry_raw = load_side_urdf("xhand", side)
        arm_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(*arm_color, 1.0),
            metallicFactor=0.25,
            roughnessFactor=0.55,
        )
        arm_geometry = _pyrender_geometry(arm_geometry_raw, material=arm_material)
        hand_geometry = _pyrender_xhand_surface_geometry(hand_geometry_raw, side)

        self.arm_joints = arm_joints
        self.hand_joints = hand_joints
        self.arm_nodes: dict[str, list[tuple[pyrender.Node, np.ndarray]]] = {}
        self.hand_nodes: dict[str, list[tuple[pyrender.Node, np.ndarray]]] = {}
        self.seg_node_map: dict[pyrender.Node, np.ndarray] = {}
        self.segmentation_alignment_stats = {
            "raster_mismatch_pixels": 0,
            "missing_surface_fallback_pixels": 0,
            "finger_mismatch_fallback_pixels": 0,
            "lateral_fallback_pixels": 0,
        }
        finger_labels = finger_link_labels(side)

        for link_name, items in arm_geometry.items():
            if link_name not in RB5_LINK_NAMES:
                continue
            nodes = []
            for index, (mesh, visual_transform) in enumerate(items):
                node = self.scene.add(
                    mesh,
                    pose=np.eye(4),
                    name=f"rb5:{link_name}:{index}",
                )
                nodes.append((node, visual_transform))
                self.seg_node_map[node] = np.zeros(3, dtype=np.uint8)
            self.arm_nodes[link_name] = nodes

        for link_name, items in hand_geometry.items():
            nodes = []
            label = finger_labels.get(link_name, 0)
            for index, (mesh, visual_transform, surface_label) in enumerate(items):
                packed_label = (
                    pack_finger_surface_label(label, surface_label)
                    if label > 0
                    else 0
                )
                seg_color = hand_segmentation_color(label, packed_label)
                node = self.scene.add(
                    mesh,
                    pose=OFFSCREEN_POSE,
                    name=f"xhand:{link_name}:{index}:surface={surface_label}",
                )
                nodes.append((node, visual_transform))
                self.seg_node_map[node] = seg_color
            self.hand_nodes[link_name] = nodes

        missing_finger_links = sorted(set(finger_labels) - set(self.hand_nodes))
        if missing_finger_links:
            raise RuntimeError(
                f"XHand URDF is missing finger visual links: {missing_finger_links}"
            )
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def close(self) -> None:
        self.renderer.delete()

    def set_frame(
        self,
        arm_root: np.ndarray,
        arm_q: np.ndarray,
        hand_root: np.ndarray,
        hand_q: dict[str, float],
        hand_valid: bool,
    ) -> None:
        arm_transforms = compute_fk(
            self.arm_joints,
            dict(zip(RB5_JOINT_NAMES, np.asarray(arm_q, dtype=np.float64))),
            arm_root,
        )
        for link_name, nodes in self.arm_nodes.items():
            link_pose = arm_transforms[link_name]
            for node, visual_transform in nodes:
                self.scene.set_pose(node, link_pose @ visual_transform)

        hand_transforms = (
            compute_fk(self.hand_joints, hand_q, hand_root) if hand_valid else None
        )
        for link_name, nodes in self.hand_nodes.items():
            link_pose = hand_transforms.get(link_name) if hand_transforms is not None else None
            for node, visual_transform in nodes:
                pose = OFFSCREEN_POSE if link_pose is None else link_pose @ visual_transform
                self.scene.set_pose(node, pose)

    def render_segmentation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render finger IDs, packed surfaces, and visible XHand ownership."""
        seg, _ = self.renderer.render(
            self.scene,
            flags=pyrender.RenderFlags.SEG,
            seg_node_map=self.seg_node_map,
        )
        labels = np.array(seg[..., 0], dtype=np.uint8, copy=True)
        surface_labels = np.array(seg[..., 1], dtype=np.uint8, copy=True)
        hand_mask = np.array(seg[..., 2] > 0, dtype=bool, copy=True)
        if labels.max(initial=0) > max(FINGER_LABEL_IDS.values()):
            raise RuntimeError("segmentation pass emitted an unknown finger label")
        if surface_labels.max(initial=0) > MAX_PACKED_SURFACE_LABEL:
            raise RuntimeError("segmentation pass emitted an unknown surface label")
        surface_labels, alignment = align_packed_surfaces_to_finger_labels(
            surface_labels,
            labels,
        )
        for key, value in alignment.items():
            self.segmentation_alignment_stats[key] += int(value)
        validate_packed_surface_fingers(
            surface_labels,
            labels,
            context="segmentation packed surface labels",
        )
        if np.any((labels > 0) & ~hand_mask):
            raise RuntimeError("finger segmentation escaped the XHand mask")
        return labels, surface_labels, hand_mask

    def render(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        rgba, depth = self.renderer.render(
            self.scene,
            flags=pyrender.RenderFlags.RGBA,
        )
        mask = np.isfinite(depth) & (depth > 0.0)
        labels, surface_labels, hand_mask = self.render_segmentation()
        labels[~mask] = 0
        surface_labels[~mask] = 0
        hand_mask &= mask
        metric_depth = np.full(depth.shape, np.inf, dtype=np.float32)
        metric_depth[mask] = depth[mask]
        return (
            np.asarray(rgba[..., :3], dtype=np.uint8),
            metric_depth,
            mask,
            labels,
            surface_labels,
            hand_mask,
        )


def _read_background_frames(
    path: Path,
    indices: list[int],
    size: tuple[int, int],
) -> dict[int, np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open preview background: {path}")
    output: dict[int, np.ndarray] = {}
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"background has no frame {frame_index}: {path}")
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        output[frame_index] = frame
    capture.release()
    return output


def _prepare_output_arrays(
    output_dir: Path,
    shape: tuple[int, int, int],
    overwrite: bool,
    compositor_layers_only: bool = False,
) -> dict[str, np.memmap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "robot_rgb.npy": (np.uint8, shape + (3,)),
        "robot_depth.npy": (np.float16, shape),
        "robot_mask.npy": (np.bool_, shape),
        "robot_finger_labels.npy": (np.uint8, shape),
        "robot_finger_surface_labels.npy": (np.uint8, shape),
        "robot_finger_mask.npy": (np.bool_, shape),
        "robot_hand_mask.npy": (np.bool_, shape),
    }
    if compositor_layers_only:
        specs = {
            "robot_rgb.npy": specs["robot_rgb.npy"],
            "robot_depth.npy": specs["robot_depth.npy"],
            "robot_mask.npy": specs["robot_mask.npy"],
            "robot_thumb_mask.npy": (np.bool_, shape),
            "robot_hand_mask.npy": specs["robot_hand_mask.npy"],
        }
    existing = [output_dir / filename for filename in specs if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing overlay arrays: "
            + ", ".join(str(path) for path in existing)
        )
    return {
        filename: np.lib.format.open_memmap(
            output_dir / filename,
            mode="w+",
            dtype=dtype,
            shape=array_shape,
        )
        for filename, (dtype, array_shape) in specs.items()
    }


def _prepare_surface_backfill(
    output_dir: Path,
    shape: tuple[int, int, int],
    overwrite: bool,
) -> tuple[np.memmap, Path, Path, np.ndarray]:
    """Create a sibling temporary surface array beside an existing overlay.

    Nothing inside the existing overlay is changed until the caller validates
    every frame and atomically replaces ``target_path`` with ``temp_path``.
    """
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"surface-only mode requires an existing overlay: {output_dir}"
        )
    target_path = output_dir / "robot_finger_surface_labels.npy"
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target_path}")
    finger_path = output_dir / "robot_finger_labels.npy"
    if not finger_path.is_file():
        raise FileNotFoundError(
            f"surface-only mode requires existing finger labels: {finger_path}"
        )
    finger_labels = np.load(finger_path, mmap_mode="r", allow_pickle=False)
    if finger_labels.shape != shape:
        raise ValueError(
            f"existing finger-label shape {finger_labels.shape}, expected {shape}"
        )
    if not np.issubdtype(finger_labels.dtype, np.integer):
        raise TypeError(
            f"existing finger labels must be integer, got {finger_labels.dtype}"
        )
    if finger_labels.size and (
        int(finger_labels.min()) < 0
        or int(finger_labels.max()) > max(FINGER_LABEL_IDS.values())
    ):
        raise ValueError("existing finger labels contain values outside 0..5")

    descriptor, temp_name = tempfile.mkstemp(
        prefix=".robot_finger_surface_labels.",
        suffix=".npy",
        dir=output_dir,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        surface_labels = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.uint8,
            shape=shape,
        )
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return surface_labels, temp_path, target_path, finger_labels


def _prepare_surface_manifest(
    output_dir: Path,
    backfill_stats: dict[str, int] | None = None,
) -> tuple[Path, Path]:
    """Write a complete sibling manifest temp for a surface backfill."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError(f"overlay manifest is not a JSON object: {manifest_path}")
    else:
        manifest = {}
    contract = finger_surface_manifest_contract()
    if backfill_stats is not None:
        contract["backfill_stats"] = dict(backfill_stats)
    manifest["finger_surface_labels"] = contract

    descriptor, temp_name = tempfile.mkstemp(
        prefix=".manifest.",
        suffix=".json",
        dir=output_dir,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return temp_path, manifest_path
    except Exception:
        # os.fdopen owns the descriptor after it succeeds.  If it failed before
        # taking ownership, closing an already-closed fd is harmless to ignore.
        try:
            os.close(descriptor)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _publish_surface_backfill(
    surface_temp_path: Path,
    surface_target_path: Path,
    manifest_temp_path: Path,
    manifest_target_path: Path,
) -> None:
    """Commit the surface array and manifest together, restoring both on error."""
    pairs = (
        (surface_temp_path, surface_target_path),
        (manifest_temp_path, manifest_target_path),
    )
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    committed = False

    def raise_on_sigterm(signum, _frame):
        raise SystemExit(128 + signum)

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, raise_on_sigterm)
    try:
        for _, target in pairs:
            if not target.exists():
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{target.name}.backup.",
                dir=target.parent,
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            os.replace(target, backup)
            backups[target] = backup

        for source, target in pairs:
            os.replace(source, target)
            installed.append(target)
        committed = True
    except BaseException:
        for target in reversed(installed):
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        for source, _ in pairs:
            source.unlink(missing_ok=True)
        if committed:
            for backup in backups.values():
                backup.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--jn", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--background", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None, help="source image width")
    parser.add_argument("--height", type=int, default=None, help="source image height")
    parser.add_argument(
        "--render_scale",
        type=float,
        default=0.75,
        help="render resolution relative to the source (default: 0.75)",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=0, help="preview count; 0 means one")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--surface_labels_only",
        action="store_true",
        help=(
            "backfill only robot_finger_surface_labels.npy into an existing "
            "<out>/overlay_processor; RGB/depth and all existing files are "
            "preserved"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compositor_layers_only",
        action="store_true",
        help=(
            "store only RGB, depth, robot mask, and thumb mask; this avoids "
            "allocating auxiliary label arrays for final video compositing"
        ),
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.jn.is_file():
        raise FileNotFoundError(args.jn)
    if args.preview and args.surface_labels_only:
        raise ValueError("--preview and --surface_labels_only are mutually exclusive")
    if args.preview and args.background is None:
        raise ValueError("--preview requires --background")

    data = np.load(args.data, allow_pickle=False)
    with args.jn.open() as stream:
        joint_meta = json.load(stream)
    total, side, joint_names = validate_input(data, joint_meta)
    frame_range = resolve_frame_range(total, args.start, args.n, args.preview)
    source_w, source_h, render_w, render_h, focal = resolve_image_size(
        data,
        args.width,
        args.height,
        args.background,
        args.render_scale,
    )
    print(
        f"[input] T={total} side={side} source={source_w}x{source_h} "
        f"render={render_w}x{render_h} focal={focal:.2f} "
        f"frames={frame_range.start}:{frame_range.stop}",
        flush=True,
    )

    robot_scene = RobotScene(
        render_w,
        render_h,
        focal,
        side,
    )
    arm_root = _root_pose_cv_to_gl(np.asarray(data["T_cam_base"], dtype=np.float64))

    arrays = None
    staging_output: Path | None = None
    final_output: Path | None = None
    surface_array: np.memmap | None = None
    surface_temp_path: Path | None = None
    surface_target_path: Path | None = None
    existing_finger_labels: np.ndarray | None = None
    backfill_stats = {
        "frame_count": 0,
        "raster_mismatch_pixels": 0,
        "missing_surface_fallback_pixels": 0,
        "finger_mismatch_fallback_pixels": 0,
        "lateral_fallback_pixels": 0,
    }
    background_frames = None
    preview_frames: list[np.ndarray] = []
    if args.preview:
        background_frames = _read_background_frames(
            args.background,
            list(frame_range),
            (render_w, render_h),
        )
    elif args.surface_labels_only:
        final_output = args.out / "overlay_processor"
        if final_output.is_symlink() or not final_output.is_dir():
            raise ValueError(f"invalid existing overlay output path: {final_output}")
        (
            surface_array,
            surface_temp_path,
            surface_target_path,
            existing_finger_labels,
        ) = _prepare_surface_backfill(
            final_output,
            (total, render_h, render_w),
            args.overwrite,
        )
        atexit.register(surface_temp_path.unlink, missing_ok=True)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        final_output = args.out / "overlay_processor"
        if final_output.is_symlink() or (
            final_output.exists() and not final_output.is_dir()
        ):
            raise ValueError(f"invalid overlay output path: {final_output}")
        if final_output.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to replace existing overlay output: {final_output}"
            )
        staging_output = Path(
            tempfile.mkdtemp(prefix=".overlay_processor.", dir=args.out)
        )
        atexit.register(shutil.rmtree, staging_output, ignore_errors=True)
        arrays = _prepare_output_arrays(
            staging_output,
            (total, render_h, render_w),
            False,
            args.compositor_layers_only,
        )

    try:
        for output_index, frame_index in enumerate(frame_range):
            hand_root = np.eye(4, dtype=np.float64)
            hand_root[:3, :3] = T_CV2GL @ data["wrist_rot"][frame_index]
            hand_root[:3, 3] = T_CV2GL @ data["wrist_pos"][frame_index]
            qpos = {
                name: float(data["qpos"][frame_index, joint_index])
                for joint_index, name in enumerate(joint_names)
            }
            robot_scene.set_frame(
                arm_root,
                data["rb5_q"][frame_index],
                hand_root,
                qpos,
                bool(data["valid"][frame_index]),
            )
            if args.surface_labels_only:
                (
                    labels,
                    rendered_surface_labels,
                    _rendered_hand_mask,
                ) = robot_scene.render_segmentation()
                assert existing_finger_labels is not None
                reference_labels = existing_finger_labels[frame_index]
                surface_labels, frame_stats = align_packed_surfaces_to_finger_labels(
                    rendered_surface_labels,
                    reference_labels,
                )
                # This is deliberately strict after alignment: every packed
                # output pixel must retain the established finger contract.
                validate_packed_surface_fingers(
                    surface_labels,
                    reference_labels,
                    context=f"surface-only frame {frame_index}",
                )
                labels = np.asarray(reference_labels)
                backfill_stats["frame_count"] += 1
                for key, value in frame_stats.items():
                    backfill_stats[key] += value
                assert surface_array is not None
                surface_array[frame_index] = surface_labels
                mask = None
            else:
                (
                    rgb,
                    depth,
                    mask,
                    labels,
                    surface_labels,
                    hand_mask,
                ) = robot_scene.render()

            if args.preview:
                background = background_frames[frame_index]
                composite = background.copy()
                assert mask is not None
                composite[mask] = rgb[..., ::-1][mask]
                import cv2

                cv2.putText(
                    composite,
                    f"frame {frame_index}",
                    (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (20, 20, 240),
                    2,
                    cv2.LINE_AA,
                )
                preview_frames.append(composite)
            elif not args.surface_labels_only:
                assert arrays is not None
                arrays["robot_rgb.npy"][output_index] = rgb
                arrays["robot_depth.npy"][output_index] = depth.astype(np.float16)
                arrays["robot_mask.npy"][output_index] = mask
                if "robot_thumb_mask.npy" in arrays:
                    arrays["robot_thumb_mask.npy"][output_index] = (
                        labels == FINGER_LABEL_IDS["thumb"]
                    )
                    arrays["robot_hand_mask.npy"][output_index] = hand_mask
                else:
                    arrays["robot_finger_labels.npy"][output_index] = labels
                    arrays["robot_finger_surface_labels.npy"][output_index] = surface_labels
                    arrays["robot_finger_mask.npy"][output_index] = labels > 0
                    arrays["robot_hand_mask.npy"][output_index] = hand_mask

            if output_index % 10 == 0 or output_index + 1 == len(frame_range):
                robot_text = (
                    "seg-only"
                    if mask is None
                    else f"robot={int(mask.sum())}px"
                )
                print(
                    f"  frame {frame_index} {robot_text} "
                    f"finger={int((labels > 0).sum())}px"
                    + (
                        " "
                        f"remapped={backfill_stats['raster_mismatch_pixels']}px "
                        f"fallback={backfill_stats['missing_surface_fallback_pixels']}px"
                        if args.surface_labels_only
                        else ""
                    ),
                    flush=True,
                )
                if arrays is not None:
                    for array in arrays.values():
                        array.flush()
                if surface_array is not None:
                    surface_array.flush()
    finally:
        robot_scene.close()

    if args.preview:
        import cv2

        thumbs = preview_frames[:6]
        if not thumbs:
            raise RuntimeError("preview produced no frames")
        if len(thumbs) == 1:
            montage = thumbs[0]
        else:
            thumb_w = min(480, render_w)
            thumb_h = max(1, int(round(render_h * thumb_w / render_w)))
            thumbs = [
                cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                for frame in thumbs
            ]
            montage = np.concatenate(thumbs, axis=1)
        args.out.mkdir(parents=True, exist_ok=True)
        preview_path = args.out / "rb5_preview.png"
        if preview_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {preview_path}")
        if not cv2.imwrite(str(preview_path), montage):
            raise RuntimeError(f"failed to write {preview_path}")
        print(f"[ok] preview: {preview_path}", flush=True)
        return

    if args.surface_labels_only:
        assert surface_array is not None
        assert surface_temp_path is not None
        assert surface_target_path is not None
        assert final_output is not None
        surface_array.flush()
        del surface_array
        existing_finger_labels = None
        with surface_temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest_temp_path, manifest_target_path = _prepare_surface_manifest(
            final_output,
            backfill_stats,
        )
        _publish_surface_backfill(
            surface_temp_path,
            surface_target_path,
            manifest_temp_path,
            manifest_target_path,
        )
        print(
            f"[ok] packed finger surface labels: {surface_target_path} "
            f"remapped={backfill_stats['raster_mismatch_pixels']}px "
            f"fallback={backfill_stats['lateral_fallback_pixels']}px",
            flush=True,
        )
        return

    assert arrays is not None
    for array in arrays.values():
        array.flush()
    del arrays
    manifest = {
        "renderer": "pyrender-egl",
        "arm": "rb5_850e_collision_stl",
        "arm_mode": "full_locked",
        "hand": "xhand_visual_urdf",
        "side": side,
        "frame_count": total,
        "source_size": [source_w, source_h],
        "render_size": [render_w, render_h],
        "img_focal": focal,
        "finger_mask": {"label_ids": FINGER_LABEL_IDS},
        "hand_mask": {
            "filename": "robot_hand_mask.npy",
            "definition": "visible XHand visuals including palm/base and fingers",
            "excludes_arm": True,
        },
        "finger_surface_labels": finger_surface_manifest_contract(),
        "depth": {"unit": "metre", "background": "positive_infinity"},
        "segmentation_alignment": dict(robot_scene.segmentation_alignment_stats),
        "trajectory_constraints": joint_meta["trajectory_constraints"],
    }
    assert staging_output is not None
    assert final_output is not None
    manifest_path = staging_output / "manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
    publish_directory(str(staging_output), str(final_output))
    print(f"[ok] overlay arrays: {final_output}", flush=True)


if __name__ == "__main__":
    main()
