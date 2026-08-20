"""Build a video-general human hand/forearm mask from HaWoR MANO geometry.

The hand is the projected MANO triangle surface. The forearm is a capsule from
the wrist in the direction opposite the palm centre until it exits the image.
No recording ID, fixed pixel coordinate, or per-video threshold is used.
When cached Grounding-DINO detections and a clean plate are supplied, visible
non-skin object pixels are protected from the human-removal mask.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[2]


def project(xyz: np.ndarray, focal: float, width: int, height: int) -> np.ndarray:
    z = np.clip(xyz[:, 2], 1e-6, None)
    return np.column_stack(
        (focal * xyz[:, 0] / z + width / 2, focal * xyz[:, 1] / z + height / 2)
    )


def ray_to_border(origin: np.ndarray, direction: np.ndarray,
                  width: int, height: int) -> np.ndarray:
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    times = []
    for axis, bound in ((0, 0), (0, width - 1), (1, 0), (1, height - 1)):
        if abs(direction[axis]) < 1e-8:
            continue
        time = (bound - origin[axis]) / direction[axis]
        other = origin[1 - axis] + time * direction[1 - axis]
        other_limit = height - 1 if axis == 0 else width - 1
        if time > 0 and 0 <= other <= other_limit:
            times.append(time)
    return origin + min(times, default=float(max(width, height))) * direction


def grounded_boxes(detections: list[dict], width: int, height: int,
                   min_score: float, padding: int) -> np.ndarray:
    """Return a union of bounded Grounding-DINO object boxes."""
    result = np.zeros((height, width), dtype=bool)
    for detection in detections:
        if float(detection.get("grounding_score", 0.0)) < min_score:
            continue
        box = np.asarray(detection.get("box_xyxy", ()), dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0 or area > 0.30 * width * height:
            continue
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = max(0, x1 - padding), min(width, x2 + padding)
        y1, y2 = max(0, y1 - padding), min(height, y2 + padding)
        result[y1:y2, x1:x2] = True
    return result


def add_connected_arm_skin(
    mask: np.ndarray,
    skin: np.ndarray,
    seed: np.ndarray,
    forearm_corridor: np.ndarray,
    mano_area: int,
    max_hand_ratio: float,
    max_frame_ratio: float,
) -> None:
    """Grow a MANO seed through a bounded, border-connected skin component.

    A visible forearm normally forms one skin component from the MANO hand to
    an image border, including when the elbow bends away from the wrist ray.
    Scene-wide false skin components are rejected using scale-normalized area
    limits.  For those ambiguous frames, growth falls back to the geometric
    corridor instead of deleting a large part of the scene.
    """
    skin_bool = skin.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
    max_area = min(
        max_hand_ratio * max(mano_area, 1),
        max_frame_ratio * mask.size,
    )
    accepted = False
    min_overlap = max(32, int(round(0.002 * max(mano_area, 1))))
    for label in range(1, count):
        component = labels == label
        overlap = int(np.count_nonzero(component & seed))
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = bool(
            component[0].any() or component[-1].any()
            or component[:, 0].any() or component[:, -1].any()
        )
        if (
            touches_border
            and overlap >= min_overlap
            and overlap >= 0.02 * area
            and area <= max_area
        ):
            mask[component] = 1
            accepted = True

    if accepted:
        return

    # Ambiguous component (typically skin-coloured floor/table merged with the
    # arm): retain the conservative geometry-bounded completion.
    local_skin = (skin_bool & forearm_corridor.astype(bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local_skin, 8)
    for label in range(1, count):
        component = labels == label
        overlap = int(np.count_nonzero(component & seed))
        area = int(stats[label, cv2.CC_STAT_AREA])
        if overlap >= min_overlap and overlap >= 0.02 * area:
            mask[component] = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawor_npz", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None,
                        help="optional source video for geometry-seeded skin completion")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object_context", type=Path,
                        help="cached Grounding-DINO/SAM object context")
    parser.add_argument("--clean_plate", type=Path,
                        help="clean static scene image for visible-object protection")
    parser.add_argument(
        "--protected_object_output", type=Path,
        help="optional output mask for measured DINO-box object pixels",
    )
    parser.add_argument("--hand_dilate_px", type=int, default=8)
    parser.add_argument("--forearm_wrist_width_scale", type=float, default=1.65)
    parser.add_argument("--forearm_border_width_scale", type=float, default=3.40)
    parser.add_argument("--object_difference_threshold", type=float, default=18.0)
    parser.add_argument("--grounding_min_score", type=float, default=0.30)
    parser.add_argument("--grounding_box_padding_px", type=int, default=12)
    parser.add_argument("--skin_component_max_hand_ratio", type=float, default=6.0)
    parser.add_argument("--skin_component_max_frame_ratio", type=float, default=0.12)
    args = parser.parse_args()

    with np.load(args.hawor_npz) as data:
        vertices = data[f"verts_{args.side}"].astype(np.float32)
        joints = data[f"joints_{args.side}"].astype(np.float32)
        valid = data["valid"][0 if args.side == "left" else 1].astype(bool)
        focal = float(data["img_focal"])
    faces = np.load(
        REPO / "src/retargeting/assets" / f"mano_faces_{args.side}.npy"
    ).astype(np.int32)
    masks = np.zeros((len(vertices), args.height, args.width), dtype=bool)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.hand_dilate_px + 1,) * 2
    )
    plate_lab = None
    grounding_frames = np.zeros(0, dtype=np.int64)
    grounding_detections = []
    if (args.object_context is None) != (args.clean_plate is None):
        parser.error("--object_context and --clean_plate must be supplied together")
    if args.protected_object_output is not None and args.object_context is None:
        parser.error("--protected_object_output requires Grounding-DINO inputs")
    protected_objects = (
        np.zeros_like(masks) if args.protected_object_output is not None else None
    )
    if args.object_context is not None:
        import torch
        context = torch.load(
            args.object_context, map_location="cpu", weights_only=False
        )
        grounding_frames = np.asarray(
            context["token_center_frame_indices"], dtype=np.int64
        )
        grounding_detections = [
            item.get("detections", []) for item in context["token_detections"]
        ]
        if len(grounding_frames) != len(grounding_detections):
            raise ValueError("Grounding-DINO context frame/detection mismatch")
        plate = cv2.imread(str(args.clean_plate), cv2.IMREAD_COLOR)
        if plate is None:
            raise FileNotFoundError(args.clean_plate)
        if plate.shape[:2] != (args.height, args.width):
            plate = cv2.resize(plate, (args.width, args.height))
        plate_lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)

    previous = None
    capture = cv2.VideoCapture(str(args.video)) if args.video is not None else None
    if capture is not None and not capture.isOpened():
        raise FileNotFoundError(args.video)
    for frame in range(len(vertices)):
        bgr = None
        if capture is not None:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {frame}")
        if not valid[frame]:
            if previous is not None:
                masks[frame] = previous
                if protected_objects is not None and frame > 0:
                    protected_objects[frame] = protected_objects[frame - 1]
            continue
        uv_vertices = project(vertices[frame], focal, args.width, args.height)
        uv_joints = project(joints[frame], focal, args.width, args.height)
        mask = np.zeros((args.height, args.width), dtype=np.uint8)
        points = np.rint(uv_vertices).astype(np.int32)
        for face in faces:
            cv2.fillConvexPoly(mask, points[face], 1, lineType=cv2.LINE_8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mano_surface = mask.astype(bool)

        wrist = uv_joints[0]
        palm = uv_joints[[5, 9, 13, 17]].mean(axis=0)
        hand_length = max(float(np.linalg.norm(palm - wrist)) * 2.0, 30.0)
        endpoint = ray_to_border(
            wrist, wrist - palm, args.width, args.height
        )
        axis = endpoint - wrist
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        normal = np.array([-axis[1], axis[0]], dtype=np.float32)
        wrist_half = max(18.0, 0.5 * args.forearm_wrist_width_scale * hand_length)
        border_half = max(
            wrist_half,
            0.5 * args.forearm_border_width_scale * hand_length,
        )
        forearm = np.rint(np.stack([
            wrist + wrist_half * normal,
            wrist - wrist_half * normal,
            endpoint - border_half * normal,
            endpoint + border_half * normal,
        ])).astype(np.int32)
        forearm_corridor = np.zeros_like(mask)
        cv2.fillConvexPoly(
            forearm_corridor, forearm, 1, lineType=cv2.LINE_8
        )
        if bgr is None:
            cv2.fillConvexPoly(mask, forearm, 1, lineType=cv2.LINE_AA)

        # Complete the geometric seed to the actual visible arm boundary. The
        # fixed YCrCb skin range is used only for components that substantially
        # overlap the MANO/forearm seed, so similarly coloured scene objects do
        # not become human masks on their own.
        if bgr is not None:
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            skin = cv2.inRange(
                ycrcb,
                np.array([20, 128, 70], dtype=np.uint8),
                np.array([255, 182, 142], dtype=np.uint8),
            )
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            skin &= cv2.inRange(
                hsv,
                np.array([0, 18, 35], dtype=np.uint8),
                np.array([30, 190, 255], dtype=np.uint8),
            )
            skin = cv2.morphologyEx(
                skin, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            )
            seed = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            ).astype(bool)
            add_connected_arm_skin(
                mask, skin, seed, forearm_corridor,
                int(np.count_nonzero(mano_surface)),
                args.skin_component_max_hand_ratio,
                args.skin_component_max_frame_ratio,
            )
            mask = cv2.dilate(
                mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                iterations=1,
            )
            if plate_lab is not None:
                token = int(np.argmin(np.abs(grounding_frames - frame)))
                boxes = grounded_boxes(
                    grounding_detections[token], args.width, args.height,
                    args.grounding_min_score, args.grounding_box_padding_px,
                )
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                foreground = (
                    np.linalg.norm(lab - plate_lab, axis=2)
                    >= args.object_difference_threshold
                )
                strict_hand = cv2.dilate(
                    mano_surface.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                ).astype(bool)
                # Keep measured object RGB outside the geometrically certain
                # hand surface. Skin-like pixels remain removable so that an
                # arm crossing a DINO box is not accidentally preserved.
                protected_object = (
                    boxes & foreground & ~(skin.astype(bool)) & ~strict_hand
                )
                mask[protected_object] = 0
                if protected_objects is not None:
                    protected_objects[frame] = protected_object
        masks[frame] = mask.astype(bool)
        previous = masks[frame]

    if capture is not None:
        capture.release()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, masks)
    if protected_objects is not None:
        args.protected_object_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.protected_object_output, protected_objects)
    print(
        f"[ok] {args.output} frames={len(masks)} "
        f"area median={np.median(masks.sum(axis=(1, 2))):.0f}px"
    )


if __name__ == "__main__":
    main()
