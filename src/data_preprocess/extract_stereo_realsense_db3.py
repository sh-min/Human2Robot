"""Extract and timestamp-synchronize two RealSense ROS2 DB3 recordings.

The RealSense recorder stores RGB, depth, and per-frame metadata as CDR
messages in a rosbag2 SQLite database.  This utility keeps the source bags
untouched and publishes a common, timestamp-aligned stereo episode:

    <output>/camera_1/rgb/rgb_frame000000.png
    <output>/camera_1/depth_raw/depth_frame000000.png
    <output>/camera_2/...
    <output>/rgb -> camera_<annotation-camera>/rgb
    <output>/depth_raw -> camera_<annotation-camera>/depth_raw
    <output>/stereo_pairs.csv
    <output>/conversion_manifest.json

The root-level ``rgb`` alias makes the episode directly discoverable by the
skill annotation GUI.  All timestamps written to text/CSV files are seconds
in the recorder's global/system clock domain.  No image rotation, resizing,
depth registration, or depth-value conversion is performed.

Run this script in an environment containing ``rosbags``, NumPy, and OpenCV.
For example:

    conda run -n hawor python src/data_preprocess/extract_stereo_realsense_db3.py \
        --camera-1-db Depth_Stereo/case01/20260802_203521.db3 \
        --camera-2-db Depth_Stereo/case01/20260802_203522.db3 \
        --output Depth_Stereo/converted/case01
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


COLOR_DATA_TOPIC = "/device_0/sensor_1/Color_0/image/data"
COLOR_METADATA_TOPIC = "/device_0/sensor_1/Color_0/image/metadata"
COLOR_INFO_TOPIC = "/device_0/sensor_1/Color_0/camera_info"
COLOR_TF_TOPIC = "/device_0/sensor_1/Color_0/tf/ref_0"
DEPTH_DATA_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
DEPTH_METADATA_TOPIC = "/device_0/sensor_0/Depth_0/image/metadata"
DEPTH_INFO_TOPIC = "/device_0/sensor_0/Depth_0/camera_info"
DEPTH_TF_TOPIC = "/device_0/sensor_0/Depth_0/tf/ref_0"
DEPTH_UNITS_TOPIC = "/device_0/sensor_0/option/Depth_Units/value"
DEVICE_INFO_TOPIC = "/device_0/info"


@dataclass(frozen=True)
class FrameRecord:
    source_index: int
    bag_timestamp_ns: int
    timestamp_s: float
    frame_number: int
    metadata: dict[str, str]


@dataclass(frozen=True)
class CameraRecording:
    path: Path
    device: dict[str, str]
    color_info: dict[str, Any]
    depth_info: dict[str, Any]
    color_tf_raw: str
    depth_tf_raw: str
    depth_units_m: float
    rgb: tuple[FrameRecord, ...]
    depth: tuple[FrameRecord, ...]


def _parse_semicolon_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in text.rstrip("\x00;").split(";"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _camera_info(text: str) -> dict[str, Any]:
    raw = _parse_semicolon_fields(text)
    result: dict[str, Any] = dict(raw)
    for key in ("width", "height"):
        if key in result:
            result[key] = int(result[key])
    for key in ("fx", "fy", "ppx", "ppy"):
        if key in result:
            result[key] = float(result[key])
    if "coeffs" in result:
        result["coeffs"] = [float(value) for value in result["coeffs"].split(",")]
    return result


def _timestamp_seconds(value: str) -> float:
    """Normalize RealSense metadata timestamps to seconds.

    Current librealsense bags write ``timestamp`` in milliseconds.  The
    magnitude checks retain compatibility with seconds or nanoseconds.
    """

    timestamp = float(value)
    magnitude = abs(timestamp)
    if magnitude >= 1.0e15:
        return timestamp / 1.0e9
    if magnitude >= 1.0e11:
        return timestamp / 1.0e3
    return timestamp


def _find_connection(reader: AnyReader, topic: str):
    matches = [connection for connection in reader.connections if connection.topic == topic]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one connection for {topic!r}, found {len(matches)}"
        )
    return matches[0]


def _read_string(reader: AnyReader, topic: str) -> str:
    connection = _find_connection(reader, topic)
    try:
        _, _, raw = next(reader.messages(connections=[connection]))
    except StopIteration as exc:
        raise ValueError(f"topic has no messages: {topic}") from exc
    return str(reader.deserialize(raw, connection.msgtype).data)


def _read_frame_index(reader: AnyReader, topic: str) -> tuple[FrameRecord, ...]:
    connection = _find_connection(reader, topic)
    records: list[FrameRecord] = []
    for source_index, (_, bag_timestamp_ns, raw) in enumerate(
        reader.messages(connections=[connection])
    ):
        text = str(reader.deserialize(raw, connection.msgtype).data)
        metadata = _parse_semicolon_fields(text)
        if "timestamp" not in metadata or "Frame number" not in metadata:
            raise ValueError(
                f"missing timestamp/frame number in {topic} message {source_index}"
            )
        records.append(
            FrameRecord(
                source_index=source_index,
                bag_timestamp_ns=int(bag_timestamp_ns),
                timestamp_s=_timestamp_seconds(metadata["timestamp"]),
                frame_number=int(metadata["Frame number"]),
                metadata=metadata,
            )
        )
    if not records:
        raise ValueError(f"topic has no frame metadata: {topic}")
    timestamps = [record.timestamp_s for record in records]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"timestamps are not strictly increasing: {topic}")
    return tuple(records)


def inspect_recording(path: Path) -> CameraRecording:
    if not path.is_file():
        raise FileNotFoundError(path)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([path], default_typestore=typestore) as reader:
        device = _parse_semicolon_fields(_read_string(reader, DEVICE_INFO_TOPIC))
        color_info = _camera_info(_read_string(reader, COLOR_INFO_TOPIC))
        depth_info = _camera_info(_read_string(reader, DEPTH_INFO_TOPIC))
        color_tf_raw = _read_string(reader, COLOR_TF_TOPIC)
        depth_tf_raw = _read_string(reader, DEPTH_TF_TOPIC)
        depth_units_m = float(_read_string(reader, DEPTH_UNITS_TOPIC))
        rgb = _read_frame_index(reader, COLOR_METADATA_TOPIC)
        depth = _read_frame_index(reader, DEPTH_METADATA_TOPIC)
        color_count = _find_connection(reader, COLOR_DATA_TOPIC).msgcount
        depth_count = _find_connection(reader, DEPTH_DATA_TOPIC).msgcount
    if color_count != len(rgb):
        raise ValueError(f"RGB data/metadata count mismatch: {color_count} != {len(rgb)}")
    if depth_count != len(depth):
        raise ValueError(
            f"depth data/metadata count mismatch: {depth_count} != {len(depth)}"
        )
    return CameraRecording(
        path=path.resolve(),
        device=device,
        color_info=color_info,
        depth_info=depth_info,
        color_tf_raw=color_tf_raw,
        depth_tf_raw=depth_tf_raw,
        depth_units_m=depth_units_m,
        rgb=rgb,
        depth=depth,
    )


def _nearest_index(records: tuple[FrameRecord, ...], timestamp_s: float) -> int:
    timestamps = [record.timestamp_s for record in records]
    right = bisect.bisect_left(timestamps, timestamp_s)
    candidates = [index for index in (right - 1, right) if 0 <= index < len(records)]
    if not candidates:
        raise ValueError("cannot match against an empty frame sequence")
    return min(candidates, key=lambda index: abs(records[index].timestamp_s - timestamp_s))


def pair_stereo_rgb(
    camera_1: CameraRecording,
    camera_2: CameraRecording,
    reference_camera: int,
    max_delta_ms: float,
) -> list[tuple[int, int]]:
    """Return unique, monotonic ``(camera_1_index, camera_2_index)`` pairs."""

    if reference_camera == 1:
        reference = camera_1.rgb
        other = camera_2.rgb
    elif reference_camera == 2:
        reference = camera_2.rgb
        other = camera_1.rgb
    else:
        raise ValueError(f"reference_camera must be 1 or 2, got {reference_camera}")

    pairs: list[tuple[int, int]] = []
    last_other = -1
    threshold_s = max_delta_ms / 1000.0
    for reference_index, record in enumerate(reference):
        other_index = _nearest_index(other, record.timestamp_s)
        delta_s = abs(other[other_index].timestamp_s - record.timestamp_s)
        if delta_s > threshold_s or other_index <= last_other:
            continue
        last_other = other_index
        if reference_camera == 1:
            pairs.append((reference_index, other_index))
        else:
            pairs.append((other_index, reference_index))
    if not pairs:
        raise ValueError(
            f"no stereo RGB pairs found within {max_delta_ms:.3f} ms"
        )
    return pairs


def pair_rgb_to_depth(
    rgb: tuple[FrameRecord, ...],
    depth: tuple[FrameRecord, ...],
    rgb_indices: Iterable[int],
    max_delta_ms: float,
) -> list[int]:
    threshold_s = max_delta_ms / 1000.0
    matched: list[int] = []
    last_depth = -1
    for rgb_index in rgb_indices:
        depth_index = _nearest_index(depth, rgb[rgb_index].timestamp_s)
        delta_s = abs(depth[depth_index].timestamp_s - rgb[rgb_index].timestamp_s)
        if delta_s > threshold_s:
            raise ValueError(
                f"RGB frame {rgb_index} has no depth within {max_delta_ms:.3f} ms "
                f"(nearest {delta_s * 1000.0:.3f} ms)"
            )
        if depth_index <= last_depth:
            raise ValueError(
                f"non-unique/non-monotonic depth match at RGB frame {rgb_index}"
            )
        matched.append(depth_index)
        last_depth = depth_index
    return matched


def filter_stereo_pairs_by_depth(
    camera_1: CameraRecording,
    camera_2: CameraRecording,
    pairs: list[tuple[int, int]],
    max_delta_ms: float,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[dict[str, Any]]]:
    """Keep only stereo RGB pairs with valid, unique depth in both cameras.

    A recording can contain an isolated dropped depth frame while its RGB
    stream continues normally.  Pairing that RGB frame to the adjacent depth
    image merely by relaxing the threshold introduces an almost one-frame
    RGB/depth error.  Instead, omit the affected common frame and retain a
    continuous output index for every remaining four-image tuple.
    """

    threshold_s = max_delta_ms / 1000.0
    filtered_pairs: list[tuple[int, int]] = []
    camera_1_depth: list[int] = []
    camera_2_depth: list[int] = []
    rejected: list[dict[str, Any]] = []
    last_depth_indices = [-1, -1]

    for camera_1_rgb_index, camera_2_rgb_index in pairs:
        rgb_records = (
            camera_1.rgb[camera_1_rgb_index],
            camera_2.rgb[camera_2_rgb_index],
        )
        recordings = (camera_1, camera_2)
        depth_indices = [
            _nearest_index(recording.depth, rgb.timestamp_s)
            for recording, rgb in zip(recordings, rgb_records)
        ]
        deltas_ms = [
            (
                recording.depth[depth_index].timestamp_s - rgb.timestamp_s
            )
            * 1000.0
            for recording, rgb, depth_index in zip(
                recordings, rgb_records, depth_indices
            )
        ]

        reasons: list[str] = []
        for camera_offset, (depth_index, delta_ms) in enumerate(
            zip(depth_indices, deltas_ms), start=1
        ):
            if abs(delta_ms) > threshold_s * 1000.0:
                reasons.append(f"camera_{camera_offset}_depth_delta")
            if depth_index <= last_depth_indices[camera_offset - 1]:
                reasons.append(f"camera_{camera_offset}_depth_non_unique")

        if reasons:
            rejected.append(
                {
                    "camera_1_rgb_source_index": camera_1_rgb_index,
                    "camera_2_rgb_source_index": camera_2_rgb_index,
                    "camera_1_depth_source_index": depth_indices[0],
                    "camera_2_depth_source_index": depth_indices[1],
                    "camera_1_depth_minus_rgb_ms": deltas_ms[0],
                    "camera_2_depth_minus_rgb_ms": deltas_ms[1],
                    "reasons": reasons,
                }
            )
            continue

        filtered_pairs.append((camera_1_rgb_index, camera_2_rgb_index))
        camera_1_depth.append(depth_indices[0])
        camera_2_depth.append(depth_indices[1])
        last_depth_indices[:] = depth_indices

    if not filtered_pairs:
        raise ValueError(
            f"no complete stereo RGB/depth tuples found within {max_delta_ms:.3f} ms"
        )
    return filtered_pairs, camera_1_depth, camera_2_depth, rejected


def _decode_rgb(message) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    raw = np.asarray(message.data, dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError(
            f"invalid RGB payload: {raw.size} bytes, expected {height * step}"
        )
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported RGB encoding: {message.encoding}")
    if step < width * 3:
        raise ValueError(f"RGB step {step} is smaller than width*3 ({width * 3})")
    packed = raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
    if encoding == "rgb8":
        return cv2.cvtColor(packed, cv2.COLOR_RGB2BGR)
    return packed.copy()


def _decode_depth(message) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if encoding not in {"mono16", "16uc1", "z16"}:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")
    raw = np.asarray(message.data, dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError(
            f"invalid depth payload: {raw.size} bytes, expected {height * step}"
        )
    if step < width * 2:
        raise ValueError(f"depth step {step} is smaller than width*2 ({width * 2})")
    packed = raw.reshape(height, step)[:, : width * 2].copy()
    byte_order = ">u2" if bool(message.is_bigendian) else "<u2"
    return packed.view(byte_order).reshape(height, width).astype(np.uint16, copy=False)


def _extract_images(
    recording: CameraRecording,
    topic: str,
    selected_indices: list[int],
    destination: Path,
    kind: str,
    png_compression: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    selected = {source_index: output_index for output_index, source_index in enumerate(selected_indices)}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    written = 0
    with AnyReader([recording.path], default_typestore=typestore) as reader:
        connection = _find_connection(reader, topic)
        for source_index, (_, _, raw) in enumerate(reader.messages(connections=[connection])):
            output_index = selected.get(source_index)
            if output_index is None:
                continue
            message = reader.deserialize(raw, connection.msgtype)
            if kind == "rgb":
                image = _decode_rgb(message)
                filename = f"rgb_frame{output_index:06d}.png"
            elif kind == "depth":
                image = _decode_depth(message)
                filename = f"depth_frame{output_index:06d}.png"
            else:
                raise ValueError(kind)
            output_path = destination / filename
            success = cv2.imwrite(
                str(output_path),
                image,
                [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)],
            )
            if not success:
                raise RuntimeError(f"failed to write {output_path}")
            written += 1
            if written % 100 == 0 or written == len(selected_indices):
                print(f"[{destination.parent.name}/{kind}] {written}/{len(selected_indices)}")
    if written != len(selected_indices):
        raise RuntimeError(
            f"wrote {written} {kind} frames, expected {len(selected_indices)}"
        )


def _write_timestamps(path: Path, records: tuple[FrameRecord, ...], indices: list[int]) -> None:
    path.write_text(
        "".join(f"{records[index].timestamp_s:.9f}\n" for index in indices),
        encoding="utf-8",
    )


def _write_camera_csv(
    path: Path,
    recording: CameraRecording,
    rgb_indices: list[int],
    depth_indices: list[int],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "output_index",
                "rgb_source_index",
                "rgb_frame_number",
                "rgb_timestamp_s",
                "depth_source_index",
                "depth_frame_number",
                "depth_timestamp_s",
                "depth_minus_rgb_ms",
            ]
        )
        for output_index, (rgb_index, depth_index) in enumerate(
            zip(rgb_indices, depth_indices)
        ):
            rgb = recording.rgb[rgb_index]
            depth = recording.depth[depth_index]
            writer.writerow(
                [
                    output_index,
                    rgb_index,
                    rgb.frame_number,
                    f"{rgb.timestamp_s:.9f}",
                    depth_index,
                    depth.frame_number,
                    f"{depth.timestamp_s:.9f}",
                    f"{(depth.timestamp_s - rgb.timestamp_s) * 1000.0:.6f}",
                ]
            )


def _write_stereo_csv(
    path: Path,
    camera_1: CameraRecording,
    camera_2: CameraRecording,
    pairs: list[tuple[int, int]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "output_index",
                "camera_1_source_index",
                "camera_1_frame_number",
                "camera_1_timestamp_s",
                "camera_2_source_index",
                "camera_2_frame_number",
                "camera_2_timestamp_s",
                "camera_2_minus_camera_1_ms",
            ]
        )
        for output_index, (camera_1_index, camera_2_index) in enumerate(pairs):
            first = camera_1.rgb[camera_1_index]
            second = camera_2.rgb[camera_2_index]
            writer.writerow(
                [
                    output_index,
                    camera_1_index,
                    first.frame_number,
                    f"{first.timestamp_s:.9f}",
                    camera_2_index,
                    second.frame_number,
                    f"{second.timestamp_s:.9f}",
                    f"{(second.timestamp_s - first.timestamp_s) * 1000.0:.6f}",
                ]
            )


def _fps(records: tuple[FrameRecord, ...], indices: list[int]) -> float:
    timestamps = [records[index].timestamp_s for index in indices]
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    return 1.0 / statistics.median(intervals) if intervals else 0.0


def _camera_manifest(
    camera_id: int,
    recording: CameraRecording,
    rgb_indices: list[int],
    depth_indices: list[int],
) -> dict[str, Any]:
    offsets_ms = [
        (recording.depth[depth_index].timestamp_s - recording.rgb[rgb_index].timestamp_s)
        * 1000.0
        for rgb_index, depth_index in zip(rgb_indices, depth_indices)
    ]
    return {
        "camera_id": camera_id,
        "source_db3": str(recording.path),
        "device": recording.device,
        "color_info": recording.color_info,
        "depth_info": recording.depth_info,
        "depth_units_m": recording.depth_units_m,
        "color_tf_ref_raw": recording.color_tf_raw,
        "depth_tf_ref_raw": recording.depth_tf_raw,
        "source_rgb_frames": len(recording.rgb),
        "source_depth_frames": len(recording.depth),
        "output_frames": len(rgb_indices),
        "estimated_fps": _fps(recording.rgb, rgb_indices),
        "rgb_depth_offset_ms": {
            "min": min(offsets_ms),
            "median": statistics.median(offsets_ms),
            "max": max(offsets_ms),
        },
    }


def convert(args: argparse.Namespace) -> Path:
    camera_1 = inspect_recording(args.camera_1_db.resolve())
    camera_2 = inspect_recording(args.camera_2_db.resolve())
    candidate_pairs = pair_stereo_rgb(
        camera_1,
        camera_2,
        reference_camera=args.reference_camera,
        max_delta_ms=args.max_stereo_delta_ms,
    )
    pairs, camera_1_depth, camera_2_depth, rejected_depth_pairs = (
        filter_stereo_pairs_by_depth(
            camera_1,
            camera_2,
            candidate_pairs,
            args.max_rgb_depth_delta_ms,
        )
    )
    camera_1_rgb = [first for first, _ in pairs]
    camera_2_rgb = [second for _, second in pairs]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for camera_id, recording, rgb_indices, depth_indices in (
            (1, camera_1, camera_1_rgb, camera_1_depth),
            (2, camera_2, camera_2_rgb, camera_2_depth),
        ):
            camera_dir = temporary / f"camera_{camera_id}"
            camera_dir.mkdir()
            _extract_images(
                recording,
                COLOR_DATA_TOPIC,
                rgb_indices,
                camera_dir / "rgb",
                "rgb",
                args.png_compression,
            )
            _extract_images(
                recording,
                DEPTH_DATA_TOPIC,
                depth_indices,
                camera_dir / "depth_raw",
                "depth",
                args.png_compression,
            )
            _write_timestamps(camera_dir / "rgb_timestamps.txt", recording.rgb, rgb_indices)
            _write_timestamps(
                camera_dir / "depth_timestamps.txt", recording.depth, depth_indices
            )
            _write_camera_csv(
                camera_dir / "frames.csv", recording, rgb_indices, depth_indices
            )

        annotation_camera = camera_1 if args.annotation_camera == 1 else camera_2
        annotation_rgb_indices = camera_1_rgb if args.annotation_camera == 1 else camera_2_rgb
        annotation_depth_indices = (
            camera_1_depth if args.annotation_camera == 1 else camera_2_depth
        )
        (temporary / "rgb").symlink_to(
            Path(f"camera_{args.annotation_camera}") / "rgb", target_is_directory=True
        )
        (temporary / "depth_raw").symlink_to(
            Path(f"camera_{args.annotation_camera}") / "depth_raw",
            target_is_directory=True,
        )
        _write_timestamps(
            temporary / "rgb_timestamps.txt", annotation_camera.rgb, annotation_rgb_indices
        )
        _write_timestamps(
            temporary / "depth_timestamps.txt",
            annotation_camera.depth,
            annotation_depth_indices,
        )
        _write_stereo_csv(temporary / "stereo_pairs.csv", camera_1, camera_2, pairs)

        stereo_offsets_ms = [
            (camera_2.rgb[second].timestamp_s - camera_1.rgb[first].timestamp_s)
            * 1000.0
            for first, second in pairs
        ]
        manifest = {
            "schema_version": 1,
            "frame_count": len(pairs),
            "candidate_stereo_pair_count": len(candidate_pairs),
            "dropped_for_rgb_depth_pairing": len(rejected_depth_pairs),
            "dropped_rgb_depth_pairs": rejected_depth_pairs,
            "reference_camera": args.reference_camera,
            "annotation_camera": args.annotation_camera,
            "image_transform": "none",
            "depth_values": "raw uint16 sensor units",
            "depth_registered_to_rgb": False,
            "pairing_clock": "RealSense metadata timestamp/system global time",
            "max_stereo_delta_ms": args.max_stereo_delta_ms,
            "max_rgb_depth_delta_ms": args.max_rgb_depth_delta_ms,
            "stereo_offset_ms": {
                "min": min(stereo_offsets_ms),
                "median": statistics.median(stereo_offsets_ms),
                "max": max(stereo_offsets_ms),
                "max_abs": max(abs(value) for value in stereo_offsets_ms),
            },
            "cameras": [
                _camera_manifest(1, camera_1, camera_1_rgb, camera_1_depth),
                _camera_manifest(2, camera_2, camera_2_rgb, camera_2_depth),
            ],
        }
        (temporary / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-1-db", type=Path, required=True)
    parser.add_argument("--camera-2-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-camera",
        type=int,
        choices=[1, 2],
        default=2,
        help="Camera whose RGB frames define the output timeline (default: 2)",
    )
    parser.add_argument(
        "--annotation-camera",
        type=int,
        choices=[1, 2],
        default=1,
        help="Camera exposed through the root rgb/depth_raw aliases (default: 1)",
    )
    parser.add_argument("--max-stereo-delta-ms", type=float, default=20.0)
    parser.add_argument("--max-rgb-depth-delta-ms", type=float, default=20.0)
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=3,
        metavar="0..9",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = convert(args)
    print(f"[ok] synchronized stereo episode: {output}")


if __name__ == "__main__":
    main()
