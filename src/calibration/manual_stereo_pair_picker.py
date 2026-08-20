#!/usr/bin/env python3
"""Local browser tool for manually matching and extracting stereo video frames.

Two source videos are shown independently.  The operator seeks each camera to
the same checkerboard instant and saves the pair as::

    <output>/camera_1/<index>_Color.png
    <output>/camera_2/<index>_Color.png

The shared numeric name is compatible with ``calibrate_stereo_checkerboard``.
Every save is also recorded in ``<output>/pairs.json`` with the two source
frame indices and timestamps.  Source videos are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
UI_PATH = Path(__file__).with_suffix(".html")
DEFAULT_CAMERA_1 = REPO_ROOT / "08_04" / "mh" / "1.mov"
DEFAULT_CAMERA_2 = REPO_ROOT / "08_04" / "sh" / "1.mov"
DEFAULT_OUTPUT = REPO_ROOT / "calibration_manual_pairs"
DEFAULT_CACHE = REPO_ROOT / "calibration_results" / "manual_pair_picker_cache"
IMAGE_NAME_RE = re.compile(r"^(?P<index>[1-9][0-9]*)_Color[.]png$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe_video(path: str | Path) -> dict[str, object]:
    """Return strict source metadata including a decoded frame count."""
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(source)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        (
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,"
            "nb_read_frames:format=duration"
        ),
        "-of",
        "json",
        str(source),
    ]
    payload = json.loads(_run(command).stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"video stream not found: {source}")
    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    frame_value = stream.get("nb_frames") or stream.get("nb_read_frames")
    frames = int(frame_value or 0)
    fps = _fraction(stream.get("avg_frame_rate")) or _fraction(
        stream.get("r_frame_rate")
    )
    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    if frames <= 0 and fps > 0.0 and duration > 0.0:
        frames = int(round(fps * duration))
    if width <= 0 or height <= 0 or frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"invalid video metadata for {source}: "
            f"{width}x{height}, frames={frames}, fps={fps}"
        )
    if duration <= 0.0:
        duration = frames / fps
    return {
        "path": str(source),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "duration": duration,
    }


def _cache_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def ensure_browser_preview(
    source: str | Path,
    cache_dir: str | Path,
) -> tuple[Path, dict[str, object]]:
    """Create a CFR H.264 proxy with a one-to-one source-frame mapping."""
    source_path = Path(source).expanduser().resolve()
    source_info = probe_video(source_path)
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"{_cache_key(source_path)}.mp4"
    if output.is_file() and output.stat().st_size > 0:
        preview_info = probe_video(output)
        if int(preview_info["frames"]) == int(source_info["frames"]):
            return output, source_info
        output.unlink()

    temporary = output.with_suffix(f".{os.getpid()}.building.mp4")
    temporary.unlink(missing_ok=True)
    fps = float(source_info["fps"])
    frame_count = int(source_info["frames"])
    # setpts removes VFR timing differences while retaining exactly one output
    # frame for every decoded source frame.  The UI can therefore address the
    # original video by integer frame index.
    filters = (
        f"setpts=N/({fps:.12g}*TB),"
        "scale=w='min(1280,iw)':h=-2:flags=lanczos,setsar=1"
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filters,
        "-frames:v",
        str(frame_count),
        "-r",
        f"{fps:.12g}",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        _run(command)
        preview_info = probe_video(temporary)
        if int(preview_info["frames"]) != frame_count:
            raise RuntimeError(
                "browser preview frame count mismatch: "
                f"{preview_info['frames']} != {frame_count}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output, source_info


def resolve_user_path(value: object, *, base: Path = REPO_ROOT) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def validate_pair_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("pair index must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("pair index must be an integer") from error
    if result < 1 or result > 999_999_999:
        raise ValueError("pair index must be in 1..999999999")
    return result


def pair_filename(pair_id: int) -> str:
    return f"{validate_pair_id(pair_id)}_Color.png"


def next_pair_id(output_root: str | Path) -> int:
    root = Path(output_root).expanduser().resolve()
    seen: set[int] = set()
    for camera_name in ("camera_1", "camera_2"):
        directory = root / camera_name
        if not directory.is_dir():
            continue
        for path in directory.glob("*_Color.png"):
            match = IMAGE_NAME_RE.fullmatch(path.name)
            if match:
                seen.add(int(match.group("index")))
    manifest = root / "pairs.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for item in payload.get("pairs", []):
                seen.add(int(item["pair_id"]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return max(seen, default=0) + 1


@dataclass(frozen=True)
class PickerConfig:
    camera_1: Path
    camera_2: Path
    output_root: Path
    preview_1: Path
    preview_2: Path
    info_1: dict[str, object]
    info_2: dict[str, object]

    def public(self) -> dict[str, object]:
        same_shape = (
            int(self.info_1["width"]),
            int(self.info_1["height"]),
        ) == (
            int(self.info_2["width"]),
            int(self.info_2["height"]),
        )
        return {
            "camera_1": {**self.info_1, "preview_url": "/media/camera_1.mp4"},
            "camera_2": {**self.info_2, "preview_url": "/media/camera_2.mp4"},
            "output_root": str(self.output_root),
            "next_pair_id": next_pair_id(self.output_root),
            "image_name_pattern": "<index>_Color.png",
            "same_image_size": same_shape,
        }


def build_config(
    camera_1: str | Path,
    camera_2: str | Path,
    output_root: str | Path,
    cache_dir: str | Path,
) -> PickerConfig:
    first = Path(camera_1).expanduser().resolve()
    second = Path(camera_2).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if first == second:
        raise ValueError("camera 1 and camera 2 videos must be different files")
    if not first.is_file():
        raise FileNotFoundError(first)
    if not second.is_file():
        raise FileNotFoundError(second)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "camera_1").mkdir(exist_ok=True)
    (output / "camera_2").mkdir(exist_ok=True)
    preview_1, info_1 = ensure_browser_preview(first, cache_dir)
    preview_2, info_2 = ensure_browser_preview(second, cache_dir)
    return PickerConfig(
        camera_1=first,
        camera_2=second,
        output_root=output,
        preview_1=preview_1,
        preview_2=preview_2,
        info_1=info_1,
        info_2=info_2,
    )


def _extract_frame(source: Path, frame_index: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    select = f"select=eq(n\\,{frame_index})"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        select,
        "-frames:v",
        "1",
        "-fps_mode",
        "passthrough",
        "-compression_level",
        "3",
        str(destination),
    ]
    _run(command)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"failed to extract source frame {frame_index}")


def _load_manifest(root: Path) -> dict[str, object]:
    path = root / "pairs.json"
    if not path.is_file():
        return {
            "schema_version": 1,
            "image_name_pattern": "<index>_Color.png",
            "pairs": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("pairs"), list):
        raise ValueError(f"invalid pair manifest: {path}")
    return payload


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    destination = root / "pairs.json"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".pairs.", suffix=".json.tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_frame_pair(
    config: PickerConfig,
    *,
    frame_1: object,
    frame_2: object,
    pair_id: object,
    overwrite: bool = False,
) -> dict[str, object]:
    """Extract two exact decoded source frames under one shared image name."""
    first_frame = int(frame_1)
    second_frame = int(frame_2)
    identifier = validate_pair_id(pair_id)
    if isinstance(frame_1, bool) or not 0 <= first_frame < int(config.info_1["frames"]):
        raise ValueError(
            f"camera 1 frame must be in 0..{int(config.info_1['frames']) - 1}"
        )
    if isinstance(frame_2, bool) or not 0 <= second_frame < int(config.info_2["frames"]):
        raise ValueError(
            f"camera 2 frame must be in 0..{int(config.info_2['frames']) - 1}"
        )
    filename = pair_filename(identifier)
    destination_1 = config.output_root / "camera_1" / filename
    destination_2 = config.output_root / "camera_2" / filename
    if not overwrite and (destination_1.exists() or destination_2.exists()):
        raise FileExistsError(
            f"pair {filename} already exists; choose another index or enable overwrite"
        )

    with tempfile.TemporaryDirectory(
        prefix=".stereo_pair.", dir=config.output_root
    ) as temporary_name:
        temporary = Path(temporary_name)
        staged_1 = temporary / "camera_1.png"
        staged_2 = temporary / "camera_2.png"
        _extract_frame(config.camera_1, first_frame, staged_1)
        _extract_frame(config.camera_2, second_frame, staged_2)
        os.replace(staged_1, destination_1)
        os.replace(staged_2, destination_2)

    record = {
        "pair_id": identifier,
        "filename": filename,
        "camera_1": {
            "video": str(config.camera_1),
            "frame": first_frame,
            "timestamp_s": first_frame / float(config.info_1["fps"]),
            "image": str(destination_1.relative_to(config.output_root)),
        },
        "camera_2": {
            "video": str(config.camera_2),
            "frame": second_frame,
            "timestamp_s": second_frame / float(config.info_2["fps"]),
            "image": str(destination_2.relative_to(config.output_root)),
        },
        "saved_at": _utc_now(),
    }
    manifest = _load_manifest(config.output_root)
    records = [
        item
        for item in manifest["pairs"]
        if int(item.get("pair_id", -1)) != identifier
    ]
    records.append(record)
    records.sort(key=lambda item: int(item["pair_id"]))
    manifest.update(
        {
            "schema_version": 1,
            "image_name_pattern": "<index>_Color.png",
            "output_root": str(config.output_root),
            "updated_at": _utc_now(),
            "pairs": records,
        }
    )
    _write_manifest(config.output_root, manifest)
    return {
        "record": record,
        "next_pair_id": next_pair_id(config.output_root),
        "manifest": str(config.output_root / "pairs.json"),
    }


class PickerApplication:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        self.config: PickerConfig | None = None
        self.lock = threading.RLock()

    def configure(self, camera_1: Path, camera_2: Path, output: Path) -> PickerConfig:
        with self.lock:
            self.config = build_config(
                camera_1,
                camera_2,
                output,
                self.cache_dir,
            )
            return self.config

    def require_config(self) -> PickerConfig:
        with self.lock:
            if self.config is None:
                raise RuntimeError("videos are not configured")
            return self.config


def _copy_range(
    handler: BaseHTTPRequestHandler,
    path: Path,
    content_type: str,
) -> None:
    size = path.stat().st_size
    start, end = 0, size - 1
    status = HTTPStatus.OK
    range_header = handler.headers.get("Range")
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        first, last = match.groups()
        if first:
            start = int(first)
            end = int(last) if last else end
        elif last:
            start = max(0, size - int(last))
        if start >= size or end < start:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        end = min(end, size - 1)
        status = HTTPStatus.PARTIAL_CONTENT
    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(length))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "no-cache")
    if status == HTTPStatus.PARTIAL_CONTENT:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if handler.command == "HEAD":
        return
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = length
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            handler.wfile.write(block)
            remaining -= len(block)


class PickerHandler(BaseHTTPRequestHandler):
    application: PickerApplication

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, status: HTTPStatus, error: Exception | str) -> None:
        self._json({"ok": False, "error": str(error)}, status)

    def _request_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("invalid request size")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                data = UI_PATH.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
                return
            if parsed.path == "/api/state":
                config = self.application.require_config()
                manifest = _load_manifest(config.output_root)
                self._json(
                    {
                        "ok": True,
                        "config": config.public(),
                        "pairs": manifest.get("pairs", [])[-20:],
                    }
                )
                return
            if parsed.path in ("/media/camera_1.mp4", "/media/camera_2.mp4"):
                config = self.application.require_config()
                path = (
                    config.preview_1
                    if parsed.path.endswith("camera_1.mp4")
                    else config.preview_2
                )
                _copy_range(self, path, "video/mp4")
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            print(f"GET failed: {error}")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, error)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._request_json()
            if parsed.path == "/api/configure":
                camera_1 = resolve_user_path(payload.get("camera_1"))
                camera_2 = resolve_user_path(payload.get("camera_2"))
                output = resolve_user_path(payload.get("output_root"))
                config = self.application.configure(camera_1, camera_2, output)
                self._json(
                    {
                        "ok": True,
                        "config": config.public(),
                        "pairs": _load_manifest(output).get("pairs", [])[-20:],
                    }
                )
                return
            if parsed.path == "/api/save":
                with self.application.lock:
                    config = self.application.require_config()
                    result = save_frame_pair(
                        config,
                        frame_1=payload.get("frame_1"),
                        frame_2=payload.get("frame_2"),
                        pair_id=payload.get("pair_id"),
                        overwrite=payload.get("overwrite") is True,
                    )
                self._json({"ok": True, **result})
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, TypeError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, error)
        except Exception as error:
            print(f"POST failed: {error}")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--camera1", type=Path, default=DEFAULT_CAMERA_1)
    parser.add_argument("--camera2", type=Path, default=DEFAULT_CAMERA_2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable not found: {executable}")
    if not UI_PATH.is_file():
        raise SystemExit(f"UI file not found: {UI_PATH}")

    application = PickerApplication(args.cache_dir.expanduser().resolve())
    print("Preparing browser previews...", flush=True)
    config = application.configure(
        args.camera1.expanduser().resolve(),
        args.camera2.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    PickerHandler.application = application
    server = ThreadingHTTPServer((args.host, args.port), PickerHandler)
    print(f"Camera 1: {config.camera_1}", flush=True)
    print(f"Camera 2: {config.camera_2}", flush=True)
    print(f"Output:   {config.output_root}", flush=True)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
