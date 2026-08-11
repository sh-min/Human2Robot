"""Build a frame-synchronized, labelled 4x2 comparison video.

The command intentionally rejects inputs whose frame counts, frame rates, or
durations differ.  It resets each input timeline to frame zero before stacking,
so panel synchronization is based on frame index rather than container start
timestamps.  Every 640x360 source panel is kept unobstructed: its label is drawn
inside a separate 640x40 black header, producing a 2560x800 output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


PANEL_WIDTH = 640
PANEL_HEIGHT = 360
HEADER_HEIGHT = 40
TILE_HEIGHT = HEADER_HEIGHT + PANEL_HEIGHT
GRID_COLUMNS = 4
GRID_ROWS = 2
EXPECTED_VIDEO_COUNT = GRID_COLUMNS * GRID_ROWS
OUTPUT_WIDTH = PANEL_WIDTH * GRID_COLUMNS
OUTPUT_HEIGHT = TILE_HEIGHT * GRID_ROWS
DEFAULT_DURATION_TOLERANCE_S = 0.001


@dataclass(frozen=True)
class NamedVideo:
    label: str
    path: Path


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    frame_count: int
    fps: Fraction
    duration_s: float
    codec_name: str = ""
    pixel_format: str = ""


@dataclass(frozen=True)
class GridLayout:
    """Geometry for a labelled comparison grid."""

    columns: int
    rows: int
    panel_width: int = PANEL_WIDTH
    panel_height: int = PANEL_HEIGHT
    header_height: int = HEADER_HEIGHT

    def __post_init__(self) -> None:
        for field_name in (
            "columns",
            "rows",
            "panel_width",
            "panel_height",
            "header_height",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")

    @property
    def video_count(self) -> int:
        return self.columns * self.rows

    @property
    def tile_height(self) -> int:
        return self.header_height + self.panel_height

    @property
    def output_width(self) -> int:
        return self.columns * self.panel_width

    @property
    def output_height(self) -> int:
        return self.rows * self.tile_height


DEFAULT_GRID_LAYOUT = GridLayout(columns=GRID_COLUMNS, rows=GRID_ROWS)


def _parse_positive_fraction(value: object, field: str) -> Fraction:
    text = str(value or "").strip()
    if not text or text in {"N/A", "0/0"}:
        raise ValueError(f"missing {field}: {value!r}")
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return result


def _parse_positive_int(value: object, field: str) -> int:
    text = str(value or "").strip()
    if not text or text == "N/A":
        raise ValueError(f"missing {field}: {value!r}")
    try:
        result = int(text)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return result


def _parse_duration(stream: dict, container: dict) -> float:
    for value in (stream.get("duration"), container.get("duration")):
        text = str(value or "").strip()
        if not text or text == "N/A":
            continue
        try:
            duration_s = float(text)
        except ValueError:
            continue
        if duration_s > 0:
            return duration_s
    raise ValueError("ffprobe did not report a positive video duration")


def probe_video(path: Path, ffprobe: str = "ffprobe") -> VideoMetadata:
    """Read exact frame count and stream metadata from the first video stream."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"video does not exist: {resolved}")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        (
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,"
            "nb_read_frames,duration,codec_name,pix_fmt:format=duration"
        ),
        "-of",
        "json",
        str(resolved),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffprobe executable not found: {ffprobe}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown ffprobe error").strip()
        raise RuntimeError(f"could not probe {resolved}: {detail}") from exc

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"invalid ffprobe response for {resolved}") from exc

    frame_value = stream.get("nb_read_frames")
    if str(frame_value or "").strip() in {"", "N/A"}:
        frame_value = stream.get("nb_frames")
    fps_value = stream.get("avg_frame_rate")
    if str(fps_value or "").strip() in {"", "N/A", "0/0"}:
        fps_value = stream.get("r_frame_rate")

    return VideoMetadata(
        path=resolved,
        width=_parse_positive_int(stream.get("width"), "width"),
        height=_parse_positive_int(stream.get("height"), "height"),
        frame_count=_parse_positive_int(frame_value, "frame count"),
        fps=_parse_positive_fraction(fps_value, "frame rate"),
        duration_s=_parse_duration(stream, payload.get("format", {})),
        codec_name=str(stream.get("codec_name") or ""),
        pixel_format=str(stream.get("pix_fmt") or ""),
    )


def parse_named_videos(raw_videos: Sequence[Sequence[str]]) -> list[NamedVideo]:
    if len(raw_videos) != EXPECTED_VIDEO_COUNT:
        raise ValueError(
            f"exactly {EXPECTED_VIDEO_COUNT} --video entries are required; "
            f"received {len(raw_videos)}"
        )

    videos: list[NamedVideo] = []
    labels: set[str] = set()
    for raw in raw_videos:
        if len(raw) != 2:
            raise ValueError("each --video requires LABEL and PATH")
        label = raw[0].strip()
        if not label:
            raise ValueError("video labels must not be empty")
        if label in labels:
            raise ValueError(f"duplicate video label: {label!r}")
        labels.add(label)
        videos.append(NamedVideo(label=label, path=Path(raw[1])))
    return videos


def validate_grid_input_metadata(
    metadata: Sequence[VideoMetadata],
    expected_video_count: int,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
) -> VideoMetadata:
    """Return the reference metadata, or fail with every detected mismatch."""

    if expected_video_count <= 0:
        raise ValueError("expected video count must be positive")
    if len(metadata) != expected_video_count:
        raise ValueError(
            f"expected metadata for {expected_video_count} videos, "
            f"received {len(metadata)}"
        )
    if duration_tolerance_s < 0:
        raise ValueError("duration tolerance must be non-negative")

    reference = metadata[0]
    errors: list[str] = []
    for candidate in metadata[1:]:
        prefix = str(candidate.path)
        if candidate.frame_count != reference.frame_count:
            errors.append(
                f"{prefix}: frame count {candidate.frame_count} != "
                f"{reference.frame_count}"
            )
        if candidate.fps != reference.fps:
            errors.append(
                f"{prefix}: fps {candidate.fps} != {reference.fps}"
            )
        duration_delta = abs(candidate.duration_s - reference.duration_s)
        if duration_delta > duration_tolerance_s:
            errors.append(
                f"{prefix}: duration {candidate.duration_s:.6f}s != "
                f"{reference.duration_s:.6f}s "
                f"(delta {duration_delta:.6f}s)"
            )
    if errors:
        raise ValueError("input videos are not synchronized:\n  " + "\n  ".join(errors))
    return reference


def validate_input_metadata(
    metadata: Sequence[VideoMetadata],
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
) -> VideoMetadata:
    """Validate the original fixed 4x2 comparison contract."""

    return validate_grid_input_metadata(
        metadata,
        EXPECTED_VIDEO_COUNT,
        duration_tolerance_s,
    )


def _quote_filter_path(path: Path) -> str:
    # Quoting here is for FFmpeg's filter parser, not a shell.  Commands are
    # always passed to subprocess as an argument vector.
    escaped = str(path).replace("\\", "\\\\").replace(":", "\\:")
    escaped = escaped.replace("'", "\\'")
    return f"'{escaped}'"


def build_grid_filter_graph(
    label_files: Sequence[Path],
    font_file: Path,
    fps: Fraction,
    layout: GridLayout,
) -> str:
    if len(label_files) != layout.video_count:
        raise ValueError(f"expected {layout.video_count} label files")

    font = _quote_filter_path(font_file)
    filters: list[str] = []
    for index, label_file in enumerate(label_files):
        textfile = _quote_filter_path(label_file)
        filters.append(
            f"[{index}:v]"
            f"scale={layout.panel_width}:{layout.panel_height}:flags=lanczos,"
            "setsar=1,"
            f"setpts=N*{fps.denominator}/({fps.numerator}*TB)"
            f"[image{index}]"
        )
        # Render the label on its own video stream.  This guarantees that even
        # oversized or multi-line text is clipped by the header boundary
        # and can never cover source imagery.
        filters.append(
            f"color=c=black:s={layout.panel_width}x{layout.header_height}:"
            f"r={fps.numerator}/{fps.denominator},"
            "setsar=1,"
            f"setpts=N*{fps.denominator}/({fps.numerator}*TB)"
            f"[header_bg{index}]"
        )
        filters.append(
            f"[header_bg{index}]"
            f"drawtext=fontfile={font}:textfile={textfile}:expansion=none:"
            "fontcolor=white:fontsize=24:"
            "x=max(10\\,min((w-text_w)/2\\,w-text_w-10)):"
            "y=max(0\\,(h-text_h)/2):fix_bounds=1"
            f"[header{index}]"
        )
        filters.append(
            f"[header{index}][image{index}]"
            f"vstack=inputs=2:shortest=1[tile{index}]"
        )

    inputs = "".join(f"[tile{index}]" for index in range(layout.video_count))
    coordinates = "|".join(
        f"{column * layout.panel_width}_{row * layout.tile_height}"
        for row in range(layout.rows)
        for column in range(layout.columns)
    )
    filters.append(
        f"{inputs}xstack=inputs={len(label_files)}:layout={coordinates}:"
        "fill=black:shortest=1,format=yuv420p[vout]"
    )
    return ";".join(filters)


def build_filter_graph(
    label_files: Sequence[Path],
    font_file: Path,
    fps: Fraction,
) -> str:
    """Build the filter graph for the original fixed 4x2 layout."""

    return build_grid_filter_graph(
        label_files,
        font_file,
        fps,
        DEFAULT_GRID_LAYOUT,
    )


def _find_default_font() -> Path:
    candidates = (
        Path("/usr/share/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no supported label font found; pass a font with --font-file"
    )


def _validate_rendered_grid_output(
    rendered: VideoMetadata,
    reference: VideoMetadata,
    duration_tolerance_s: float,
    layout: GridLayout,
) -> None:
    errors: list[str] = []
    if (rendered.width, rendered.height) != (
        layout.output_width,
        layout.output_height,
    ):
        errors.append(
            f"geometry {rendered.width}x{rendered.height} != "
            f"{layout.output_width}x{layout.output_height}"
        )
    if rendered.frame_count != reference.frame_count:
        errors.append(
            f"frame count {rendered.frame_count} != {reference.frame_count}"
        )
    if rendered.fps != reference.fps:
        errors.append(f"fps {rendered.fps} != {reference.fps}")
    if abs(rendered.duration_s - reference.duration_s) > duration_tolerance_s:
        errors.append(
            f"duration {rendered.duration_s:.6f}s != "
            f"{reference.duration_s:.6f}s"
        )
    if rendered.codec_name != "h264":
        errors.append(f"codec {rendered.codec_name!r} != 'h264'")
    if rendered.pixel_format != "yuv420p":
        errors.append(f"pixel format {rendered.pixel_format!r} != 'yuv420p'")
    if errors:
        raise RuntimeError("rendered comparison failed validation: " + "; ".join(errors))


def _validate_rendered_output(
    rendered: VideoMetadata,
    reference: VideoMetadata,
    duration_tolerance_s: float,
) -> None:
    """Validate the original fixed 4x2 output contract."""

    _validate_rendered_grid_output(
        rendered,
        reference,
        duration_tolerance_s,
        DEFAULT_GRID_LAYOUT,
    )


def render_comparison_grid_layout(
    videos: Sequence[NamedVideo],
    output: Path,
    *,
    layout: GridLayout,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    font_file: Path | None = None,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
    overwrite: bool = False,
    crf: int = 18,
    preset: str = "medium",
) -> VideoMetadata:
    if len(videos) != layout.video_count:
        raise ValueError(f"exactly {layout.video_count} videos are required")
    if not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51")

    output = output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must use the .mp4 extension")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    input_metadata = [probe_video(video.path, ffprobe) for video in videos]
    reference = validate_grid_input_metadata(
        input_metadata,
        layout.video_count,
        duration_tolerance_s,
    )
    selected_font = (font_file or _find_default_font()).expanduser().resolve()
    if not selected_font.is_file():
        raise FileNotFoundError(f"font does not exist: {selected_font}")

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.",
        suffix=".mp4",
        dir=output.parent,
    )
    os.close(temporary_fd)
    temporary_output = Path(temporary_name)
    temporary_output.unlink()
    try:
        with tempfile.TemporaryDirectory(prefix="comparison_grid_labels_") as temp_dir:
            label_files: list[Path] = []
            for index, video in enumerate(videos):
                label_file = Path(temp_dir) / f"label_{index}.txt"
                label_file.write_text(video.label, encoding="utf-8")
                label_files.append(label_file)
            filter_graph = build_grid_filter_graph(
                label_files,
                selected_font,
                reference.fps,
                layout,
            )

            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
            for video in videos:
                command.extend(["-i", str(video.path.expanduser().resolve())])
            command.extend(
                [
                    "-filter_complex",
                    filter_graph,
                    "-map",
                    "[vout]",
                    "-an",
                    "-frames:v",
                    str(reference.frame_count),
                    "-r",
                    f"{reference.fps.numerator}/{reference.fps.denominator}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                ]
            )
            try:
                subprocess.run(command, check=True)
            except FileNotFoundError as exc:
                raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"ffmpeg grid render failed with status {exc.returncode}") from exc

        rendered = probe_video(temporary_output, ffprobe)
        _validate_rendered_grid_output(
            rendered,
            reference,
            duration_tolerance_s,
            layout,
        )
        os.replace(temporary_output, output)
        return VideoMetadata(
            path=output,
            width=rendered.width,
            height=rendered.height,
            frame_count=rendered.frame_count,
            fps=rendered.fps,
            duration_s=rendered.duration_s,
            codec_name=rendered.codec_name,
            pixel_format=rendered.pixel_format,
        )
    finally:
        temporary_output.unlink(missing_ok=True)


def render_comparison_grid(
    videos: Sequence[NamedVideo],
    output: Path,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    font_file: Path | None = None,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
    overwrite: bool = False,
    crf: int = 18,
    preset: str = "medium",
) -> VideoMetadata:
    """Render the original fixed 4x2 comparison layout."""

    return render_comparison_grid_layout(
        videos,
        output,
        layout=DEFAULT_GRID_LAYOUT,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        font_file=font_file,
        duration_tolerance_s=duration_tolerance_s,
        overwrite=overwrite,
        crf=crf,
        preset=preset,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="panel label and input path; repeat exactly eight times in display order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-file", type=Path, default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_S,
        metavar="SECONDS",
        help="maximum accepted container-duration difference (default: 0.001)",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        videos = parse_named_videos(args.video)
        result = render_comparison_grid(
            videos,
            args.output,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            font_file=args.font_file,
            duration_tolerance_s=args.duration_tolerance,
            overwrite=args.overwrite,
            crf=args.crf,
            preset=args.preset,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Wrote {result.path} ({result.width}x{result.height}, "
        f"{result.frame_count} frames, {float(result.fps):.6f} fps, "
        f"{result.duration_s:.6f}s)"
    )


if __name__ == "__main__":
    main()
