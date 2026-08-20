#!/usr/bin/env python3
"""Build a browser gallery and representative comparison for overlay QA."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote


def relative_url(path: Path, start: Path) -> str:
    return quote(Path(os.path.relpath(path.resolve(), start.resolve())).as_posix())


def midpoint_thumbnail(video: Path, output: Path) -> None:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    midpoint = max(0.0, float(probe.stdout.strip()) / 2.0)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{midpoint:.6f}", "-i", str(video), "-frames:v", "1",
            "-vf", "scale=320:320:force_original_aspect_ratio=decrease",
            "-q:v", "3", str(output),
        ],
        check=True,
    )


def representative_comparison(episodes: list[dict], output: Path) -> None:
    representatives = []
    for tag in ("0724", "0727", "0728"):
        item = next(value for value in episodes if value["source_tag"] == tag)
        root = Path(item["work_episode"])
        representatives.append((tag, (root / "source_video").resolve(), root / "robot_overlay.mp4"))

    command = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
    for _, source, overlay in representatives:
        command.extend(["-i", str(source), "-i", str(overlay)])
    filters = []
    rows = []
    for row, (tag, _, _) in enumerate(representatives):
        left = row * 2
        right = left + 1
        label = {"0724": "07/24", "0727": "07/27", "0728": "07/28"}[tag]
        for index, suffix, title in (
            (left, "source", f"{label} Original"),
            (right, "overlay", f"{label} Robot overlay"),
        ):
            filters.append(
                f"[{index}:v]setpts=PTS-STARTPTS,fps=10,"
                "scale=480:270:force_original_aspect_ratio=decrease,"
                "pad=480:270:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"drawtext=text='{title}':x=12:y=12:fontsize=22:"
                "fontcolor=white:box=1:boxcolor=black@0.65"
                f"[{tag}_{suffix}]"
            )
        filters.append(
            f"[{tag}_source][{tag}_overlay]hstack=inputs=2:shortest=1[row{row}]"
        )
        rows.append(f"[row{row}]")
    filters.append("".join(rows) + "vstack=inputs=3:shortest=1[out]")
    command.extend(
        [
            "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
            "-c:v", "libx264", "-crf", "22", "-preset", "medium",
            "-pix_fmt", "yuv420p", str(output),
        ]
    )
    subprocess.run(command, check=True)


def build_html(manifest: dict, output_dir: Path) -> None:
    cards = []
    for item in manifest["episodes"]:
        root = Path(item["work_episode"])
        flat_id = item["flat_id"]
        thumbnail = output_dir / "thumbnails" / f"{flat_id}.jpg"
        labels = ", ".join(f"{name}: {frames}" for name, frames in item["labels"].items())
        videos = [
            ("원본", (root / "source_video").resolve()),
            ("정합 확인", root / "alignment_preview.mp4"),
            ("최종 오버레이", root / "robot_overlay.mp4"),
        ]
        video_html = "".join(
            f"<figure><figcaption>{html.escape(title)}</figcaption>"
            f"<video controls preload='metadata' src='{relative_url(path, output_dir)}'></video></figure>"
            for title, path in videos
        )
        cards.append(
            "<section class='card'>"
            f"<h2>{html.escape(flat_id)}</h2>"
            f"<p>{item['frames']} frames · {item['fps']:.3g} fps · {html.escape(labels)}</p>"
            f"<img class='thumb' loading='lazy' src='{relative_url(thumbnail, output_dir)}'>"
            f"<div class='videos'>{video_html}</div>"
            "</section>"
        )

    comparison = relative_url(output_dir / "representative_original_vs_overlay.mp4", output_dir)
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kitchen robot overlay visual QA</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
a{{color:#8cc8ff}} .summary{{max-width:1100px}} .card{{border-top:1px solid #444;padding:18px 0}}
.thumb{{width:240px;max-height:240px;object-fit:contain;background:#000}}
.videos{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px}}
figure{{margin:0}} figcaption{{margin:5px 0}} video{{width:100%;max-height:430px;background:#000}}
@media(max-width:900px){{.videos{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="summary"><h1>Kitchen robot overlay visual QA</h1>
<p>총 {manifest['audited_episodes']}개 에피소드입니다. 각 항목은 원본, 사람 손이 남은 정합 확인 영상,
사람 손을 제거한 최종 로봇 오버레이 순서입니다.</p>
<p><a href="{comparison}">날짜별 원본/최종 대표 비교영상 열기</a></p></div>
{''.join(cards)}
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("audited_episodes") != manifest.get("expected_episodes"):
        raise ValueError("visual QA requires a complete audited manifest")
    output_dir = args.output_dir.resolve()
    thumbnails = output_dir / "thumbnails"
    thumbnails.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(manifest["episodes"], 1):
        output = thumbnails / f"{item['flat_id']}.jpg"
        if not output.is_file():
            midpoint_thumbnail(Path(item["robot_overlay"]), output)
        print(f"thumbnail {index}/{len(manifest['episodes'])}: {item['flat_id']}")
    comparison = output_dir / "representative_original_vs_overlay.mp4"
    if not comparison.is_file():
        representative_comparison(manifest["episodes"], comparison)
    build_html(manifest, output_dir)
    print(f"gallery={output_dir / 'index.html'}")
    print(f"comparison={comparison}")


if __name__ == "__main__":
    main()
