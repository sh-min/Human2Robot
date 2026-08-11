"""
Download YouTube videos of two-handed Rubik's cube solving.
Saves MP4 files to /virtual_lab/ljw_rvlab/lab/cube_dataset/web_data

- 다운로드 기록은 save_dir/download_log.json에 저장
- 재실행 시 이미 다운로드된 video_id는 스킵
- 기본 실행 시 최대 50개 (5 쿼리 × max_results=10)
"""

import os
import json
import argparse
from datetime import datetime
import yt_dlp


SAVE_DIR = "/virtual_lab/ljw_rvlab/lab/cube_dataset/web_data"
LOG_FILE = "download_log.json"

SEARCH_QUERIES = [
    "rubik's cube tutorial POV both hands close up",
    "how to solve rubik's cube step by step hands close up",
    "rubik's cube beginner tutorial first person view",
    "rubik's cube slow solve tutorial both hands",
    "큐브 맞추는 방법 손 클로즈업 튜토리얼",
    "rubik's cube tutorial wrist cam",
    "rubik cube solve tutorial top view hands",
    "rubik's cube finger tricks tutorial close up",
]


def load_log(log_path: str) -> dict:
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {}


def save_log(log_path: str, log: dict):
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def get_ydl_opts(save_dir: str, max_duration: int, downloaded_ids: set) -> dict:
    def already_downloaded(info, *, incomplete):
        if info.get("id") in downloaded_ids:
            return "Already downloaded, skipping"
        if info.get("duration", 0) >= max_duration:
            return f"Duration {info.get('duration')}s exceeds limit {max_duration}s"
        return None

    return {
        "format": "bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[acodec^=mp4a]/bestvideo[vcodec^=avc1]+bestaudio/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(save_dir, "%(id)s_%(title).80s.%(ext)s"),
        "ignoreerrors": True,
        "quiet": False,
        "no_warnings": False,
        "match_filter": already_downloaded,
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
    }


def extract_entries(info: dict) -> list:
    if not info:
        return []
    if "entries" in info:
        return [e for e in info["entries"] if e]
    return [info]


def search_and_download(
    query: str,
    save_dir: str,
    log: dict,
    max_results: int = 10,
    max_duration: int = 600,
) -> list:
    downloaded_ids = set(log.keys())
    opts = get_ydl_opts(save_dir, max_duration, downloaded_ids)

    newly_downloaded = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=True)
        for entry in extract_entries(info):
            vid_id = entry.get("id")
            if vid_id and vid_id not in downloaded_ids:
                file_path = ydl.prepare_filename(entry).replace(".webm", ".mp4").replace(".mkv", ".mp4")
                log[vid_id] = {
                    "title": entry.get("title", ""),
                    "url": entry.get("webpage_url", ""),
                    "duration": entry.get("duration"),
                    "file_path": file_path,
                    "query": query,
                    "downloaded_at": datetime.now().isoformat(),
                }
                newly_downloaded.append(vid_id)
    return newly_downloaded


def download_from_urls(
    urls: list,
    save_dir: str,
    log: dict,
    max_duration: int = 600,
) -> list:
    downloaded_ids = set(log.keys())
    opts = get_ydl_opts(save_dir, max_duration, downloaded_ids)

    newly_downloaded = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=True)
                for entry in extract_entries(info):
                    vid_id = entry.get("id")
                    if vid_id and vid_id not in downloaded_ids:
                        file_path = ydl.prepare_filename(entry).replace(".webm", ".mp4").replace(".mkv", ".mp4")
                        log[vid_id] = {
                            "title": entry.get("title", ""),
                            "url": url,
                            "duration": entry.get("duration"),
                            "file_path": file_path,
                            "query": None,
                            "downloaded_at": datetime.now().isoformat(),
                        }
                        newly_downloaded.append(vid_id)
            except Exception as e:
                print(f"[WARN] Failed: {url}: {e}")
    return newly_downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download two-handed Rubik's cube solving videos from YouTube"
    )
    parser.add_argument("--save_dir", default=SAVE_DIR)
    parser.add_argument("--max_results", type=int, default=10, help="Max videos per query (default: 10, total: queries×max_results)")
    parser.add_argument("--max_duration", type=int, default=600, help="Max duration in seconds (default: 600)")
    parser.add_argument("--queries", nargs="+", default=None, help="Custom search queries")
    parser.add_argument("--urls", nargs="+", default=None, help="Explicit YouTube URLs to download")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, LOG_FILE)
    log = load_log(log_path)
    print(f"[INFO] Save dir: {args.save_dir}")
    print(f"[INFO] Already downloaded: {len(log)} videos (from {log_path})")

    if args.urls:
        new_ids = download_from_urls(args.urls, args.save_dir, log, args.max_duration)
        save_log(log_path, log)
        print(f"[DONE] Downloaded {len(new_ids)} new videos. Total: {len(log)}")
        return

    queries = args.queries if args.queries else SEARCH_QUERIES
    print(f"[INFO] Max per query: {args.max_results}, queries: {len(queries)} → up to {args.max_results * len(queries)} videos total")

    total_new = []
    for q in queries:
        print(f"\n[INFO] Searching: '{q}'")
        new_ids = search_and_download(q, args.save_dir, log, args.max_results, args.max_duration)
        save_log(log_path, log)  # 쿼리마다 저장 (중간에 끊겨도 기록 보존)
        print(f"[INFO] +{len(new_ids)} new videos (total so far: {len(log)})")
        total_new.extend(new_ids)

    print(f"\n[DONE] New this run: {len(total_new)}, Total in log: {len(log)}")


if __name__ == "__main__":
    main()
