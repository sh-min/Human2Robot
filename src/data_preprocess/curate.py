"""
curate.py - WebUI for curating downloaded YouTube videos into frame datasets.

Usage:
    conda activate vjepa2-312
    python curate.py [--port 7860] [--host 0.0.0.0]

Remote access options:
    - Direct:     http://<server-ip>:7860  (if port is open)
    - SSH tunnel: ssh -L 7860:localhost:7860 user@server  →  http://localhost:7860

curate_log.json is saved alongside download_log.json in web_data/.
Each video has status: pending | curated | denied
Curated frames are saved to: web_data_curated/<scene_name>/rgb/frame_XXXXXX.jpg
"""

import os
import json
import subprocess
import argparse
import re
from pathlib import Path

import gradio as gr

WEB_DATA_DIR = "/virtual_lab/ljw_rvlab/lab/cube_dataset/web_data"
CURATED_DIR = "/virtual_lab/ljw_rvlab/lab/cube_dataset/web_data_curated"
DOWNLOAD_LOG = os.path.join(WEB_DATA_DIR, "download_log.json")
CURATE_LOG = os.path.join(WEB_DATA_DIR, "curate_log.json")
DEFAULT_VIDEO_FPS = 30
DEFAULT_EXTRACT_FPS = 10
DEFAULT_PORT = 7860


# --- Helpers ---

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def find_video_file(vid_id, stored_path):
    if stored_path and os.path.exists(stored_path):
        return stored_path
    for f in Path(WEB_DATA_DIR).glob(f"{vid_id}_*.mp4"):
        return str(f)
    return None


def sanitize(s):
    return re.sub(r"[^\w\-]", "_", s).strip("_")[:50]


def frame_to_sec(frame, video_fps):
    return round(int(frame) / float(video_fps), 4)


def sec_to_frame(sec, video_fps):
    return int(round(float(sec) * float(video_fps)))


# --- Data ---

def get_video_choices(filter_status="pending"):
    download_log = load_json(DOWNLOAD_LOG)
    curate_log = load_json(CURATE_LOG)
    choices = []
    for vid_id, info in download_log.items():
        if not find_video_file(vid_id, info.get("file_path", "")):
            continue
        status = curate_log.get(vid_id, {}).get("status", "pending")
        if filter_status != "all" and status != filter_status:
            continue
        n_clips = len(curate_log.get(vid_id, {}).get("clips", []))
        clip_tag = f" [{n_clips}clips]" if n_clips else ""
        label = f"[{status.upper()}]{clip_tag} {info.get('title', vid_id)[:60]}"
        choices.append((label, vid_id))
    return choices


def get_info_text(vid_id):
    if not vid_id:
        return ""
    download_log = load_json(DOWNLOAD_LOG)
    curate_log = load_json(CURATE_LOG)
    info = download_log.get(vid_id, {})
    cur = curate_log.get(vid_id, {})
    clips = cur.get("clips", [])
    lines = [
        f"ID     : {vid_id}",
        f"Title  : {info.get('title', '')}",
        f"Duration: {info.get('duration', '?')}s  |  Status: {cur.get('status', 'pending')}",
        f"Query  : {info.get('query', '')}",
        f"Clips  : {len(clips)}",
    ]
    for i, c in enumerate(clips):
        sf = c.get("start_frame", "?")
        ef = c.get("end_frame", "?")
        lines.append(
            f"  [{i+1}] {c['scene_name']}  frame {sf}~{ef}  {c['frame_count']}frames@{c['extract_fps']}fps"
        )
    return "\n".join(lines)


# --- Actions ---

def do_save_clip(vid_id, start_frame, end_frame, scene_name, video_fps, extract_fps):
    if not vid_id:
        return "No video selected.", "", ""
    start_frame, end_frame = int(start_frame), int(end_frame)
    video_fps = float(video_fps)
    extract_fps = int(extract_fps)
    if start_frame >= end_frame:
        return "Error: start frame must be less than end frame.", get_info_text(vid_id), scene_name
    if not scene_name.strip():
        return "Error: scene name is required.", get_info_text(vid_id), scene_name

    download_log = load_json(DOWNLOAD_LOG)
    curate_log = load_json(CURATE_LOG)

    fp = find_video_file(vid_id, download_log.get(vid_id, {}).get("file_path", ""))
    if not fp:
        return f"Error: video file not found for {vid_id}", "", scene_name

    start_sec = frame_to_sec(start_frame, video_fps)
    end_sec = frame_to_sec(end_frame, video_fps)

    out_dir = os.path.join(CURATED_DIR, scene_name.strip(), "rgb")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec), "-to", str(end_sec),
        "-i", fp,
        "-vf", f"fps={extract_fps}",
        "-q:v", "2",
        os.path.join(out_dir, "frame_%06d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"ffmpeg error:\n{result.stderr[-600:]}", get_info_text(vid_id), scene_name

    frame_count = len(list(Path(out_dir).glob("*.jpg")))

    if vid_id not in curate_log:
        curate_log[vid_id] = {"status": "curated", "clips": []}
    curate_log[vid_id]["status"] = "curated"
    curate_log[vid_id].setdefault("clips", []).append({
        "scene_name": scene_name.strip(),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start": start_sec,
        "end": end_sec,
        "video_fps": video_fps,
        "extract_fps": extract_fps,
        "output_dir": out_dir,
        "frame_count": frame_count,
    })
    save_json(CURATE_LOG, curate_log)

    n_clips = len(curate_log[vid_id]["clips"])
    title = download_log.get(vid_id, {}).get("title", vid_id)
    next_name = f"{sanitize(title)}_{n_clips + 1:03d}"

    return f"Saved {frame_count} frames → {out_dir}", get_info_text(vid_id), next_name


def do_deny(vid_id, filter_status):
    if not vid_id:
        return "No video selected.", gr.Dropdown(choices=[]), None, ""
    curate_log = load_json(CURATE_LOG)
    curate_log[vid_id] = {"status": "denied", "clips": []}
    save_json(CURATE_LOG, curate_log)
    choices = get_video_choices(filter_status)
    return f"Denied: {vid_id}", gr.Dropdown(choices=choices, value=None), None, ""


# --- UI ---

# JS: read current video playback time and convert to frame number
_JS_MARK = "(video_fps) => { const v = document.querySelector('video'); if (!v) return 0; return Math.round(v.currentTime * video_fps); }"


def build_ui():
    with gr.Blocks(title="Cube Video Curator") as demo:
        gr.Markdown("# Cube Video Curator")
        selected_vid_id = gr.State(None)

        with gr.Row():
            # Left: video list
            with gr.Column(scale=1, min_width=280):
                filter_radio = gr.Radio(
                    choices=["pending", "curated", "denied", "all"],
                    value="pending",
                    label="Filter",
                )
                video_dropdown = gr.Dropdown(label="Video list", choices=[], interactive=True)
                refresh_btn = gr.Button("Refresh list", size="sm")
                video_info = gr.Textbox(label="Video info", lines=9, interactive=False)

            # Right: player + controls
            with gr.Column(scale=2):
                video_player = gr.Video(
                    label="Preview",
                    interactive=False,
                    height=420,
                )

                with gr.Row():
                    prev_frame_btn = gr.Button("◀ 1 frame", size="sm")
                    next_frame_btn = gr.Button("1 frame ▶", size="sm")

                with gr.Row():
                    video_fps_input = gr.Number(
                        label="Video FPS (for frame↔time)",
                        value=DEFAULT_VIDEO_FPS, precision=1, minimum=1, maximum=240,
                    )
                    extract_fps_input = gr.Number(
                        label="Extract FPS (frames to save)",
                        value=DEFAULT_EXTRACT_FPS, precision=0, minimum=1, maximum=120,
                    )

                gr.Markdown("**Start frame**")
                with gr.Row():
                    mark_start_btn = gr.Button("Mark here", size="sm")
                    start_dec_btn = gr.Button("-1", size="sm", min_width=40)
                    start_frame = gr.Number(value=0, precision=0, minimum=0, label="Start frame")
                    start_inc_btn = gr.Button("+1", size="sm", min_width=40)
                    start_time_display = gr.Textbox(value="0.0000s", label="= seconds", interactive=False, min_width=90)

                gr.Markdown("**End frame**")
                with gr.Row():
                    mark_end_btn = gr.Button("Mark here", size="sm")
                    end_dec_btn = gr.Button("-1", size="sm", min_width=40)
                    end_frame = gr.Number(value=30, precision=0, minimum=0, label="End frame")
                    end_inc_btn = gr.Button("+1", size="sm", min_width=40)
                    end_time_display = gr.Textbox(value="1.0000s", label="= seconds", interactive=False, min_width=90)

                scene_name_input = gr.Textbox(
                    label="Scene name",
                    placeholder="e.g. cube_solve_001",
                )
                with gr.Row():
                    save_btn = gr.Button("Save clip", variant="primary")
                    deny_btn = gr.Button("Deny video", variant="stop")
                status_box = gr.Textbox(label="Status", interactive=False, lines=2)

        # --- Handlers ---

        def refresh(filter_status):
            return gr.Dropdown(choices=get_video_choices(filter_status), value=None)

        def on_select(vid_id):
            if not vid_id:
                return None, "", None, ""
            download_log = load_json(DOWNLOAD_LOG)
            info = download_log.get(vid_id, {})
            fp = find_video_file(vid_id, info.get("file_path", ""))
            curate_log = load_json(CURATE_LOG)
            n_clips = len(curate_log.get(vid_id, {}).get("clips", []))
            suggested = f"{sanitize(info.get('title', vid_id))}_{n_clips + 1:03d}"
            return vid_id, get_info_text(vid_id), fp, suggested

        def update_start_time(frame, fps):
            return f"{frame_to_sec(frame, fps):.4f}s"

        def update_end_time(frame, fps):
            return f"{frame_to_sec(frame, fps):.4f}s"

        def update_both_times(fps):
            return gr.update(), gr.update()  # triggers will re-fire

        def inc_start(f): return int(f) + 1
        def dec_start(f): return max(0, int(f) - 1)
        def inc_end(f):   return int(f) + 1
        def dec_end(f):   return max(0, int(f) - 1)

        # List refresh
        refresh_btn.click(refresh, inputs=[filter_radio], outputs=[video_dropdown])
        filter_radio.change(refresh, inputs=[filter_radio], outputs=[video_dropdown])

        # Video select
        video_dropdown.change(
            on_select,
            inputs=[video_dropdown],
            outputs=[selected_vid_id, video_info, video_player, scene_name_input],
        )

        # Frame seek buttons (pure JS, no Python call)
        prev_frame_btn.click(
            fn=None,
            inputs=[video_fps_input],
            outputs=[],
            js="(fps) => { const v = document.querySelector('video'); if(v){ v.pause(); v.currentTime = Math.max(0, v.currentTime - 1/fps); } }",
        )
        next_frame_btn.click(
            fn=None,
            inputs=[video_fps_input],
            outputs=[],
            js="(fps) => { const v = document.querySelector('video'); if(v){ v.pause(); v.currentTime = v.currentTime + 1/fps; } }",
        )

        # Mark buttons (JS reads video.currentTime → frame number)
        mark_start_btn.click(
            fn=None,
            inputs=[video_fps_input],
            outputs=[start_frame],
            js=_JS_MARK,
        )
        mark_end_btn.click(
            fn=None,
            inputs=[video_fps_input],
            outputs=[end_frame],
            js=_JS_MARK,
        )

        # +1/-1 buttons
        start_inc_btn.click(inc_start, inputs=[start_frame], outputs=[start_frame])
        start_dec_btn.click(dec_start, inputs=[start_frame], outputs=[start_frame])
        end_inc_btn.click(inc_end, inputs=[end_frame], outputs=[end_frame])
        end_dec_btn.click(dec_end, inputs=[end_frame], outputs=[end_frame])

        # Seconds display update
        start_frame.change(update_start_time, inputs=[start_frame, video_fps_input], outputs=[start_time_display])
        end_frame.change(update_end_time, inputs=[end_frame, video_fps_input], outputs=[end_time_display])
        video_fps_input.change(update_start_time, inputs=[start_frame, video_fps_input], outputs=[start_time_display])
        video_fps_input.change(update_end_time, inputs=[end_frame, video_fps_input], outputs=[end_time_display])

        # Save / Deny
        save_btn.click(
            do_save_clip,
            inputs=[selected_vid_id, start_frame, end_frame, scene_name_input, video_fps_input, extract_fps_input],
            outputs=[status_box, video_info, scene_name_input],
        )
        deny_btn.click(
            do_deny,
            inputs=[selected_vid_id, filter_radio],
            outputs=[status_box, video_dropdown, video_player, video_info],
        )

        demo.load(refresh, inputs=[filter_radio], outputs=[video_dropdown])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Cube video curation WebUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    os.makedirs(CURATED_DIR, exist_ok=True)

    print(f"[INFO] Web data dir : {WEB_DATA_DIR}")
    print(f"[INFO] Curated dir  : {CURATED_DIR}")
    print(f"[INFO] Curate log   : {CURATE_LOG}")
    print(f"[INFO] Launching on  {args.host}:{args.port}")
    print(f"[INFO] Remote access: ssh -L {args.port}:localhost:{args.port} user@<server>  →  http://localhost:{args.port}")

    build_ui().launch(server_name=args.host, server_port=args.port, share=False, allowed_paths=[WEB_DATA_DIR])


if __name__ == "__main__":
    main()
