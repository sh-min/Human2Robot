"""
Web-based GT labeling tool for long-horizon skill sequences.

Usage:
    conda activate vjepa2-312
    cd /virtual_lab/ljw_rvlab/byeonggyeol/3dgs-visual-grounding/RFM_Proj
    python -m skill_segmentor.annotation_tool --data_dir data/cube_dataset/0325 --port 7860

    # SSH tunnel from local machine:
    ssh -L 7860:localhost:7860 user@server
    # Then open http://localhost:7860

    # FPS is auto-detected from the built video (default 15 if not yet built).

Output:
    {data_dir}/{episode_name}/gt_labels.json
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from utils.labels import ACTION_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_episodes(data_dir):
    data_dir = Path(data_dir)
    episodes = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        rgb_dir = d / "rgb"
        if not rgb_dir.is_dir():
            continue
        frames = sorted(rgb_dir.glob("*.png"))
        if not frames:
            continue
        episodes.append({
            "name": d.name,
            "path": str(d),
            "rgb_dir": str(rgb_dir),
            "num_frames": len(frames),
            "frame_names": [f.stem for f in frames],
        })
    return episodes


def get_video_fps(video_path):
    """Extract FPS from an existing video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 15.0


def build_rgb_video(rgb_dir, frame_names, fps, out_path):
    """Build a left-rotated H.264 video from raw RGB frames. Cached as _raw_video.mp4."""
    import tempfile
    list_fd, list_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(list_fd, "w") as f:
            duration = 1.0 / fps
            for fn in frame_names:
                abs_path = os.path.abspath(os.path.join(rgb_dir, f"{fn}.png"))
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {duration}\n")
        ret = os.system(
            f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
            f'-vf transpose=2 -vcodec libx264 -r {fps} -crf 23 '
            f'-pix_fmt yuv420p "{out_path}" -loglevel error'
        )
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (exit {ret}) when building {out_path}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)


def load_gt(episode_path):
    gt_path = os.path.join(episode_path, "gt_labels.json")
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            return json.load(f)
    return {"segments": []}


def frames_to_ranges(frames):
    """Convert a sorted list of frame indices to compact range strings."""
    if not frames:
        return ""
    parts = []
    start = end = frames[0]
    for f in frames[1:]:
        if f == end + 1:
            end = f
        else:
            parts.append(str(start) if start == end else f"{start}-{end}")
            start = end = f
    parts.append(str(start) if start == end else f"{start}-{end}")
    # Truncate if too many ranges
    if len(parts) > 10:
        parts = parts[:10] + [f"... (+{len(parts) - 10} more)"]
    return ", ".join(parts)


def validate_segments(num_frames, segments):
    """Return (unlabeled_ranges_str, overlap_ranges_str) or empty strings."""
    coverage = [0] * num_frames
    for seg in segments:
        s = max(0, int(seg["start_frame"]))
        e = min(num_frames - 1, int(seg["end_frame"]))
        for f in range(s, e + 1):
            coverage[f] += 1
    unlabeled   = [f for f in range(num_frames) if coverage[f] == 0]
    overlapping = [f for f in range(num_frames) if coverage[f] > 1]
    return frames_to_ranges(unlabeled), frames_to_ranges(overlapping), len(unlabeled), len(overlapping)


def save_gt(episode_path, episode_name, num_frames, fps, segments):
    gt_path = os.path.join(episode_path, "gt_labels.json")
    with open(gt_path, "w") as f:
        json.dump({
            "episode": episode_name,
            "num_frames": num_frames,
            "fps": fps,
            "segments": segments,
        }, f, indent=2)
    return gt_path


# ---------------------------------------------------------------------------
# HTML labeling panel  (no <script>, no onclick — JS injected via gr.Blocks)
# ---------------------------------------------------------------------------

def make_panel_html(action_labels: list) -> str:
    buttons_html = "\n".join(
        f'<button class="skill-btn" data-skill="{lbl}">{lbl}</button>'
        for lbl in action_labels
    )
    return f"""
<div id="lbl-root" style="font-family:monospace; padding:4px;">

  <!-- Frame counter (large, prominent) -->
  <div style="padding:8px 12px;background:#1e1e1e;border-radius:6px;margin-bottom:8px;">
    <span id="lbl-frame" style="color:#facc15;font-size:18px;font-weight:bold;">Frame: --</span>
    <span id="lbl-time"  style="color:#aaa;font-size:12px;margin-left:12px;">Time: --</span>
  </div>

  <!-- Video controls -->
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
    <button id="lbl-prev-btn"
            style="background:#374151;color:#fff;border:none;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:14px;"
            title="Previous frame (Left arrow)">
      &#9664;&#9664;
    </button>
    <button id="lbl-playpause-btn"
            style="background:#374151;color:#fff;border:none;padding:6px 16px;border-radius:4px;cursor:pointer;font-size:14px;min-width:64px;">
      &#9654; Play
    </button>
    <button id="lbl-next-btn"
            style="background:#374151;color:#fff;border:none;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:14px;"
            title="Next frame (Right arrow)">
      &#9654;&#9654;
    </button>
    <button id="lbl-slower-btn"
            style="background:#374151;color:#fff;border:none;padding:6px 10px;border-radius:4px;cursor:pointer;font-size:13px;"
            title="Slower (PageDown)">
      &#9660;
    </button>
    <span id="lbl-speed" style="color:#e5e7eb;font-size:13px;font-family:monospace;min-width:38px;text-align:center;background:#1f2937;border-radius:3px;padding:2px 4px;">1.0x</span>
    <button id="lbl-faster-btn"
            style="background:#374151;color:#fff;border:none;padding:6px 10px;border-radius:4px;cursor:pointer;font-size:13px;"
            title="Faster (PageUp)">
      &#9650;
    </button>
    <span style="color:#555;font-size:11px;margin-left:4px;">&#8592;&#8594; frame &nbsp; PgUp/Dn speed</span>
  </div>

  <!-- Recording status -->
  <div id="lbl-status"
       style="padding:8px;background:#1a1a2e;color:#ddd;border-radius:6px;margin-bottom:8px;font-size:13px;">
    &#9898; Not recording
  </div>

  <!-- Instructions -->
  <div style="color:#888;font-size:11px;margin-bottom:8px;line-height:1.5;">
    &#9654; Click skill &#8594; start &nbsp;|&nbsp; same button &#8594; end &nbsp;|&nbsp; other button &#8594; switch
  </div>

  <!-- Skill buttons -->
  <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;">
    {buttons_html}
  </div>

  <!-- Undo / Clear -->
  <div style="display:flex;gap:6px;margin-bottom:10px;">
    <button id="lbl-undo-btn"
            style="background:#78350f;color:#fff;border:none;padding:0;border-radius:4px;cursor:pointer;font-size:12px;font-family:monospace;line-height:1;display:inline-flex;align-items:center;justify-content:center;width:72px;height:30px;box-sizing:border-box;">
      Undo
    </button>
    <button id="lbl-clear-btn"
            style="background:#7f1d1d;color:#fff;border:none;padding:0;border-radius:4px;cursor:pointer;font-size:12px;font-family:monospace;line-height:1;display:inline-flex;align-items:center;justify-content:center;width:72px;height:30px;box-sizing:border-box;">
      Clear
    </button>
  </div>

  <!-- Segments list (editable) -->
  <div style="color:#aaa;font-size:12px;margin-bottom:4px;">Segments:</div>
  <div id="lbl-segs"
       style="background:#111;border-radius:4px;font-size:11px;
              max-height:600px;overflow-y:auto;padding:4px;">
    <span style="color:#555;">(empty)</span>
  </div>

</div>

<style>
.skill-btn {{
  background: #2563eb; color: #fff; border: none;
  padding: 7px 11px; border-radius: 5px; cursor: pointer;
  font-size: 12px; font-family: monospace;
  transition: background 0.15s;
}}
.skill-btn:hover  {{ background: #1d4ed8; }}
.skill-btn.active {{
  background: #ef4444 !important;
  background-color: #ef4444 !important;
  outline: 3px solid #fca5a5;
  box-shadow: 0 0 10px 2px rgba(239,68,68,0.7);
  color: #fff !important;
  font-weight: bold;
}}
#lbl-prev-btn:hover, #lbl-next-btn:hover, #lbl-playpause-btn:hover {{ background: #4b5563; }}
</style>
"""


# JS injected via gr.Blocks(js=...) — must be IIFE so it executes immediately.
# Segments are stored in window._lbl_segs (global) so save_btn js= can read them directly,
# bypassing the unreliable Gradio hidden-textbox sync.
PANEL_JS = """
(function() {
  var _fps = 15;
  var _active = null;
  var _startF = null;
  window._lbl_segs = [];   // global — read by save_btn js= at click time

  function getVid() {
    return document.querySelector('#video-player video') || document.querySelector('video');
  }

  function curFrame() {
    var v = getVid();
    return v ? Math.floor(v.currentTime * _fps) : 0;
  }

  function stepFrame(delta) {
    var v = getVid();
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, v.currentTime + delta / _fps);
  }

  var _speed = 1.0;

  function setSpeed(s) {
    _speed = Math.round(Math.min(1.0, Math.max(0.1, s)) * 10) / 10;
    var v = getVid();
    if (v) v.playbackRate = _speed;
    var el = document.getElementById('lbl-speed');
    if (el) el.textContent = _speed.toFixed(1) + 'x';
  }

  function togglePlay() {
    var v = getVid();
    if (!v) return;
    if (v.paused) { v.playbackRate = _speed; v.play(); } else { v.pause(); }
  }

  function syncPlayBtn() {
    var v = getVid();
    var btn = document.getElementById('lbl-playpause-btn');
    if (!btn) return;
    btn.innerHTML = (v && !v.paused) ? '&#9646;&#9646; Pause' : '&#9654; Play';
  }

  var _ov = null;  // cached overlay element

  function getLabelsAtFrame(f) {
    var segs = window._lbl_segs || [];
    var found = [];
    for (var i = 0; i < segs.length; i++) {
      if (f >= segs[i].start_frame && f <= segs[i].end_frame) found.push(segs[i].label);
    }
    return found;
  }

  function initOverlay() {
    var container = document.querySelector('#video-player');
    if (!container) return;
    var existing = document.getElementById('lbl-overlay');
    if (existing) { _ov = existing; return; }
    container.style.position = 'relative';
    _ov = document.createElement('div');
    _ov.id = 'lbl-overlay';
    _ov.style.cssText = [
      'position:absolute', 'top:36px', 'left:10px', 'z-index:999',
      'padding:10px 20px', 'border-radius:8px',
      'font-family:monospace', 'font-size:22px', 'font-weight:bold',
      'pointer-events:none',
    ].join(';');
    container.appendChild(_ov);
  }

  function updateOverlay(f) {
    if (!_ov) return;
    var labels = getLabelsAtFrame(f);
    if (labels.length === 0) {
      _ov.textContent = '\u26A0 No label';
      _ov.style.background = 'rgba(161,98,7,0.88)';
      _ov.style.color = '#fef9c3';
      _ov.style.outline = '2px solid rgba(234,179,8,0.7)';
    } else if (labels.length === 1) {
      _ov.textContent = '\u2713 ' + labels[0];
      _ov.style.background = 'rgba(21,128,61,0.88)';
      _ov.style.color = '#dcfce7';
      _ov.style.outline = '2px solid rgba(74,222,128,0.7)';
    } else {
      _ov.textContent = '\u26A0 ' + labels.join(', ');
      _ov.style.background = 'rgba(185,28,28,0.88)';
      _ov.style.color = '#fee2e2';
      _ov.style.outline = '2px solid rgba(248,113,113,0.7)';
    }
  }

  // Try to init overlay once video-player is in DOM (retry until found)
  var _overlayInitInterval = setInterval(function() {
    initOverlay();
    if (_ov) clearInterval(_overlayInitInterval);
  }, 500);

  // Live frame/time counter + play button sync + overlay
  setInterval(function() {
    var v = getVid();
    if (!v) return;
    var f = Math.floor(v.currentTime * _fps);
    var fe = document.getElementById('lbl-frame');
    var te = document.getElementById('lbl-time');
    if (fe) fe.textContent = 'Frame: ' + f;
    if (te) te.textContent = 'Time: ' + v.currentTime.toFixed(2) + 's';
    syncPlayBtn();
    updateOverlay(f);
  }, 100);

  // Keyboard shortcuts: Space=play/pause, Left/Right=step frame
  document.addEventListener('keydown', function(e) {
    var tag = (e.target || e.srcElement).tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.code === 'Space')      { e.preventDefault(); togglePlay(); }
    if (e.code === 'ArrowLeft')  { e.preventDefault(); stepFrame(-1); }
    if (e.code === 'ArrowRight') { e.preventDefault(); stepFrame(+1); }
    if (e.code === 'PageUp')     { e.preventDefault(); setSpeed(_speed + 0.1); }
    if (e.code === 'PageDown')   { e.preventDefault(); setSpeed(_speed - 0.1); }
  });

  function refreshStatus() {
    var el = document.getElementById('lbl-status');
    if (!el) return;
    if (_active) {
      el.innerHTML = '&#128308; Recording: <b>' + _active + '</b> &mdash; started @ frame ' + _startF;
      el.style.background = '#450a0a';
    } else {
      el.innerHTML = '&#9898; Not recording';
      el.style.background = '#1a1a2e';
    }
    document.querySelectorAll('.skill-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.skill === _active);
    });
  }

  function getActionLabels() {
    return Array.from(document.querySelectorAll('.skill-btn')).map(function(b) { return b.dataset.skill; });
  }

  var INP = 'background:#1a1a1a;color:#e5e7eb;border:1px solid #374151;border-radius:3px;padding:2px 4px;font-family:monospace;font-size:11px;';
  var SEL = 'background:#1a1a1a;color:#e5e7eb;border:1px solid #374151;border-radius:3px;padding:2px 3px;font-size:11px;';
  var DEL = 'background:#7f1d1d;color:#fca5a5;border:none;border-radius:3px;padding:1px 6px;cursor:pointer;font-size:11px;';

  function refreshSegs() {
    var el = document.getElementById('lbl-segs');
    if (!el) return;
    var segs = window._lbl_segs;
    if (!segs.length) {
      el.innerHTML = '<span style="color:#555;padding:6px;display:block;">(empty)</span>';
      return;
    }
    var labels = getActionLabels();
    var rows = segs.map(function(s, i) {
      var opts = labels.map(function(lbl) {
        return '<option value="' + lbl + '" style="background:#1e1e1e;color:#ffffff;"' + (s.label === lbl ? ' selected' : '') + '>' + lbl + '</option>';
      }).join('');
      return '<tr style="border-bottom:1px solid #222;">' +
        '<td style="color:#6b7280;padding:3px 5px;text-align:right;">' + (i + 1) + '</td>' +
        '<td style="padding:3px 3px;"><input type="number" min="0" value="' + s.start_frame + '" data-si="' + i + '" data-f="start_frame" style="' + INP + 'width:58px;"></td>' +
        '<td style="color:#4b5563;padding:0 3px;">-</td>' +
        '<td style="padding:3px 3px;"><input type="number" min="0" value="' + s.end_frame + '" data-si="' + i + '" data-f="end_frame" style="' + INP + 'width:58px;"></td>' +
        '<td style="padding:3px 3px;"><select data-si="' + i + '" data-f="label" style="' + SEL + '">' + opts + '</select></td>' +
        '<td style="padding:3px 3px;"><button data-del="' + i + '" style="' + DEL + '">&#10005;</button></td>' +
        '</tr>';
    }).join('');
    el.innerHTML = '<table style="width:100%;border-collapse:collapse;">' + rows + '</table>';
    el.scrollTop = el.scrollHeight;
  }

  // Poll hidden textboxes for Python→JS updates (fps + episode change)
  var _lastSegs = null;
  var _lastFps  = null;
  setInterval(function() {
    var fta = document.querySelector('#fps-hidden textarea');
    if (fta && fta.value !== _lastFps) {
      _lastFps = fta.value;
      var v = parseFloat(fta.value);
      if (v > 0) _fps = v;
    }
    var ta = document.querySelector('#segments-hidden textarea');
    if (!ta) return;
    if (ta.value === _lastSegs) return;
    _lastSegs = ta.value;
    try { window._lbl_segs = JSON.parse(ta.value || '[]'); } catch(e) { window._lbl_segs = []; }
    _active = null; _startF = null;
    refreshSegs(); refreshStatus();
  }, 300);

  // Edit segment fields (number inputs + select) — update window._lbl_segs in place,
  // no re-render needed (DOM already shows new value)
  function onSegFieldChange(e) {
    var ds = e.target.dataset;
    if (!ds || ds.si === undefined || !ds.f) return;
    var idx = parseInt(ds.si);
    if (isNaN(idx) || idx >= window._lbl_segs.length) return;
    if (ds.f === 'label') {
      window._lbl_segs[idx].label = e.target.value;
    } else {
      var v = parseInt(e.target.value);
      if (!isNaN(v)) window._lbl_segs[idx][ds.f] = v;
    }
  }
  document.addEventListener('input',  onSegFieldChange);
  document.addEventListener('change', onSegFieldChange);

  // Click delegation
  document.addEventListener('click', function(e) {
    if (e.target.id === 'lbl-playpause-btn') { togglePlay(); return; }
    if (e.target.id === 'lbl-prev-btn')      { stepFrame(-1); return; }
    if (e.target.id === 'lbl-next-btn')      { stepFrame(+1); return; }
    if (e.target.id === 'lbl-slower-btn')    { setSpeed(_speed - 0.1); return; }
    if (e.target.id === 'lbl-faster-btn')    { setSpeed(_speed + 0.1); return; }

    // Delete segment row
    if (e.target.dataset && e.target.dataset.del !== undefined) {
      window._lbl_segs.splice(parseInt(e.target.dataset.del), 1);
      refreshSegs(); refreshStatus();
      return;
    }

    var btn = e.target.closest ? e.target.closest('.skill-btn') : null;
    if (btn) {
      var skill = btn.dataset.skill;
      var f = curFrame();
      if (_active === null) {
        _active = skill; _startF = f;
      } else if (_active === skill) {
        window._lbl_segs.push({start_frame: _startF, end_frame: f, label: skill});
        _active = null; _startF = null;
        refreshSegs();
      } else {
        window._lbl_segs.push({start_frame: _startF, end_frame: f - 1, label: _active});
        _active = skill; _startF = f;
        refreshSegs();
      }
      refreshStatus();
      return;
    }
    if (e.target.id === 'lbl-undo-btn') {
      if (_active !== null) {
        _active = null; _startF = null;
      } else if (window._lbl_segs.length > 0) {
        window._lbl_segs.pop(); refreshSegs();
      }
      refreshStatus();
    }
    if (e.target.id === 'lbl-clear-btn') {
      window._lbl_segs = []; _active = null; _startF = null;
      refreshSegs(); refreshStatus();
    }
  });
})();
"""

# JS run at save-button click time: reads window._lbl_segs directly (bypasses hidden textbox sync)
SAVE_JS = "(ep_name, _segs_ignored) => [ep_name, JSON.stringify(window._lbl_segs || [])]"
APP_CSS = """
#segments-hidden, #fps-hidden {
  display: none !important;
}

/* Keep the labeling video compact and aligned to the left. */
#video-player {
  width: min(100%, 640px) !important;
  max-width: 640px !important;
  margin-left: 0 !important;
  margin-right: auto !important;
}

#video-player video {
  width: 100% !important;
  max-width: 640px !important;
  max-height: 360px !important;
  object-fit: contain !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app(data_dir, fps):
    episodes = discover_episodes(data_dir)
    if not episodes:
        raise ValueError(f"No episodes found in {data_dir}")

    ep_names = [ep["name"] for ep in episodes]
    ep_map   = {ep["name"]: ep for ep in episodes}

    def on_ep_change(ep_name):
        ep = ep_map[ep_name]
        video_path = os.path.join(ep["path"], "_raw_video.mp4")
        if not os.path.exists(video_path):
            print(f"  Building video: {ep_name} ({ep['num_frames']} frames)...")
            build_rgb_video(ep["rgb_dir"], ep["frame_names"], fps, video_path)
            print(f"  Done: {video_path}")
        detected_fps = get_video_fps(video_path)
        gt = load_gt(ep["path"])
        segs_json = json.dumps(gt["segments"])
        info = f"{ep['num_frames']} frames  |  FPS: {detected_fps:.1f}"
        if gt["segments"]:
            info += f"  |  {len(gt['segments'])} segments saved"
        return video_path, segs_json, str(detected_fps), info

    def on_save(ep_name, segs_json):
        ep = ep_map[ep_name]
        try:
            segments = json.loads(segs_json or "[]")
        except Exception as e:
            return f"❌ Parse error: {e}"
        video_path = os.path.join(ep["path"], "_raw_video.mp4")
        detected_fps = get_video_fps(video_path) if os.path.exists(video_path) else fps
        gt_path = save_gt(ep["path"], ep_name, ep["num_frames"], detected_fps, segments)

        msg = f"✅ Saved {len(segments)} segments → {gt_path}"

        unlabeled_str, overlap_str, n_unlabeled, n_overlap = validate_segments(
            ep["num_frames"], segments
        )
        if unlabeled_str:
            msg += f"\n⚠ Unlabeled ({n_unlabeled} frames): {unlabeled_str}"
        if overlap_str:
            msg += f"\n⚠ Overlap ({n_overlap} frames): {overlap_str}"
        if not unlabeled_str and not overlap_str:
            msg += "\n✓ All frames labeled exactly once."

        return msg

    with gr.Blocks(title="Skill GT Labeling") as app:

        gr.Markdown("## Skill GT Labeling Tool")

        with gr.Row():
            ep_dropdown = gr.Dropdown(
                choices=ep_names, value=ep_names[0],
                label="Episode", scale=4,
            )
            ep_info = gr.Textbox(label="Info", interactive=False, scale=2)

        with gr.Row():
            with gr.Column(scale=3):
                video_comp = gr.Video(
                    label="Raw RGB Video",
                    elem_id="video-player",
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.HTML(make_panel_html(ACTION_LABELS))

        with gr.Row():
            save_btn   = gr.Button("Save GT Labels", variant="primary", scale=1)
            status_out = gr.Textbox(label="Status", interactive=False, scale=3, lines=4)

        # Bridges: segments (JS↔Python) and fps (Python→JS)
        # visible=True so Gradio syncs DOM value; hidden via CSS in gr.Blocks(css=...)
        segments_hidden = gr.Textbox(
            value="[]", elem_id="segments-hidden",
        )
        fps_hidden = gr.Textbox(
            value="15", elem_id="fps-hidden",
        )

        ep_dropdown.change(
            on_ep_change,
            inputs=[ep_dropdown],
            outputs=[video_comp, segments_hidden, fps_hidden, ep_info],
        )
        save_btn.click(
            on_save,
            inputs=[ep_dropdown, segments_hidden],
            outputs=[status_out],
            js=SAVE_JS,
        )
        app.load(
            lambda: on_ep_change(ep_names[0]),
            outputs=[video_comp, segments_hidden, fps_hidden, ep_info],
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="e.g. data/cube_dataset/0325")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps",  type=float, default=30.0,
                        help="FPS used only when building _raw_video.mp4 for the first time. "
                             "After that, FPS is auto-detected from the video file.")
    args = parser.parse_args()

    print(f"Labeling tool for: {args.data_dir}")
    print(f"SSH tunnel : ssh -L {args.port}:localhost:{args.port} <server>")
    print(f"Browser    : http://localhost:{args.port}")

    app = build_app(args.data_dir, args.fps)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        allowed_paths=[str(Path(args.data_dir).resolve())],
        js=PANEL_JS,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
