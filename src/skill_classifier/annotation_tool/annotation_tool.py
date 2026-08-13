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
from skill_classifier.action_semantics import load_action_semantics


ACTION_SEMANTICS = load_action_semantics(
    Path(__file__).resolve().parents[1] / "config/kitchen_action_semantics.yaml"
)
ACTION_DESCRIPTIONS = {
    label: ACTION_SEMANTICS["actions"][label]["en"] for label in ACTION_LABELS
}
VIDEO_SUFFIXES = (".mov", ".mp4", ".avi", ".mkv")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
LABEL_PROFILES = {
    "kitchen_milk": {
        "labels": ["Cup", "Lock", "Milk", "Snack", "Sweep", "Trans"],
        "descriptions": {
            "Cup": "Hang the cup on the cup holder",
            "Lock": "Stack the food containers",
            "Milk": "Throw away the milk carton",
            "Snack": "Throw away the snack container",
            "Sweep": "Wipe the floor with the sponge",
            "Trans": "Transition between actions",
        },
    },
    "kitchen_choco": {
        "labels": list(ACTION_LABELS),
        "descriptions": dict(ACTION_DESCRIPTIONS),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_episodes(data_dir):
    data_dir = Path(data_dir)
    episodes = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        rgb_path = d / "rgb"
        frames = []
        direct_video = None
        if rgb_path.is_dir():
            frames = sorted(
                path for path in rgb_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not frames:
                continue
            num_frames = len(frames)
        elif rgb_path.is_file() and rgb_path.resolve().suffix.lower() in VIDEO_SUFFIXES:
            direct_video = str(rgb_path.resolve())
            num_frames = get_video_info(direct_video)["frames"]
        else:
            continue
        episodes.append({
            "name": d.name,
            "path": str(d),
            "rgb_dir": str(rgb_path) if rgb_path.is_dir() else None,
            "direct_video": direct_video,
            "num_frames": num_frames,
            "frame_names": [f.stem for f in frames],
        })
    return episodes


def get_video_info(video_path):
    """Return validated FPS, frame count, and dimensions for a preview video."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open preview video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"invalid preview metadata: fps={fps}, frames={frames}, "
            f"size={width}x{height}"
        )
    return {"fps": fps, "frames": frames, "width": width, "height": height}


def get_video_fps(video_path):
    return get_video_info(video_path)["fps"]


def build_rgb_video(rgb_dir, frame_names, fps, out_path, rotation="ccw"):
    """Build an H.264 preview from raw RGB frames with an explicit rotation."""
    video_filters = {
        "none": "",
        "ccw": "-vf transpose=2 ",
        "cw": "-vf transpose=1 ",
        "180": "-vf hflip,vflip ",
    }
    if rotation not in video_filters:
        raise ValueError(f"unsupported rotation: {rotation}")
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
            f'{video_filters[rotation]}-vcodec libx264 -r {fps} -crf 23 '
            f'-frames:v {len(frame_names)} -pix_fmt yuv420p "{out_path}" '
            f'-loglevel error'
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


def validate_segments_for_save(num_frames, segments, allowed_labels=None):
    """Validate user-provided segment values before writing a GT file."""
    if not isinstance(segments, list):
        raise ValueError("segments must be a list")
    allowed_labels = set(ACTION_LABELS if allowed_labels is None else allowed_labels)
    normalized = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index + 1} is not an object")
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        label = segment.get("label")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError(f"segment {index + 1} frame bounds must be integers")
        if not 0 <= start <= end < num_frames:
            raise ValueError(
                f"segment {index + 1} is out of range: {start}-{end}; "
                f"valid frames are 0-{num_frames - 1}"
            )
        if label not in allowed_labels:
            raise ValueError(f"segment {index + 1} has unknown label: {label!r}")
        normalized.append(
            {"start_frame": start, "end_frame": end, "label": label}
        )
    normalized.sort(key=lambda segment: (segment["start_frame"], segment["end_frame"]))
    for previous, current in zip(normalized, normalized[1:]):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError(
                "overlapping segments: "
                f"{previous['start_frame']}-{previous['end_frame']} and "
                f"{current['start_frame']}-{current['end_frame']}"
            )
    return normalized


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

def make_panel_html(action_labels: list, descriptions=None) -> str:
    descriptions = descriptions or {}
    buttons_html = "\n".join(
        f'<button class="skill-btn" data-skill="{lbl}" '
        f'data-description="{descriptions.get(lbl, lbl)}">'
        f'<b>{lbl}</b><br><small>{descriptions.get(lbl, lbl)}</small></button>'
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
  var _numFrames = 1;
  var _active = null;
  var _startF = null;
  var _videoBound = null;
  var _lastFrame = null;
  var _lastPaused = null;
  var _lastOverlayKey = null;
  window._lbl_segs = [];   // global — read by save_btn js= at click time

  function getVid() {
    if (_videoBound && _videoBound.isConnected) return _videoBound;
    return document.querySelector('#video-player video') || document.querySelector('video');
  }

  function clampFrame(frame) {
    return Math.max(0, Math.min(_numFrames - 1, frame));
  }

  function curFrame() {
    var v = getVid();
    return v ? clampFrame(Math.floor(v.currentTime * _fps)) : 0;
  }

  function stepFrame(delta) {
    var v = getVid();
    if (!v) return;
    v.pause();
    var target = clampFrame(curFrame() + delta);
    v.currentTime = (target + 0.5) / _fps;
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

  function syncPlayBtn(force) {
    var v = getVid();
    var btn = document.getElementById('lbl-playpause-btn');
    if (!btn) return;
    var paused = !v || v.paused;
    if (!force && paused === _lastPaused) return;
    _lastPaused = paused;
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

  function displaySkill(label) {
    var button = Array.from(document.querySelectorAll('.skill-btn')).find(
      function(candidate) { return candidate.dataset.skill === label; }
    );
    return button && button.dataset.description
      ? label + ' \u2014 ' + button.dataset.description
      : label;
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

  function updateOverlay(f, force) {
    if (!_ov) return;
    var labels = getLabelsAtFrame(f);
    var overlayKey = labels.length + ':' + labels.join('|');
    if (!force && overlayKey === _lastOverlayKey) return;
    _lastOverlayKey = overlayKey;
    if (labels.length === 0) {
      _ov.textContent = '\u26A0 No label';
      _ov.style.background = 'rgba(161,98,7,0.88)';
      _ov.style.color = '#fef9c3';
      _ov.style.outline = '2px solid rgba(234,179,8,0.7)';
    } else if (labels.length === 1) {
      _ov.textContent = '\u2713 ' + displaySkill(labels[0]);
      _ov.style.background = 'rgba(21,128,61,0.88)';
      _ov.style.color = '#dcfce7';
      _ov.style.outline = '2px solid rgba(74,222,128,0.7)';
    } else {
      _ov.textContent = '\u26A0 ' + labels.map(displaySkill).join(', ');
      _ov.style.background = 'rgba(185,28,28,0.88)';
      _ov.style.color = '#fee2e2';
      _ov.style.outline = '2px solid rgba(248,113,113,0.7)';
    }
  }

  function renderVideoState(force) {
    var v = getVid();
    if (!v) return;
    var f = curFrame();
    if (!force && f === _lastFrame) return;
    _lastFrame = f;
    var fe = document.getElementById('lbl-frame');
    var te = document.getElementById('lbl-time');
    if (fe) fe.textContent = 'Frame: ' + f;
    if (te) te.textContent = 'Time: ' + v.currentTime.toFixed(2) + 's';
    updateOverlay(f, force);
  }

  function bindVideoElement() {
    var candidate = document.querySelector('#video-player video') || document.querySelector('video');
    if (!candidate || candidate === _videoBound) return;
    _videoBound = candidate;
    _lastFrame = null;
    _lastPaused = null;
    _lastOverlayKey = null;
    initOverlay();
    candidate.addEventListener('play', function() { syncPlayBtn(true); });
    candidate.addEventListener('pause', function() { syncPlayBtn(true); });
    candidate.addEventListener('ended', function() { syncPlayBtn(true); renderVideoState(true); });
    candidate.addEventListener('seeked', function() { renderVideoState(true); });
    candidate.addEventListener('loadeddata', function() { renderVideoState(true); });
    syncPlayBtn(true);
    renderVideoState(true);

    if (typeof candidate.requestVideoFrameCallback === 'function') {
      var onVideoFrame = function() {
        if (_videoBound !== candidate) return;
        renderVideoState(false);
        candidate.requestVideoFrameCallback(onVideoFrame);
      };
      candidate.requestVideoFrameCallback(onVideoFrame);
    } else {
      candidate.addEventListener('timeupdate', function() { renderVideoState(false); });
    }
  }

  // Gradio may replace the <video> element when the episode changes.  Poll
  // only for element replacement; frame UI updates are driven by decoded
  // video frames and do not repaint the video layer on a fixed timer.
  setInterval(bindVideoElement, 500);

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
      el.innerHTML = '&#128308; Recording: <b>' + displaySkill(_active) + '</b> &mdash; started @ frame ' + _startF;
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
    _lastOverlayKey = null;
    var segs = window._lbl_segs;
    if (!segs.length) {
      el.innerHTML = '<span style="color:#555;padding:6px;display:block;">(empty)</span>';
      updateOverlay(curFrame(), true);
      return;
    }
    var labels = getActionLabels();
    var rows = segs.map(function(s, i) {
      var opts = labels.map(function(lbl) {
        return '<option value="' + lbl + '" style="background:#1e1e1e;color:#ffffff;"' + (s.label === lbl ? ' selected' : '') + '>' + displaySkill(lbl) + '</option>';
      }).join('');
      return '<tr style="border-bottom:1px solid #222;">' +
        '<td style="color:#6b7280;padding:3px 5px;text-align:right;">' + (i + 1) + '</td>' +
        '<td style="padding:3px 3px;"><input type="number" min="0" max="' + (_numFrames - 1) + '" value="' + s.start_frame + '" data-si="' + i + '" data-f="start_frame" style="' + INP + 'width:58px;"></td>' +
        '<td style="color:#4b5563;padding:0 3px;">-</td>' +
        '<td style="padding:3px 3px;"><input type="number" min="0" max="' + (_numFrames - 1) + '" value="' + s.end_frame + '" data-si="' + i + '" data-f="end_frame" style="' + INP + 'width:58px;"></td>' +
        '<td style="padding:3px 3px;"><select data-si="' + i + '" data-f="label" style="' + SEL + '">' + opts + '</select></td>' +
        '<td style="padding:3px 3px;"><button data-del="' + i + '" style="' + DEL + '">&#10005;</button></td>' +
        '</tr>';
    }).join('');
    el.innerHTML = '<table style="width:100%;border-collapse:collapse;">' + rows + '</table>';
    el.scrollTop = el.scrollHeight;
    updateOverlay(curFrame(), true);
  }

  // Poll hidden textboxes for Python→JS updates (fps + episode change)
  var _lastSegs = null;
  var _lastFps  = null;
  var _lastNumFrames = null;
  setInterval(function() {
    var fta = document.querySelector('#fps-hidden textarea');
    if (fta && fta.value !== _lastFps) {
      _lastFps = fta.value;
      var v = parseFloat(fta.value);
      if (v > 0) { _fps = v; _lastFrame = null; }
    }
    var nfa = document.querySelector('#num-frames-hidden textarea');
    if (nfa && nfa.value !== _lastNumFrames) {
      _lastNumFrames = nfa.value;
      var n = parseInt(nfa.value);
      if (n > 0) { _numFrames = n; _lastFrame = null; }
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
      if (!isNaN(v)) {
        v = clampFrame(v);
        e.target.value = v;
        window._lbl_segs[idx][ds.f] = v;
      }
    }
    updateOverlay(curFrame(), true);
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
        if (f >= _startF) {
          window._lbl_segs.push({start_frame: _startF, end_frame: f, label: skill});
        }
        _active = null; _startF = null;
        refreshSegs();
      } else {
        if (f > _startF) {
          window._lbl_segs.push({start_frame: _startF, end_frame: f - 1, label: _active});
        }
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
#segments-hidden, #fps-hidden, #num-frames-hidden {
  display: none !important;
}

/* Preserve enough width to inspect both synchronized camera views. */
#video-player {
  width: min(100%, 1280px) !important;
  max-width: 1280px !important;
  margin-left: 0 !important;
  margin-right: auto !important;
}

#video-player video {
  width: 100% !important;
  max-width: 1280px !important;
  max-height: 720px !important;
  object-fit: contain !important;
}
"""


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _raw_video_path(episode_path, rotation):
    # Retain the historical cache filename for the historical CCW default.
    filename = "_raw_video.mp4" if rotation == "ccw" else f"_raw_video_{rotation}.mp4"
    return os.path.join(episode_path, filename)


def _stereo_video_path(episode_path, rotation, num_frames, fps):
    fps_tag = f"{fps:g}".replace(".", "p")
    filename = (
        f"_annotation_stereo_v2_{rotation}_{num_frames}f_{fps_tag}fps.mp4"
    )
    return os.path.join(episode_path, filename)


def _ensure_preview(ep, fps, rotation, view):
    if view == "primary":
        video_path = _raw_video_path(ep["path"], rotation)
        if not os.path.exists(video_path):
            print(f"  Building primary video: {ep['name']} ({ep['num_frames']} frames)...")
            if ep.get("direct_video"):
                filters = {
                    "none": None,
                    "ccw": "transpose=2",
                    "cw": "transpose=1",
                    "180": "hflip,vflip",
                }
                import subprocess
                command = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", ep["direct_video"],
                ]
                if filters[rotation]:
                    command.extend(["-vf", filters[rotation]])
                command.extend([
                    "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "22", "-pix_fmt", "yuv420p", video_path,
                ])
                subprocess.run(command, check=True)
            else:
                build_rgb_video(
                    ep["rgb_dir"], ep["frame_names"], fps,
                    video_path, rotation=rotation
                )
    elif view == "stereo":
        if rotation != "none":
            raise ValueError("stereo preview currently requires --rotation none")
        from data_preprocess.build_stereo_sync_preview import (
            _read_pairs,
            _rgb_paths,
            build_preview,
        )

        episode = Path(ep["path"])
        camera_1_paths = _rgb_paths(episode, 1)
        camera_2_paths = _rgb_paths(episode, 2)
        pairs = _read_pairs(episode / "stereo_pairs.csv")
        counts = (ep["num_frames"], len(camera_1_paths), len(camera_2_paths), len(pairs))
        if len(set(counts)) != 1:
            raise ValueError(
                "stereo preview requires identical frame counts: "
                f"root/camera1/camera2/pairs={counts}"
            )
        video_path = _stereo_video_path(
            ep["path"], rotation, ep["num_frames"], fps
        )
        if not os.path.exists(video_path):
            print(f"  Building stereo video: {ep['name']} ({ep['num_frames']} pairs)...")
            build_preview(
                camera_1_paths,
                camera_2_paths,
                pairs,
                Path(video_path),
                fps,
                panel_width=640,
                panel_height=360,
            )
    else:
        raise ValueError(view)

    info = get_video_info(video_path)
    if info["frames"] != ep["num_frames"]:
        raise ValueError(
            f"preview/frame mismatch: {info['frames']} != {ep['num_frames']} "
            f"({video_path})"
        )
    if abs(info["fps"] - fps) > 0.05:
        raise ValueError(
            f"preview/FPS mismatch: {info['fps']:.6f} != {fps:.6f} "
            f"({video_path})"
        )
    return video_path, info


def build_app(
    data_dir,
    fps,
    rotation="ccw",
    view="primary",
    action_labels=None,
    action_descriptions=None,
):
    action_labels = list(ACTION_LABELS if action_labels is None else action_labels)
    action_descriptions = (
        dict(ACTION_DESCRIPTIONS)
        if action_descriptions is None else dict(action_descriptions)
    )
    episodes = discover_episodes(data_dir)
    if not episodes:
        raise ValueError(f"No episodes found in {data_dir}")

    ep_names = [ep["name"] for ep in episodes]
    ep_map   = {ep["name"]: ep for ep in episodes}

    def on_ep_change(ep_name):
        ep = ep_map[ep_name]
        video_path, video_info = _ensure_preview(ep, fps, rotation, view)
        detected_fps = video_info["fps"]
        gt = load_gt(ep["path"])
        if "num_frames" in gt and int(gt["num_frames"]) != ep["num_frames"]:
            raise ValueError(
                f"existing GT frame count does not match episode: "
                f"{gt['num_frames']} != {ep['num_frames']}"
            )
        if "fps" in gt and abs(float(gt["fps"]) - detected_fps) > 0.05:
            raise ValueError(
                f"existing GT FPS does not match preview: "
                f"{gt['fps']} != {detected_fps}"
            )
        segs_json = json.dumps(gt["segments"])
        info = (
            f"{ep['num_frames']} common frames  |  FPS: {detected_fps:.1f}"
            f"  |  view: {view}"
        )
        if gt["segments"]:
            info += f"  |  {len(gt['segments'])} segments saved"
        return video_path, segs_json, str(detected_fps), str(ep["num_frames"]), info

    def on_save(ep_name, segs_json):
        ep = ep_map[ep_name]
        try:
            segments = json.loads(segs_json or "[]")
            segments = validate_segments_for_save(
                ep["num_frames"], segments, allowed_labels=action_labels
            )
        except Exception as e:
            return f"❌ Validation error: {e}"
        video_path, video_info = _ensure_preview(ep, fps, rotation, view)
        detected_fps = video_info["fps"]
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
            video_comp = gr.Video(
                label="Synchronized Stereo Video" if view == "stereo" else "Raw RGB Video",
                elem_id="video-player",
                interactive=False,
            )

        with gr.Row():
            gr.HTML(make_panel_html(action_labels, action_descriptions))

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
        num_frames_hidden = gr.Textbox(
            value="1", elem_id="num-frames-hidden",
        )

        ep_dropdown.change(
            on_ep_change,
            inputs=[ep_dropdown],
            outputs=[
                video_comp,
                segments_hidden,
                fps_hidden,
                num_frames_hidden,
                ep_info,
            ],
        )
        save_btn.click(
            on_save,
            inputs=[ep_dropdown, segments_hidden],
            outputs=[status_out],
            js=SAVE_JS,
        )
        app.load(
            lambda: on_ep_change(ep_names[0]),
            outputs=[
                video_comp,
                segments_hidden,
                fps_hidden,
                num_frames_hidden,
                ep_info,
            ],
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
    parser.add_argument(
        "--rotation",
        choices=["none", "ccw", "cw", "180"],
        default="ccw",
        help="Preview rotation applied while building the cached video (default: ccw).",
    )
    parser.add_argument(
        "--view",
        choices=["primary", "stereo"],
        default="primary",
        help="Show the root RGB stream or a synchronized camera_1/camera_2 composite.",
    )
    parser.add_argument(
        "--label-profile",
        choices=sorted(LABEL_PROFILES),
        default="kitchen_milk",
        help="Label vocabulary shown and accepted by the annotation tool.",
    )
    args = parser.parse_args()

    profile = LABEL_PROFILES[args.label_profile]
    print(f"Labeling tool for: {args.data_dir}")
    print(f"Label profile: {args.label_profile} ({', '.join(profile['labels'])})")
    print(f"SSH tunnel : ssh -L {args.port}:localhost:{args.port} <server>")
    print(f"Browser    : http://localhost:{args.port}")

    app = build_app(
        args.data_dir,
        args.fps,
        rotation=args.rotation,
        view=args.view,
        action_labels=profile["labels"],
        action_descriptions=profile["descriptions"],
    )
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
