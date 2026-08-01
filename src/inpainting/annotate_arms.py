"""Stage 6 (interactive): SAM2 *video* annotation → masks_arm.npy.

This is the DEFAULT Stage-6 segmenter for the layered pipeline (run_layered.py):
instead of auto-seeding SAM2 from HaWoR hand boxes, a human annotates a few
frames with positive/negative point prompts, SAM2 propagates one consistent
mask across the whole clip, and the result is saved as masks_arm.npy.

It runs a small gradio app and BLOCKS until you click "Save & finish", at which
point masks_arm.npy is written and the process exits 0 so run_layered.py resumes.

Workflow:
  1. slide to a frame, click positive (green) / negative (red) points
     (optional "Load HaWoR auto-prompts" seeds a frame from the hand keypoints)
  2. repeat on a few frames — each frame's points are remembered
  3. "Propagate through video" -> SAM2 video predictor forward+reverse
  4. play/scrub the preview, then "Save & finish" to write the mask and continue

Usage:
    CUDA_VISIBLE_DEVICES=5 python annotate_arms.py \
        --processed_demo /result/skill2policy/processed/OCC/0 \
        --host 127.0.0.1 --port 7860
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr

# guarded gradio_client schema shim (no-op on fixed versions)
try:
    import gradio_client.utils as _gcu
    if hasattr(_gcu, "_json_schema_to_python_type"):
        _oj = _gcu._json_schema_to_python_type
        _gcu._json_schema_to_python_type = lambda s, d=None: "bool" if isinstance(s, bool) else _oj(s, d)
    if hasattr(_gcu, "get_type"):
        _ot = _gcu.get_type
        _gcu.get_type = lambda s: "bool" if isinstance(s, bool) else _ot(s)
except Exception as _e:
    print("[annotate] gcu shim skipped:", _e)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _paths import SAM2_CHECKPOINT, SAM2_CONFIG_NAME, ensure_sam2_importable
ensure_sam2_importable()
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import segment_arms as SA

FOREARM_SCALES = (0.75, 1.5, 2.5, 4.0, 6.0, 8.0)
NEG_SCALES = (0.6, 1.2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- module state, populated in main() from --processed_demo ----
FRAMES = []
FRAMES_DIR = None
MASK_PATH = None
PREVIEW_MP4 = None
KPTS_L = None          # (T,21,2) or None
DET_L = None           # (T,) or None
_H = _W = None
IPRED = None
VPRED = None
_cur = {"idx": -1}
PROP = {"masks": None}          # propagated arm mask (T,H,W) bool
PROP_OBJ = {"masks": None}      # propagated object mask from negative prompts


def _frame_rgb(idx):
    return cv2.cvtColor(cv2.imread(FRAMES[int(idx)]), cv2.COLOR_BGR2RGB)


def _iset(idx, rgb):
    if _cur["idx"] != idx:
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            IPRED.set_image(rgb)
        _cur["idx"] = idx


def _ipredict(idx, rgb, pts):
    if not pts:
        return None
    _iset(idx, rgb)
    pc = np.array([[p[0], p[1]] for p in pts], np.float32)
    pl = np.array([p[2] for p in pts], np.int32)
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        masks, scores, _ = IPRED.predict(point_coords=pc, point_labels=pl,
                                         multimask_output=(len(pts) == 1))
    return masks[int(np.argmax(scores))].astype(bool)


def _render(idx, pts, preview_prop):
    rgb = _frame_rgb(idx)
    out = rgb.copy()
    if preview_prop and PROP["masks"] is not None:
        m = PROP["masks"][int(idx)].astype(bool)
        tint = out.copy(); tint[m] = (0, 220, 0)
        out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 255, 0), 2)
    else:
        m = _ipredict(idx, rgb, pts)
        if m is not None:
            tint = out.copy(); tint[m] = (0, 90, 255)
            out = cv2.addWeighted(tint, 0.4, out, 0.6, 0)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, (255, 255, 0), 2)
    for (x, y, lab) in pts:
        c = (0, 220, 0) if lab == 1 else (255, 40, 40)
        cv2.circle(out, (int(x), int(y)), 9, (255, 255, 255), -1)
        cv2.circle(out, (int(x), int(y)), 7, c, -1)
    return out


def _anno_summary(annos):
    items = []
    for f in sorted(annos):
        p = annos[f]
        if not p:
            continue
        npos = sum(1 for q in p if q[2] == 1)
        items.append(f"f{f} ({npos}+/{len(p)-npos}-)")
    return "**annotated frames:** " + (", ".join(items) if items else "none yet")


def _has_hand():
    return KPTS_L is not None and DET_L is not None


def load_frame(idx, annos, preview_prop):
    idx = int(idx)
    pts = list(annos.get(idx, []))
    det = bool(DET_L[idx]) if (_has_hand() and idx < len(DET_L)) else "n/a"
    return _render(idx, pts, preview_prop), f"frame {idx}  (HaWoR det={det})", _anno_summary(annos)


def click(idx, annos, label_mode, preview_prop, evt: gr.SelectData):
    idx = int(idx)
    if preview_prop and PROP["masks"] is not None:
        return annos, _render(idx, list(annos.get(idx, [])), True), "preview mode — uncheck 'preview propagated' to edit points"
    annos = dict(annos)
    pts = list(annos.get(idx, []))
    pts.append((float(evt.index[0]), float(evt.index[1]), 1 if label_mode == "positive" else 0))
    annos[idx] = pts
    return annos, _render(idx, pts, False), _anno_summary(annos)


def undo(idx, annos, preview_prop):
    idx = int(idx); annos = dict(annos)
    pts = list(annos.get(idx, []))[:-1]
    if pts:
        annos[idx] = pts
    else:
        annos.pop(idx, None)
    return annos, _render(idx, pts, preview_prop), _anno_summary(annos)


def clear_frame(idx, annos, preview_prop):
    idx = int(idx); annos = dict(annos); annos.pop(idx, None)
    return annos, _render(idx, [], preview_prop), _anno_summary(annos)


def clear_all(idx, preview_prop):
    return {}, _render(int(idx), [], preview_prop), _anno_summary({})


def load_auto(idx, annos, preview_prop):
    idx = int(idx)
    if not _has_hand() or idx >= len(DET_L) or not DET_L[idx]:
        return annos, _render(idx, list(annos.get(idx, [])), preview_prop), f"frame {idx}: no HaWoR detection to seed"
    kp = KPTS_L[idx:idx + 1]
    pos = SA._augment_with_forearm_points(kp, _H, _W, FOREARM_SCALES)[0]
    neg = SA._object_negative_points(kp, _H, _W, NEG_SCALES)[0]
    pts = [(float(x), float(y), 1) for (x, y) in pos] + [(float(x), float(y), 0) for (x, y) in neg]
    annos = dict(annos); annos[idx] = pts
    return annos, _render(idx, pts, False), _anno_summary(annos) + "  |  seeded auto-prompts here"


def _render_prop_video(masks):
    os.makedirs(os.path.dirname(PREVIEW_MP4), exist_ok=True)
    T = len(FRAMES)
    ow = 960; oh = int(960 * _H / _W) // 2 * 2
    tmp = PREVIEW_MP4 + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 20, (ow, oh))
    obj = PROP_OBJ["masks"]
    for t in range(T):
        rgb = _frame_rgb(t)
        m = masks[t].astype(bool)
        over = rgb.copy(); over[m] = (0, 220, 0)
        bl = cv2.addWeighted(over, 0.45, rgb, 0.55, 0)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bl, cs, -1, (0, 255, 0), 2)          # arm = green
        if obj is not None:
            om = obj[t].astype(bool)
            oc, _ = cv2.findContours(om.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bl, oc, -1, (0, 220, 255), 2)    # object = cyan
        vw.write(cv2.cvtColor(cv2.resize(bl, (ow, oh)), cv2.COLOR_RGB2BGR))
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", PREVIEW_MP4], check=False)
    if os.path.exists(tmp):
        os.remove(tmp)
    return PREVIEW_MP4


def _get_vpred():
    global VPRED
    if VPRED is None:
        VPRED = build_sam2_video_predictor(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)
    return VPRED


def propagate(annos, do_smooth, progress=gr.Progress()):
    frames_with_pts = {int(f): v for f, v in annos.items() if v}
    if not frames_with_pts:
        return "no annotated frames — add points on at least one frame first", gr.update(), gr.update()
    progress(0.05, desc="init video state...")
    vp = _get_vpred()
    T = len(FRAMES)
    masks = np.zeros((T, _H, _W), dtype=bool)
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = vp.init_state(video_path=FRAMES_DIR, offload_video_to_cpu=True)
        vp.reset_state(state)
        for fidx, pts in sorted(frames_with_pts.items()):
            pc = np.array([[p[0], p[1]] for p in pts], np.float32)
            pl = np.array([p[2] for p in pts], np.int32)
            vp.add_new_points_or_box(state, frame_idx=fidx, obj_id=0, points=pc, labels=pl)
        for rev in (False, True):
            progress(0.2 if not rev else 0.6, desc=f"propagate {'reverse' if rev else 'forward'}...")
            for out_idx, _obj, logits in vp.propagate_in_video(state, reverse=rev):
                masks[out_idx] |= (logits[0] > 0.0).cpu().numpy()[0]
    torch.cuda.empty_cache()
    if do_smooth:
        progress(0.9, desc="pipeline denoise...")
        masks, _rep = SA._repair_temporal_mask_outliers(masks)
        masks, _sp = SA._smooth_masks(masks)
    PROP["masks"] = masks
    # Object mask from the NEGATIVE prompts (they sit on the grasped/occluding
    # object): re-seed SAM2 with negatives as object-positives, arm-positives as
    # object-negatives, and propagate. Feeds the contact-occlusion compositor.
    progress(0.85, desc="propagate object from negatives...")
    n_obj = _propagate_object(annos, do_smooth)
    progress(0.95, desc="rendering preview video...")
    vid = _render_prop_video(masks)
    areas = masks.reshape(T, -1).sum(1)
    obj_note = ""
    if PROP_OBJ["masks"] is not None:
        oa = PROP_OBJ["masks"].reshape(T, -1).sum(1)
        obj_note = f"  |  object mask from {n_obj} neg-frame(s): {int((oa>0).sum())}/{T} frames"
    msg = (f"✅ arm from {len(frames_with_pts)} frame(s) {sorted(frames_with_pts)}: "
           f"coverage {int((areas>0).sum())}/{T}, avg {areas.mean():.0f} px"
           f"{', denoised' if do_smooth else ', raw'}.{obj_note} Play/scrub below, then Save & finish.")
    return msg, gr.update(value=True), vid


def _propagate_object(annos, do_smooth):
    """SAM2 object mask seeded from negative prompts (object) vs positive (arm)."""
    obj_frames = {}
    for f, pts in annos.items():
        negs = [(p[0], p[1]) for p in pts if p[2] == 0]
        poss = [(p[0], p[1]) for p in pts if p[2] == 1]
        if negs:
            obj_frames[int(f)] = (negs, poss)
    if not obj_frames:
        PROP_OBJ["masks"] = None
        return 0
    vp = _get_vpred()
    T = len(FRAMES)
    masks = np.zeros((T, _H, _W), dtype=bool)
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = vp.init_state(video_path=FRAMES_DIR, offload_video_to_cpu=True)
        vp.reset_state(state)
        for fidx, (negs, poss) in sorted(obj_frames.items()):
            pc = np.array(negs + poss, np.float32)
            pl = np.array([1] * len(negs) + [0] * len(poss), np.int32)  # neg->object, arm-pos->negative
            vp.add_new_points_or_box(state, frame_idx=fidx, obj_id=0, points=pc, labels=pl)
        for rev in (False, True):
            for out_idx, _o, logits in vp.propagate_in_video(state, reverse=rev):
                masks[out_idx] |= (logits[0] > 0.0).cpu().numpy()[0]
    torch.cuda.empty_cache()
    if do_smooth:
        masks, _r = SA._repair_temporal_mask_outliers(masks)
        masks, _s = SA._smooth_masks(masks)
    PROP_OBJ["masks"] = masks
    return len(obj_frames)


def _save_npy(path, arr):
    """Write a .npy durably (flush + fsync) so the Save&finish os._exit can't
    truncate a large array mid-write."""
    with open(path, "wb") as fh:
        np.save(fh, arr)
        fh.flush(); os.fsync(fh.fileno())


def _save(annos=None):
    if PROP["masks"] is None:
        return False, "nothing to save — propagate first"
    os.makedirs(os.path.dirname(MASK_PATH), exist_ok=True)
    if os.path.exists(MASK_PATH):
        auto = MASK_PATH.replace(".npy", "_auto.npy")
        bak = auto if not os.path.exists(auto) else MASK_PATH.replace(".npy", f"_prev_{int(time.time())}.npy")
        shutil.copy2(MASK_PATH, bak)
    _save_npy(MASK_PATH, PROP["masks"])
    extra = ""
    # object mask (from negatives) for the contact-occlusion compositor
    if PROP_OBJ["masks"] is not None:
        obj_path = MASK_PATH.replace("masks_arm.npy", "object_mask.npy")
        _save_npy(obj_path, PROP_OBJ["masks"])
        extra += f"  +object_mask {PROP_OBJ['masks'].shape}"
    # persist the prompts so masks are reproducible without re-clicking
    if annos is not None:
        import json
        pr_path = MASK_PATH.replace("masks_arm.npy", "arm_prompts.json")
        json.dump({str(f): [[float(x), float(y), int(l)] for (x, y, l) in pts]
                   for f, pts in annos.items() if pts}, open(pr_path, "w"))
        extra += "  +prompts.json"
    return True, f"💾 saved {PROP['masks'].shape} -> {MASK_PATH}{extra}"


def save_mask(annos):
    return _save(annos)[1]


def save_and_finish(annos):
    ok, msg = _save(annos)
    if not ok:
        return msg
    threading.Timer(1.5, lambda: os._exit(0)).start()
    return msg + "  |  ✅ finishing — the pipeline will now continue. You can close this tab."


def build_app():
    with gr.Blocks(title="Stage 6 SAM2 video annotator") as demo:
        gr.Markdown("## Stage 6 — SAM2 **video** annotator → masks_arm.npy (+ object_mask.npy)\n"
                    "**Positive** (green) points = the arm/hand. **Negative** (red) points = the "
                    "grasped/occluding **object** — these also become an object mask for the "
                    "contact-occlusion compositor. **Propagate**, then **Save & finish**. "
                    "Preview: arm = green, object = cyan.")
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="numpy", height=640, show_label=False, interactive=False)
            with gr.Column(scale=1):
                frame = gr.Slider(0, len(FRAMES) - 1, value=len(FRAMES) // 2, step=1, label="frame")
                label_mode = gr.Radio(["positive", "negative"], value="positive", label="click adds")
                auto_btn = gr.Button("Load HaWoR auto-prompts into this frame")
                with gr.Row():
                    undo_btn = gr.Button("Undo point")
                    clearf_btn = gr.Button("Clear frame")
                    clara_btn = gr.Button("Clear ALL")
                anno_md = gr.Markdown("**annotated frames:** none yet")
                gr.Markdown("---")
                do_smooth = gr.Checkbox(True, label="apply pipeline denoise on propagate")
                prop_btn = gr.Button("Propagate through video", variant="primary")
                preview_prop = gr.Checkbox(False, label="preview propagated result (scrub frames)")
                with gr.Row():
                    save_btn = gr.Button("Save (stay)")
                    finish_btn = gr.Button("Save & finish", variant="stop")
                status = gr.Markdown("")
        with gr.Row():
            prop_video = gr.Video(label="propagated result ▶ (play / scrub)", height=420,
                                  autoplay=False, interactive=False)
        annos = gr.State({})

        frame.change(load_frame, [frame, annos, preview_prop], [img, status, anno_md])
        img.select(click, [frame, annos, label_mode, preview_prop], [annos, img, anno_md])
        undo_btn.click(undo, [frame, annos, preview_prop], [annos, img, anno_md])
        clearf_btn.click(clear_frame, [frame, annos, preview_prop], [annos, img, anno_md])
        clara_btn.click(clear_all, [frame, preview_prop], [annos, img, anno_md])
        auto_btn.click(load_auto, [frame, annos, preview_prop], [annos, img, anno_md])
        preview_prop.change(load_frame, [frame, annos, preview_prop], [img, status, anno_md])
        prop_btn.click(propagate, [annos, do_smooth], [status, preview_prop, prop_video])
        save_btn.click(save_mask, [annos], [status])
        finish_btn.click(save_and_finish, [annos], [status])
        demo.load(load_frame, [frame, annos, preview_prop], [img, status, anno_md])
    return demo


def main():
    global FRAMES, FRAMES_DIR, MASK_PATH, PREVIEW_MP4, KPTS_L, DET_L, _H, _W, IPRED
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_demo", type=Path, required=True)
    ap.add_argument("--host", default=os.environ.get("VIEWER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("VIEWER_PORT", "7860")))
    ap.add_argument("--share", action="store_true", default=os.environ.get("VIEWER_SHARE", "0") == "1")
    args = ap.parse_args()

    pd = args.processed_demo
    video_path = pd / "video_L.mp4"
    if not video_path.exists():
        sys.exit(f"[annotate] missing {video_path} (run stages 1-2 first)")
    seg_dir = pd / "segmentation_processor"
    MASK_PATH = str(seg_dir / "masks_arm.npy")
    PREVIEW_MP4 = str(seg_dir / "annot_preview.mp4")
    FRAMES_DIR = str(pd / "original_images")

    print(f"[annotate] dumping frames from {video_path} ...")
    SA._dump_frames_as_jpegs(video_path, Path(FRAMES_DIR))
    FRAMES = sorted(glob.glob(f"{FRAMES_DIR}/*.jpg"))
    if not FRAMES:
        sys.exit(f"[annotate] no frames in {FRAMES_DIR}")
    _H, _W = cv2.imread(FRAMES[0]).shape[:2]

    hand_npz = pd / "hand_processor" / "hand_data_left.npz"
    if hand_npz.exists():
        hd = np.load(hand_npz)
        KPTS_L, DET_L = hd["kpts_2d"], hd["hand_detected"]
        print(f"[annotate] hand keypoints loaded for auto-prompt seeding")
    else:
        print("[annotate] no hand_data_left.npz — auto-prompt seeding disabled")

    print(f"[annotate] {len(FRAMES)} frames @ {_W}x{_H}; building SAM2 image predictor...")
    imodel = build_sam2(SAM2_CONFIG_NAME, SAM2_CHECKPOINT, device=DEVICE)
    IPRED = SAM2ImagePredictor(imodel)

    demo = build_app()
    print(f"[annotate] launching on {args.host}:{args.port} — annotate, propagate, then 'Save & finish'")
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share,
                        show_error=True, allowed_paths=[str(seg_dir)])


if __name__ == "__main__":
    main()
