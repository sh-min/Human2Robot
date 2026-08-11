"""Parallel orchestrator for the layered overlay pipeline.

The linear pipeline (run_layered.py) is a chain. But its stages form a DAG with
three independent branches that join only at the final composite:

    1 prepare → 2 inject ─┬─ PIPE1 bg:    3 SAM2-hand → 4 E2FGVI
                          ├─ PIPE2 robot:  5 pyrender xhand RGBD
                          └─ PIPE3 cube:   6 DA2 depth → 7 align
                                             → 8a SAM2-cube → 8b Diffusion-VAS amodal
                                                  ▲ needs masks_arm.npy from PIPE1.3
                          (join) → 10 composite_layered

Two cross-edges constrain it:
  * PIPE3.8a (segment_cube) reads masks_arm.npy → waits on PIPE1.3.
  * composite needs all three pipe outputs.

So this runs stages 1-2 first, then PIPE1/2/3 concurrently (threads, GPU-pinned),
with PIPE3 blocking on a hand-mask-ready event before 8a. Stage 8b is sharded
across the whole GPU pool (amodal windows are independent). Finally composite.

Each stage skips itself if its output exists, same as run_layered.py.

Usage:
    conda activate inpaint
    python run_layered_parallel.py \
        --input        /data/.../extracted_images \
        --hawor_npz    /data/.../retarget_input.npz \
        --right_pkl    /data/.../qpos_xhand_right.pkl \
        --left_pkl     /data/.../qpos_xhand_left.pkl \
        --data_root      /result/skill2policy/raw \
        --processed_root /result/skill2policy/processed \
        --demo_name cam0 --gpus 0,1,5,6,7
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent

# Stage scripts that belong to this pipeline (for GPU-memory attribution).
_OUR_SCRIPTS = ("segment_arms.py", "inpaint_hands.py",
                "render_xhand_overlay_depth.py", "estimate_depth.py",
                "align_depth.py", "segment_cube.py", "amodal_cube.py",
                "composite_layered.py")

_timing = {}            # stage name -> elapsed seconds
_timing_lock = threading.Lock()


def _run(name, cmd, gpu=None, extra_env=None):
    """Run one stage as a subprocess, pinned to `gpu`, recording its wall time."""
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if extra_env:
        env.update(extra_env)
    tag = f"{name}" + (f" [gpu {gpu}]" if gpu is not None else "")
    print(f"\n$ ({tag}) {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    dt = time.time() - t0
    with _timing_lock:
        _timing[name] = _timing.get(name, 0.0) + dt
    print(f"[time] {name}: {dt/60:.1f} min", flush=True)
    return dt


# ── GPU-memory sampler ──────────────────────────────────────────────────────
class GpuMemSampler(threading.Thread):
    """Polls nvidia-smi compute-apps, attributing memory to processes whose
    cmdline references one of this pipeline's stage scripts. Records peak total."""

    def __init__(self, interval=3.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_total_mib = 0
        self.peak_per_gpu = {}      # gpu index -> peak MiB (our procs)
        self._stop = threading.Event()

    @staticmethod
    def _our_pids():
        pids = set()
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-compute-apps=pid,used_memory,gpu_uuid",
                 "--format=csv,noheader,nounits"], text=True)
        except Exception:
            return {}
        # map gpu_uuid -> index
        try:
            idx_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid",
                 "--format=csv,noheader"], text=True)
            uuid2idx = {}
            for line in idx_out.strip().splitlines():
                i, u = [s.strip() for s in line.split(",", 1)]
                uuid2idx[u] = int(i)
        except Exception:
            uuid2idx = {}
        rows = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            parts = [s.strip() for s in line.split(",")]
            pid, mem = int(parts[0]), int(parts[1])
            uuid = parts[2] if len(parts) > 2 else ""
            rows.append((pid, mem, uuid2idx.get(uuid, -1)))
        return rows

    def _is_ours(self, pid):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cl = f.read().decode("utf-8", "ignore")
        except Exception:
            return False
        return any(s in cl for s in _OUR_SCRIPTS)

    def run(self):
        while not self._stop.is_set():
            per_gpu = {}
            for pid, mem, gidx in self._our_pids():
                if self._is_ours(pid):
                    per_gpu[gidx] = per_gpu.get(gidx, 0) + mem
            total = sum(per_gpu.values())
            self.peak_total_mib = max(self.peak_total_mib, total)
            for g, m in per_gpu.items():
                self.peak_per_gpu[g] = max(self.peak_per_gpu.get(g, 0), m)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ── Pipes ───────────────────────────────────────────────────────────────────
def pipe_bg(pd, py, gpu, hand_ready):
    """SAM2 hand seg → E2FGVI inpaint. Signals hand_ready after segmentation."""
    masks = pd / "segmentation_processor" / "masks_arm.npy"
    if masks.exists():
        print("[skip] PIPE1.3 masks_arm.npy exists", flush=True)
    else:
        _run("3_segment_arms", [py, str(HERE / "segment_arms.py"),
                                "--processed_demo", str(pd)], gpu=gpu)
    hand_ready.set()                                     # unblock PIPE3.8a

    inp = pd / "inpaint_processor" / "video_human_inpaint.mkv"
    if inp.exists():
        print("[skip] PIPE1.4 video_human_inpaint.mkv exists", flush=True)
    else:
        _run("4_inpaint", [py, str(HERE / "inpaint_hands.py"),
                           "--processed_demo", str(pd), "--mode", "legacy"], gpu=gpu)


def pipe_robot(pd, py, gpu, hawor_npz, right_pkl, left_pkl, hand):
    out = pd / "overlay_processor" / "robot_mask.npy"
    if out.exists():
        print("[skip] PIPE2.5 robot_mask.npy exists", flush=True)
        return
    _run("5_robot_render",
         [py, str(HERE / "render_xhand_overlay_depth.py"),
          "--processed_demo", str(pd), "--hawor_npz", str(hawor_npz),
          "--right_pkl", str(right_pkl), "--left_pkl", str(left_pkl),
          "--hand", hand],
         gpu=gpu, extra_env={"PYOPENGL_PLATFORM": "egl"})


def pipe_cube(pd, py, gpus, dvas_env, hand_ready, args):
    serial_gpu = gpus[-1]      # distinct from bg(gpus[0]) / robot(gpus[1])
    # 6 depth + 7 align — independent, start immediately
    if (pd / "depth_processor" / "depth_raw.npy").exists():
        print("[skip] PIPE3.6 depth_raw.npy exists", flush=True)
    else:
        _run("6_estimate_depth", [py, str(HERE / "estimate_depth.py"),
                                  "--processed_demo", str(pd),
                                  "--encoder", args.encoder], gpu=serial_gpu)
    if (pd / "depth_processor" / "depth_aligned.npy").exists():
        print("[skip] PIPE3.7 depth_aligned.npy exists", flush=True)
    else:
        _run("7_align_depth", [py, str(HERE / "align_depth.py"),
                               "--processed_demo", str(pd)], gpu=serial_gpu)

    # 8a SAM2 cube modal — needs masks_arm (PIPE1.3)
    if (pd / "cube_layer" / "cube_mask_amodal.npy").exists():
        print("[skip] PIPE3.8 cube_mask_amodal.npy exists", flush=True)
        return
    if not (pd / "cube_layer" / "cube_mask_raw.npy").exists():
        print("[wait] PIPE3.8a waiting on hand mask (PIPE1.3)...", flush=True)
        hand_ready.wait()
        _run("8a_segment_cube", [py, str(HERE / "segment_cube.py"),
                                 "--processed_demo", str(pd),
                                 "--quantile", str(args.cube_quantile)],
             gpu=serial_gpu)

    # 8b Diffusion-VAS amodal — shard windows across the whole GPU pool
    n = len(gpus)
    print(f"[info] PIPE3.8b amodal sharded across {n} GPU(s): {gpus}", flush=True)
    threads = []
    for i, g in enumerate(gpus):
        cmd = ["conda", "run", "-n", dvas_env, "--no-capture-output",
               "python", str(HERE / "amodal_cube.py"),
               "--processed_demo", str(pd),
               "--overlap", str(args.overlap),
               "--top_percentile", str(args.top_percentile),
               "--smooth_sigma", str(args.smooth_sigma),
               "--bbox_margin", str(args.bbox_margin),
               "--shard_id", str(i), "--num_shards", str(n)]
        t = threading.Thread(target=_run,
                             args=(f"8b_amodal_shard{i}", cmd, g))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    # assemble (no GPU needed)
    _run("8b_amodal_assemble",
         ["conda", "run", "-n", dvas_env, "--no-capture-output",
          "python", str(HERE / "amodal_cube.py"),
          "--processed_demo", str(pd),
          "--top_percentile", str(args.top_percentile),
          "--smooth_sigma", str(args.smooth_sigma),
          "--bbox_margin", str(args.bbox_margin),
          "--assemble_only"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--hawor_npz", type=Path, required=True)
    ap.add_argument("--right_pkl", type=Path, required=True)
    ap.add_argument("--left_pkl", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--processed_root", type=Path, required=True)
    ap.add_argument("--demo_name", default="cam0")
    ap.add_argument("--demo_num", default="0")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--hand", choices=["left", "right", "both"], default="both")
    ap.add_argument("--gpus", default="0",
                    help="comma-separated GPU pool, e.g. 0,1,5,6,7")
    ap.add_argument("--dvas_env", default="diffusion_vas")
    ap.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--cube_quantile", type=float, default=0.25)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--top_percentile", type=float, default=1.0)
    ap.add_argument("--smooth_sigma", type=float, default=2.0)
    ap.add_argument("--bbox_margin", type=int, default=25)
    ap.add_argument("--threshold_joint", type=int, default=5)
    ap.add_argument("--zmcp_sigma_t", type=float, default=8.0)
    ap.add_argument("--edge_sigma", type=float, default=1.5)
    args = ap.parse_args()

    py = sys.executable                                  # inpaint env
    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    pd = args.processed_root / args.demo_name / args.demo_num

    sampler = GpuMemSampler()
    sampler.start()
    wall0 = time.time()

    # ── Shared prefix: 1 prepare, 2 inject (fast, sequential) ──
    if (pd / "video_L.mp4").exists():
        print("[skip] 1 prepare_demo", flush=True)
    else:
        _run("1_prepare", [py, str(HERE / "prepare_demo.py"),
                           "--input", str(args.input),
                           "--data_root", str(args.data_root),
                           "--processed_root", str(args.processed_root),
                           "--demo_name", args.demo_name,
                           "--demo_num", args.demo_num, "--glob", args.glob])
    _run("2_inject", [py, str(HERE / "inject_hawor_data.py"),
                      "--processed_demo", str(pd),
                      "--hawor_npz", str(args.hawor_npz)])

    # ── Launch the 3 pipes concurrently ──
    hand_ready = threading.Event()
    g_bg = gpus[0]
    g_robot = gpus[1 % len(gpus)]
    pipes = [
        threading.Thread(target=pipe_bg,
                         args=(pd, py, g_bg, hand_ready)),
        threading.Thread(target=pipe_robot,
                         args=(pd, py, g_robot, args.hawor_npz,
                               args.right_pkl, args.left_pkl, args.hand)),
        threading.Thread(target=pipe_cube,
                         args=(pd, py, gpus, args.dvas_env, hand_ready, args)),
    ]
    for p in pipes:
        p.start()
    for p in pipes:
        p.join()

    # ── Join: composite ──
    final = pd / "overlay_processor_layered" / "video_overlay.mp4"
    if final.exists():
        print(f"[skip] 10 composite — {final} exists", flush=True)
    else:
        _run("10_composite",
             [py, str(HERE / "composite_layered.py"),
              "--processed_demo", str(pd), "--hawor_npz", str(args.hawor_npz),
              "--cube_mask_npy", "cube_layer/cube_mask_amodal.npy",
              "--threshold_joint", str(args.threshold_joint),
              "--zmcp_sigma_t", str(args.zmcp_sigma_t),
              "--edge_sigma", str(args.edge_sigma)], gpu=gpus[0])

    wall = time.time() - wall0
    sampler.stop()

    # ── Cost summary ──
    lines = ["=" * 56, " Parallel pipeline complete", "=" * 56,
             f" final: {final}", "-" * 56, " per-stage CPU time (sum of stage wall times):"]
    for k in sorted(_timing):
        lines.append(f"   {k:24s} {_timing[k]/60:6.1f} min")
    serial = sum(_timing.values())
    lines += ["-" * 56,
              f" sum of stage times (serial-equiv): {serial/60:.1f} min",
              f" actual wall-clock (parallel):       {wall/60:.1f} min",
              f" speedup:                            {serial/max(wall,1e-9):.2f}x",
              "-" * 56,
              f" peak GPU memory (this pipeline):    {sampler.peak_total_mib/1024:.1f} GiB total"]
    for g in sorted(sampler.peak_per_gpu):
        lines.append(f"   gpu {g}: {sampler.peak_per_gpu[g]/1024:.1f} GiB peak")
    lines.append("=" * 56)
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    (pd / "parallel_timing.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
