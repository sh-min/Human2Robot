"""Interactive 3D viewer (matplotlib): MANO vertex point cloud + HACO contact
verts overlaid, with palmar filter toggle.

Controls (focus the figure window first):
    SPACE       toggle pause / play
    LEFT/RIGHT  step ±1 frame (auto-pauses)
    HOME/END    first / last frame
    p           toggle palmar filter on/off
    t           toggle fingertip-only filter on/off
    r           toggle right-hand visibility
    l           toggle left-hand  visibility

Usage:
    conda activate RFM_retarget
    cd <repo_root>/src/retargeting
    python inspect_contact_3d.py \
        --npz /path/to/<seq>_hawor/retarget_input.npz \
        --contact_dir /path/to/contact
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
NV = 778


def load_mask(prefix):
    out = {}
    for h in ("right", "left"):
        p = os.path.join(ASSETS, f"{prefix}_{h}.npy")
        out[h] = np.load(p).astype(bool) if os.path.exists(p) else np.ones(NV, bool)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--contact_dir", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.npz)
    v_left = data["verts_left"]
    v_right = data["verts_right"]
    valid = data["valid"]               # (2, T)  [0]=left, [1]=right
    start_idx_data = int(data["start_idx"])
    T = v_right.shape[0]

    npz_dir = os.path.dirname(os.path.abspath(args.npz))
    contact_dir = args.contact_dir or os.path.join(os.path.dirname(npz_dir), "contact")
    print(f"T={T}  start_idx={start_idx_data}  contact_dir={contact_dir}")

    palmar = load_mask("palmar_mask")
    tip = load_mask("fingertip_mask")
    print(f"palmar mask:    right={int(palmar['right'].sum())}/{NV}  "
          f"left={int(palmar['left'].sum())}/{NV}")
    print(f"fingertip mask: right={int(tip['right'].sum())}/{NV}  "
          f"left={int(tip['left'].sum())}/{NV}")

    def load_contact(frame_idx):
        out = {"right": np.zeros(NV, bool), "left": np.zeros(NV, bool)}
        path = os.path.join(contact_dir,
                            f"rgb_frame{start_idx_data + frame_idx:05d}.npz")
        if os.path.exists(path):
            d = np.load(path)
            for h in ("right", "left"):
                k = f"{h}_contact_mask"
                if k in d.files:
                    out[h] = d[k].astype(bool)
        return out

    # --- figure / axes ---
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("MANO + contact (P=palmar  SPACE=pause  ←/→=step)")

    skin_r = ax.scatter([], [], [], c="#888", s=4, alpha=0.45, depthshade=False)
    skin_l = ax.scatter([], [], [], c="#a76", s=4, alpha=0.45, depthshade=False)
    contact_r = ax.scatter([], [], [], c="#e52", s=40, alpha=0.95, depthshade=False,
                            edgecolors="black", linewidths=0.3)
    contact_l = ax.scatter([], [], [], c="#37c", s=40, alpha=0.95, depthshade=False,
                            edgecolors="black", linewidths=0.3)

    # Set view limits from first valid frame (or whole sequence range)
    def set_axes_limits():
        cat = []
        if valid[1].any():
            cat.append(v_right[valid[1]])
        if valid[0].any():
            cat.append(v_left[valid[0]])
        if not cat:
            return
        all_v = np.concatenate([c.reshape(-1, 3) for c in cat], axis=0)
        mn = all_v.min(axis=0); mx = all_v.max(axis=0)
        c = (mn + mx) / 2.0
        r = max(mx - mn) * 0.6
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect([1, 1, 1])
    set_axes_limits()
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    title = ax.set_title("")

    state = {"frame": max(0, min(T - 1, args.start)),
             "paused": False, "palmar_on": True, "tip_on": False,
             "show_right": True, "show_left": True}

    def status():
        return (f"frame {state['frame']}/{T-1}  "
                f"palmar={'ON' if state['palmar_on'] else 'OFF'}  "
                f"tip={'ON' if state['tip_on'] else 'OFF'}  "
                f"R={'on' if state['show_right'] else 'off'}  "
                f"L={'on' if state['show_left'] else 'off'}")

    def render(t):
        contacts = load_contact(t)

        for h, V, skin, contact in (("right", v_right, skin_r, contact_r),
                                     ("left",  v_left,  skin_l, contact_l)):
            h_idx = 0 if h == "left" else 1
            visible = (state["show_right"] if h == "right" else state["show_left"])
            shown = visible and bool(valid[h_idx, t])

            if shown:
                pts = V[t]                     # (778, 3)
                skin._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

                mask = contacts[h]
                if state["palmar_on"]:
                    mask = mask & palmar[h]
                if state["tip_on"]:
                    mask = mask & tip[h]
                cpts = pts[mask]
                contact._offsets3d = (cpts[:, 0], cpts[:, 1], cpts[:, 2])
            else:
                skin._offsets3d = ([], [], [])
                contact._offsets3d = ([], [], [])

        title.set_text(status())
        return skin_r, skin_l, contact_r, contact_l, title

    def step():
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % T

    def anim(frame):
        step()
        return render(state["frame"])

    ani = FuncAnimation(fig, anim, interval=1000.0 / args.fps, blit=False, cache_frame_data=False)

    def on_key(event):
        k = event.key
        if k == " " or k == "space":
            state["paused"] = not state["paused"]
        elif k == "left":
            state["paused"] = True
            state["frame"] = max(0, state["frame"] - 1)
        elif k == "right":
            state["paused"] = True
            state["frame"] = min(T - 1, state["frame"] + 1)
        elif k == "home":
            state["frame"] = 0
        elif k == "end":
            state["frame"] = T - 1
        elif k == "p":
            state["palmar_on"] = not state["palmar_on"]
        elif k == "t":
            state["tip_on"] = not state["tip_on"]
        elif k == "r":
            state["show_right"] = not state["show_right"]
        elif k == "l":
            state["show_left"] = not state["show_left"]
        else:
            return
        render(state["frame"])
        fig.canvas.draw_idle()
        print(status())

    fig.canvas.mpl_connect("key_press_event", on_key)

    render(state["frame"])
    plt.show()


if __name__ == "__main__":
    main()
