"""RB5-850e arm + xhand hand Isaac overlay renderer.

Camera at the CV origin looking -Z (OpenGL = pyrender frame). The RB5 arm base
is fixed at (T_CV2GL @ T_cam_base) from the IK adapter; its 6 joints are driven
per frame from rb5_q. The xhand floats at the tracked wrist (T_CV2GL @ wrist),
fingers from qpos. Outputs the robot_{rgb,depth,mask}.npy contract.

Run: OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=4 \
   python render_rb5_isaac_overlay.py --headless --enable_cameras \
     --width 1920 --height 1080 --start 110 --n 6 --preview
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/result/skill2policy/rb5_overlay_input.npz")
ap.add_argument("--jn", default="/result/skill2policy/rb5_overlay_input_jointnames.json")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n", type=int, default=0, help="0 = all frames")
ap.add_argument("--out", default="/result/skill2policy/rb5_isaac_overlay")
ap.add_argument("--preview", action="store_true", help="composite over real frames -> PNG, no npy")
ap.add_argument("--img_dir", default="/data/uhnam/skill2policy/kakao/rgb_hawor/extracted_images")
ap.add_argument("--bg_video", default=None,
                help="preview background video (e.g. inpaint_processor/video_human_inpaint.mkv) "
                     "so the human arm is removed; else raw frames from --img_dir")
ap.add_argument("--tint", default="1.0,0.759,0.412")
ap.add_argument("--light_dir", default="0.25,0.85,0.45")
ap.add_argument("--calib", action="store_true",
                help="render frame --start with a grid of candidate hand->flange "
                     "mount rotations, over bg; pick the natural one.")
ap.add_argument("--mount_flange", default=None,
                help="rx,ry,rz euler deg: mount the xhand on the RB5 flange (link6) "
                     "with this rotation offset, instead of at the true wrist.")
ap.add_argument("--key_int", type=float, default=450.0)
ap.add_argument("--dome_int", type=float, default=90.0)
ap.add_argument("--fill_int", type=float, default=125.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from rb5_finger_semantics import (
    FINGER_LABEL_IDS, SEMANTIC_TYPE, finger_labels_from_semantics,
    xhand_finger_link_names, expected_finger_link_names, validate_finger_link_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(
    os.environ.get("SKILL2POLICY_ISAAC_ASSETS", REPO_ROOT / "isaac_assets")
).resolve()
XHAND_USD = {
    "right": str(ASSET_ROOT / "xhand_right_urdf" / "xhand_right.usd"),
    "left": str(ASSET_ROOT / "xhand_left_urdf" / "xhand_left.usd"),
}
RB5_USD = str(ASSET_ROOT / "rb5_850e_urdf" / "rb5_850e.usd")
RB5_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
T_CV2GL = np.diag([1.0, -1.0, -1.0])
H_APERTURE = 20.955
OFFSCREEN = np.array([0.0, 0.0, 5.0])


def quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = 2*np.sqrt(1+t); return np.array([0.25*s,(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s])
    i = int(np.argmax([R[0,0],R[1,1],R[2,2]]))
    if i==0: s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); return np.array([(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s])
    if i==1: s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); return np.array([(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s])
    s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); return np.array([(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s])


def main():
    d = np.load(args.data); joint_meta = json.load(open(args.jn)); jn = joint_meta["joint_names"]
    side_value = d["side"]
    side = str(side_value.item() if np.asarray(side_value).ndim == 0 else side_value)
    if side not in XHAND_USD:
        raise ValueError(f"unsupported hand side in overlay input: {side!r}")
    for path in (Path(XHAND_USD[side]), Path(RB5_USD)):
        if not path.is_file():
            raise FileNotFoundError(f"missing Isaac USD asset: {path}")
    if joint_meta.get("side", side) != side:
        raise ValueError("joint-name sidecar does not match overlay input side")
    if joint_meta.get("embodiment") != "xhand":
        raise ValueError("joint-name sidecar must declare embodiment='xhand'")
    if not joint_meta.get("trajectory_constraints", {}).get("enforced", False):
        raise ValueError("joint-name sidecar lacks an enforced trajectory contract")
    if "safety_constraints_enforced" not in d or not bool(
        np.asarray(d["safety_constraints_enforced"]).item()
    ):
        raise ValueError(
            "overlay input is not execution-constrained; regenerate it with "
            "rb5_build_overlay_input.py"
        )
    focal = float(d["img_focal"])
    W, Hh = args.width, args.height
    if W <= 0 or Hh <= 0:
        raise ValueError(f"invalid render size: {W}x{Hh}")
    if "img_width" in d and "img_height" in d:
        expected_wh = (int(d["img_width"]), int(d["img_height"]))
        if expected_wh != (W, Hh):
            raise ValueError(
                f"render size {(W, Hh)} does not match adapter size {expected_wh}"
            )
    T = d["rb5_q"].shape[0]
    s0, s1 = args.start, (T if args.n == 0 else min(args.start+args.n, T)); Tn = s1-s0
    if s0 < 0 or s0 >= T or Tn <= 0:
        raise ValueError(f"empty/invalid frame range: start={args.start}, n={args.n}, T={T}")
    expected_shapes = {
        "rb5_q": (T, 6), "wrist_pos": (T, 3), "wrist_rot": (T, 3, 3),
        "qpos": (T, 12), "valid": (T,), "T_cam_base": (4, 4),
    }
    for key, shape in expected_shapes.items():
        if key not in d or d[key].shape != shape:
            actual = None if key not in d else d[key].shape
            raise ValueError(f"overlay input {key} shape {actual}, expected {shape}")
    focal_mm = focal / W * H_APERTURE

    # RB5 base pose in GL
    Tcb = d["T_cam_base"]
    base_pos = T_CV2GL @ Tcb[:3, 3]; base_R = T_CV2GL @ Tcb[:3, :3]
    base_quat = quat_wxyz(base_R)

    sim = SimulationContext(SimulationCfg(dt=1/60, gravity=(0.0, 0.0, 0.0)))
    tint = tuple(float(x) for x in args.tint.split(","))
    d_cam = np.array([float(x) for x in args.light_dir.split(",")]); d_cam /= np.linalg.norm(d_cam)
    d_gl = T_CV2GL @ d_cam; d_gl /= np.linalg.norm(d_gl)

    def dir_quat(dv):
        zc = np.array([0.0,0.0,-1.0]); v = np.cross(zc, dv); s = np.linalg.norm(v); c = float(zc @ dv)
        if s < 1e-8: R = np.eye(3) if c > 0 else np.diag([1.,-1.,-1.])
        else:
            vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); R = np.eye(3)+vx+vx@vx*((1-c)/(s*s))
        return tuple(quat_wxyz(R))

    sim_utils.DomeLightCfg(intensity=args.dome_int, color=tint).func("/World/dome", sim_utils.DomeLightCfg(intensity=args.dome_int, color=tint))
    k = sim_utils.DistantLightCfg(intensity=args.key_int, color=tint, angle=8.0); k.func("/World/key", k, orientation=dir_quat(d_gl))
    f = sim_utils.DistantLightCfg(intensity=args.fill_int, color=tint, angle=25.0); f.func("/World/fill", f, orientation=dir_quat(np.array([0.,0.,-1.])))

    cam = Camera(CameraCfg(prim_path="/World/Cam", update_period=0, height=Hh, width=W,
        data_types=["rgb","distance_to_image_plane","semantic_segmentation"],
        semantic_filter=[SEMANTIC_TYPE],
        colorize_semantic_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal_mm, horizontal_aperture=H_APERTURE, clipping_range=(0.01,10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0,0,0), rot=(1.0,0,0,0), convention="opengl")))

    hand = Articulation(ArticulationCfg(prim_path="/World/hand",
        spawn=sim_utils.UsdFileCfg(usd_path=XHAND_USD[side]),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(OFFSCREEN)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=200.0, damping=10.0)}))
    rb5 = Articulation(ArticulationCfg(prim_path="/World/rb5",
        spawn=sim_utils.UsdFileCfg(usd_path=RB5_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(base_pos), rot=tuple(base_quat)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1e4, damping=1e3)}))
    sim.reset(); dev = hand.device
    # Label the 12 XHand finger link prims so the camera emits per-finger
    # semantic segmentation (consumed by the contact-occlusion compositor).
    import omni.usd as _omni_usd
    _stage = _omni_usd.get_context().get_stage()
    _expected = expected_finger_link_names(side)
    _finger_prims = {}
    for _p in _stage.Traverse():
        if _p.GetPath().pathString.startswith("/World/hand") and _p.GetName() in _expected:
            _finger_prims[_p.GetName()] = _p
    validate_finger_link_names(side, _finger_prims.keys())
    for _finger, _links in xhand_finger_link_names(side).items():
        for _ln in _links:
            sim_utils.add_labels(_finger_prims[_ln], labels=[_finger], instance_name=SEMANTIC_TYPE)
    print(f"[semantic] labeled {len(_finger_prims)} finger prims ({SEMANTIC_TYPE})", flush=True)
    # Optionally hide the RB5 flange (link6). NOTE: link6 bridges link5 to the
    # hand, so hiding it leaves a gap — off by default.
    if os.environ.get("HIDE_FLANGE"):
        from pxr import Usd, UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for p in stage.Traverse():
            if p.GetPath().pathString.startswith("/World/rb5") and p.GetName() == "link6":
                UsdGeom.Imageable(p).MakeInvisible()
                print(f"[flange] hid {p.GetPath()}", flush=True)
    cam.update(0)
    print(f"[intr] K=\n{cam.data.intrinsic_matrices[0].cpu().numpy().round(1)}  expect fx={focal:.1f} cx={W/2} cy={Hh/2}", flush=True)
    print(f"[rb5] base_pos_gl={base_pos.round(3)} joints={rb5.joint_names}", flush=True)

    # joint index maps
    hq = {n:i for i,n in enumerate(hand.joint_names)}
    fids = [(hq[n], ki) for ki,n in enumerate(jn) if n in hq]
    rq = {n:i for i,n in enumerate(rb5.joint_names)}
    rids = [(rq[n], ci) for ci,n in enumerate(RB5_JOINTS) if n in rq]
    print(f"[map] hand fingers {len(fids)}/12  rb5 joints {len(rids)}/6", flush=True)
    if len(fids) != 12 or len(rids) != 6:
        raise RuntimeError(
            f"Isaac articulation contract mismatch: hand={len(fids)}/12, rb5={len(rids)}/6"
        )

    import cv2
    from scipy.spatial.transform import Rotation as _Rsc
    li6 = rb5.body_names.index("link6")

    def _Rq(q):   # wxyz -> matrix
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    if args.calib:
        t = s0
        rqpos = rb5.data.default_joint_pos[0].clone()
        for ji, ci in rids: rqpos[ji] = float(d["rb5_q"][t, ci])
        rb5.write_joint_state_to_sim(rqpos.unsqueeze(0), torch.zeros_like(rqpos).unsqueeze(0)); rb5.write_data_to_sim()
        hqp = hand.data.default_joint_pos[0].clone()
        for ji, ci in fids: hqp[ji] = float(d["qpos"][t, ci])
        sim.step(); rb5.update(1/60)
        b6pos = rb5.data.body_pos_w[0, li6].cpu().numpy(); Rb6 = _Rq(rb5.data.body_quat_w[0, li6].cpu().numpy())
        bg = None
        if args.bg_video:
            cap = cv2.VideoCapture(args.bg_video)
            for _ in range(t): cap.read()
            ok, bg = cap.read(); cap.release()
            bg = cv2.resize(bg, (W, Hh)) if bg is not None else None
        cands = [("id", [0,0,0]), ("x90",[90,0,0]), ("x180",[180,0,0]), ("y90",[0,90,0]),
                 ("y180",[0,180,0]), ("z90",[0,0,90]), ("z180",[0,0,180]), ("x-90",[-90,0,0])]
        imgs = []
        for name, eul in cands:
            hR = Rb6 @ _Rsc.from_euler("xyz", eul, degrees=True).as_matrix()
            hroot = torch.tensor(np.concatenate([b6pos, quat_wxyz(hR)]), dtype=torch.float32, device=dev).unsqueeze(0)
            hand.write_root_pose_to_sim(hroot); hand.write_root_velocity_to_sim(torch.zeros((1,6),device=dev))
            hand.write_joint_state_to_sim(hqp.unsqueeze(0), torch.zeros_like(hqp).unsqueeze(0)); hand.write_data_to_sim()
            sim.step(); hand.update(1/60); cam.update(1/60)
            rgb = cam.data.output["rgb"][0,...,:3].cpu().numpy()
            dep = cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
            m = np.isfinite(dep) & (dep>0) & (dep<4.0)
            comp = bg.copy() if bg is not None else np.zeros((Hh,W,3),np.uint8)
            comp[m] = rgb[m][:, ::-1]
            cv2.putText(comp, name, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,0,255), 5)
            imgs.append(cv2.resize(comp, (W//2, Hh//2)))
        grid = np.vstack([np.hstack(imgs[0:4]), np.hstack(imgs[4:8])])
        preview_root = Path(args.out)
        preview_root.mkdir(parents=True, exist_ok=True)
        calib_path = preview_root / "rb5_calib.png"
        cv2.imwrite(str(calib_path), grid)
        print(f"[calib] wrote {calib_path}", flush=True); print("RB5_DONE", flush=True); os._exit(0)

    rgb_buf = np.zeros((Tn,Hh,W,3), np.uint8); depth_buf = np.full((Tn,Hh,W), np.inf, np.float32); mask_buf = np.zeros((Tn,Hh,W), bool)
    finger_label_buf = np.zeros((Tn,Hh,W), np.uint8)
    import cv2
    bg_frames = None
    if args.preview and args.bg_video:
        cap = cv2.VideoCapture(args.bg_video); bg_frames = {}
        fi = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            if s0 <= fi < s1:
                bg_frames[fi] = cv2.resize(fr, (W, Hh)) if (fr.shape[1] != W or fr.shape[0] != Hh) else fr
            fi += 1
            if fi >= s1: break
        cap.release()
        print(f"[bg] loaded {len(bg_frames)} inpainted bg frames from {args.bg_video}", flush=True)
    preview_imgs = []
    offroot = torch.tensor(np.concatenate([OFFSCREEN,[1,0,0,0]]), dtype=torch.float32, device=dev).unsqueeze(0)
    R_mount = (_Rsc.from_euler("xyz", [float(x) for x in args.mount_flange.split(",")], degrees=True).as_matrix()
               if args.mount_flange else None)
    for ki, t in enumerate(range(s0, s1)):
        valid = bool(d["valid"][t])
        # RB5 joints
        rqpos = rb5.data.default_joint_pos[0].clone()
        for ji, ci in rids: rqpos[ji] = float(d["rb5_q"][t, ci])
        rb5.write_joint_state_to_sim(rqpos.unsqueeze(0), torch.zeros_like(rqpos).unsqueeze(0)); rb5.write_data_to_sim()
        # hand
        hq2 = hand.data.default_joint_pos[0].clone()
        if valid:
            for ji, ci in fids: hq2[ji] = float(d["qpos"][t, ci])
            if R_mount is not None:               # mount the hand on the RB5 flange (link6)
                sim.step(); rb5.update(1/60)
                b6p = rb5.data.body_pos_w[0, li6].cpu().numpy(); Rb6 = _Rq(rb5.data.body_quat_w[0, li6].cpu().numpy())
                hroot = torch.tensor(np.concatenate([b6p, quat_wxyz(Rb6 @ R_mount)]), dtype=torch.float32, device=dev).unsqueeze(0)
            else:                                  # place at the true tracked wrist
                Rgl = T_CV2GL @ d["wrist_rot"][t].astype(np.float64); pgl = T_CV2GL @ d["wrist_pos"][t].astype(np.float64)
                hroot = torch.tensor(np.concatenate([pgl, quat_wxyz(Rgl)]), dtype=torch.float32, device=dev).unsqueeze(0)
        else:
            hroot = offroot
        hand.write_root_pose_to_sim(hroot); hand.write_root_velocity_to_sim(torch.zeros((1,6),device=dev))
        hand.write_joint_state_to_sim(hq2.unsqueeze(0), torch.zeros_like(hq2).unsqueeze(0)); hand.write_data_to_sim()
        sim.step(); hand.update(1/60); rb5.update(1/60); cam.update(1/60)
        rgb = cam.data.output["rgb"][0,...,:3].cpu().numpy()
        depth = cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze()
        m = np.isfinite(depth) & (depth>0) & (depth<4.0)
        if args.preview:
            real = bg_frames.get(t) if bg_frames is not None else cv2.imread(f"{args.img_dir}/{t:04d}.jpg")
            comp = real.copy() if real is not None else np.zeros((Hh,W,3),np.uint8)
            comp[m] = rgb[m][:, ::-1]
            preview_imgs.append(comp)
        else:
            rgb_buf[ki]=rgb; mask_buf[ki]=m; depth_buf[ki][m]=depth[m]
            sem = cam.data.output["semantic_segmentation"][0].cpu().numpy()
            _info = cam.data.info
            _si = (_info[0] if isinstance(_info, (list, tuple)) else _info)["semantic_segmentation"]
            finger_label_buf[ki] = finger_labels_from_semantics(sem, m, _si)
        if ki % 5 == 0: print(f"  frame {t} mask={int(m.sum())} fingers={int((finger_label_buf[ki]>0).sum()) if not args.preview else 0}", flush=True)

    if args.preview:
        preview_root = Path(args.out)
        preview_root.mkdir(parents=True, exist_ok=True)
        if len(preview_imgs) > 6:
            preview_path = preview_root / "rb5_overlay.mp4"
            vw = cv2.VideoWriter(str(preview_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, Hh))
            for im in preview_imgs:
                vw.write(im)
            vw.release()
            print(f"[ok] wrote {preview_path}", flush=True)
        else:
            preview_path = preview_root / "rb5_preview.png"
            cv2.imwrite(str(preview_path), np.concatenate(preview_imgs[:6], axis=1))
            print(f"[ok] wrote {preview_path}", flush=True)
    else:
        if np.any(d["valid"][s0:s1]) and not np.any(finger_label_buf):
            raise RuntimeError(
                "Isaac rendered valid hand frames but emitted no per-finger semantics"
            )
        out = args.out; os.makedirs(f"{out}/overlay_processor", exist_ok=True)
        np.save(f"{out}/overlay_processor/robot_rgb.npy", rgb_buf)
        np.save(f"{out}/overlay_processor/robot_depth.npy", depth_buf.astype(np.float16))
        np.save(f"{out}/overlay_processor/robot_mask.npy", mask_buf)
        np.save(f"{out}/overlay_processor/robot_finger_labels.npy", finger_label_buf)
        np.save(f"{out}/overlay_processor/robot_finger_mask.npy", finger_label_buf > 0)
        json.dump({"side": side, "finger_mask": {"label_ids": FINGER_LABEL_IDS}},
                  open(f"{out}/overlay_processor/manifest.json", "w"))
        print(f"[ok] wrote {out} (+ finger semantics + manifest)", flush=True)
    print("RB5_DONE", flush=True)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        status = 1
    try:
        app.close()
    except Exception:
        pass
    os._exit(status)
