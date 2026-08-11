# 08_04 MH/SH stereo pipeline

## Fixed camera roles

- `camera_1 = SH`: auxiliary visibility and HaCo evidence only
- `camera_2 = MH`: V-JEPA input, robot trajectory, inpainting, object mask, and final overlay

This mapping is intentional: `composite_rb5_stereo_occlusion.py` treats camera 2 as the final view.

The prepared data lives at `data/kitchen_dataset/26.08.04_stereo/<episode>/`. The root `rgb`, `rgb_hawor`, and `contact` paths are relative symlinks to `camera_2` (MH), so existing single-view V-JEPA and robot code continues to consume MH.

## Prepared data

Build or validate the 24 paired episodes with:

```bash
python scripts/prepare_0804_stereo_dataset.py
```

```text
<episode>/
├── gt_labels.json
├── stereo_manifest.json
├── rgb -> camera_2/rgb
├── rgb_hawor -> camera_2/rgb_hawor
├── contact -> camera_2/contact
├── camera_1/                      # SH auxiliary
│   ├── rgb/rgb_frame000000.jpg
│   ├── rgb_hawor/retarget_input.npz
│   └── contact/rgb_frame000000.npz
└── camera_2/                      # MH primary
    ├── rgb/rgb_frame000000.jpg
    ├── rgb_hawor/retarget_input.npz
    └── contact/rgb_frame000000.npz
```

Both source streams are truncated to the annotation `num_frames`; MH and its
GT are never reordered. A six-cue motion-correlation audit plus high-motion
visual review found same-index alignment for 22 episodes. Episodes 16 and 18
have a fixed capture phase offset: while producing dual-view evidence only,
MH frame `k` reads SH frame `k-1`. This is recorded as
`temporal_alignment.camera1_frame_offset = -1` in their manifests; every other
episode records `0`. The unmatched first-frame SH cue fails open. There is no
whole-dataset shift and no temporal drift correction.

## Phone intrinsics required

### Calibration-deferred SH+MH contact/overlay pilot

For a non-metric pilot that combines SH/MH HaCo and draws the robot directly
over the original MH video, calibration can be deferred explicitly.
The opt-in approximation uses the 26mm full-frame-equivalent lens tag and the
1280-pixel image width: `fx = 1280 * 26 / 36 = 924.44px`.

```bash
FOCAL_MODE=approx_26mm VIEWS=both STAGES=hawor,haco \
  bash scripts/run_0804_stereo_hawor_haco.sh 1

BACKGROUND_MODE=source STAGES=retarget,preview \
  bash scripts/run_0804_mh_robot_overlay.sh 1

BACKGROUND_MODE=source STAGES=render,composite \
  bash scripts/run_0804_mh_robot_overlay.sh 1

# Use the exact per-view focal values stored in the reviewed HaWoR caches.
SH_FOCAL=<sh_approx_focal_px> MH_FOCAL=<mh_approx_focal_px> \
  STAGES=object bash scripts/run_0804_stereo_visibility_assets.sh 1

BACKGROUND_MODE=source USE_SH_HACO=1 STAGES=occlude \
  bash scripts/run_0804_mh_robot_overlay.sh 1
```

The raw result is
`camera_2/visibility/processed/view/0/video_overlay_robot_raw.mp4`. It does not
remove the human or place the object in front of the robot. The `occlude` stage
adds the contact-local result at
`contact_occlusion_dual_haco_raw/video_overlay_contact.mp4`: SH and MH HaCo
scores are aligned by the recorded frame offset. MH keeps the GitHub branch
decision unchanged; SH can only propose contact for the same one of five
fingers. An SH-only proposal must first pass the MH contact-local object-support
gate (`hidden_fraction >= 0.22`), and a new active run still needs the existing
MH onset gate (`hidden_fraction >= 0.42`). The compositor then hides only
matching robot-finger pixels that overlap the MH modal object mask and lie
behind the projected MH HaCo contact-surface depth (12 mm tolerance). SH never
contributes coordinates or depth; without stereo calibration, its 3D points
are not transformed into MH coordinates. Palm, arm, unrelated fingers, and
ambiguous depth remain visible. `report.json` records every SH frame lookup,
proposal, MH-qualified rescue, and activation added over MH-only. Its
`debug_contact_occlusion.mp4` shows the evidence and removed pixels.

### Contact-interior A/B comparison

The original projected-contact disk remains the default.  To compare it with
the conservative boundary-triggered completion pilot, generate the baseline
and the separate 3-pixel/25-percent-cap variant without overwriting either:

```bash
BACKGROUND_MODE=source USE_SH_HACO=1 \
  CONTACT_INTERIOR_EXPAND_PX=0 STAGES=occlude FORCE=1 \
  bash scripts/run_0804_mh_robot_overlay.sh 1

BACKGROUND_MODE=source USE_SH_HACO=1 \
  CONTACT_INTERIOR_EXPAND_PX=3 \
  CONTACT_INTERIOR_EXPAND_CAP_FRACTION=0.25 \
  STAGES=occlude FORCE=1 \
  bash scripts/run_0804_mh_robot_overlay.sh 1

conda run -n inpaint-gpu --no-capture-output \
  python src/inpainting/compare_contact_interior_expansion.py \
  --baseline_dir data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0/contact_occlusion_dual_haco_raw \
  --expanded_dir data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0/contact_occlusion_dual_haco_boundaryfill_3px_cap25_raw \
  --out_dir data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0/contact_occlusion_compare_baseline_vs_boundaryfill_3px_cap25_raw
```

The comparison video has three panels: baseline, completed result, and a dark
difference view where magenta is newly hidden and cyan would be removed.  The
completion never changes raw HaCo vertex labels.  It grows only an already
verified candidate that touches the inner semantic-finger boundary, stays
inside the same MH finger/modal-object/contact-depth component, and adds at
most 25 percent of that frame/finger's verified seed pixels.  SH still
contributes confidence only.  The runner encodes the parameters in the output
directory and validates them from `report.json`, preventing a baseline cache
from being reused as an expansion result.

### Visibility-force A/B comparison

For the deliberately aggressive pilot, the pure `visibility` mode treats a
temporally stable `SH visible AND MH hidden` event as sufficient to put that
same MH robot finger behind the MH modal object. It does not use HaCo or depth.
The event needs two onset frames and is held for three frames; removal is still
limited to the matching MH semantic finger inside the object mask.

Generate it over the original MH video without replacing either contact result:

```bash
EP=data/kitchen_dataset/26.08.04_stereo/1
PD="$EP/camera_2/visibility/processed/view/0"
conda run -n inpaint-gpu --no-capture-output \
  python src/inpainting/composite_rb5_stereo_occlusion.py \
  --camera1_rgb_dir "$EP/camera_1/rgb" \
  --camera2_rgb_dir "$EP/camera_2/rgb" \
  --camera1_hawor "$EP/camera_1/rgb_hawor/retarget_input.npz" \
  --camera2_hawor "$EP/camera_2/rgb_hawor/retarget_input.npz" \
  --camera1_visible_mask "$EP/camera_1/visibility/processed/view/0/segmentation_processor/masks_arm.npy" \
  --camera2_visible_mask "$PD/segmentation_processor/masks_arm.npy" \
  --camera1_contact_dir "$EP/camera_1/contact" \
  --contact_dir "$EP/camera_2/contact" \
  --background 08_04/mh/1.mov \
  --overlay_dir "$PD/overlay_processor" \
  --object_mask "$PD/object_layer/object_mask_modal.npy" \
  --out_dir "$PD/stereo_occlusion_visibility_force_raw" \
  --camera1_frame_offset 0 \
  --fps 24 \
  --include_visibility_haco \
  --include_haco_only
```

Build the synchronized 2x2 review video:

```bash
conda run -n inpaint-gpu --no-capture-output \
  python src/inpainting/compare_contact_occlusion_variants.py \
  --baseline_dir data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0/contact_occlusion_dual_haco_raw \
  --boundary_dir data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0/contact_occlusion_dual_haco_boundaryfill_3px_cap25_raw \
  --force_dir data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0/stereo_occlusion_visibility_force_raw \
  --out_dir data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0/contact_occlusion_compare_baseline_boundaryfill_visibility_force_raw
```

The panels are baseline HaCo, boundary fill, visibility force, and force minus
baseline. Magenta in the last panel is hidden only by force; cyan is hidden
only by baseline. The JSON report validates frame alignment, report counts,
and the semantic-finger invariant, then records all three pairwise comparisons.
This mode is a diagnostic alternative, not an additive superset of HaCo.

### XHand-thickness multi-strategy comparison

The HaCo contact-depth pilot can apply an opt-in virtual depth bias without
moving the XHand pose or changing RB5 IK:

```text
effective_robot_depth = rendered_front_depth
                      + scale * per_finger_full_thickness
```

The measured transverse OBB constants are 39.16 mm for the thumb and 29.30 mm
for the other four fingers. The bias applies only to the MH HaCo contact-surface
proxy gate; an independent metric object-depth gate remains unchanged. Scale
zero preserves the reviewed baseline exactly. Generate half- and full-thickness
pilots in separate directories with:

```bash
BACKGROUND_MODE=source USE_SH_HACO=1 \
  CONTACT_DEPTH_THICKNESS_SCALE=0.5 STAGES=occlude FORCE=1 \
  bash scripts/run_0804_mh_robot_overlay.sh 1

BACKGROUND_MODE=source USE_SH_HACO=1 \
  CONTACT_DEPTH_THICKNESS_SCALE=1.0 STAGES=occlude FORCE=1 \
  bash scripts/run_0804_mh_robot_overlay.sh 1
```

Build the six-way synchronized review:

```bash
PD=data/kitchen_dataset/26.08.04_stereo/1/camera_2/visibility/processed/view/0
conda run -n inpaint-gpu --no-capture-output \
  python src/inpainting/compare_xhand_thickness_strategies.py \
  --baseline_dir "$PD/contact_occlusion_dual_haco_raw" \
  --half_thickness_dir "$PD/contact_occlusion_dual_haco_xhanddepth_s0p5_t39p16mm_f29p3mm_raw" \
  --full_thickness_dir "$PD/contact_occlusion_dual_haco_xhanddepth_s1_t39p16mm_f29p3mm_raw" \
  --force_dir "$PD/stereo_occlusion_visibility_force_raw" \
  --overlay_dir "$PD/overlay_processor" \
  --surface_labels "$PD/overlay_processor/robot_finger_surface_labels.npy" \
  --object_mask "$PD/object_layer/object_mask_modal.npy" \
  --out_dir "$PD/contact_occlusion_compare_xhand_surface_strategies_raw"
```

The top row is zero, half, and full contact-depth bias. The bottom row is pure
SH/MH visibility force, baseline OR force, and a deliberately aggressive 2-D
safety-shell diagnostic. The shell stays on the same semantic finger, touches
an existing force seed, uses an adaptive 3--20 px radius with five-frame median
smoothing, and caps additions at 75 percent of that frame/finger seed. It can
hide pixels outside the modal object boundary, so it is not a calibrated depth
result or a recommended final compositor mode. The comparison report verifies
the 0/0.5/1.0 input contracts and records pairwise/per-finger statistics.

When the packed surface labels are supplied, the same command also produces a
surface-aware 2x2 review. Its panels are the anatomical surface debug view,
the unchanged front/baseline result, front plus side at half thickness, and
front plus side at half thickness with the back at full thickness. The derived
files are:

```text
video_xhand_surface_labels_debug.mp4
video_compare_xhand_surface_strategies_2x2.mp4
video_overlay_surface_front_side_half.mp4
video_overlay_surface_front_side_half_back_full.mp4
occluded_finger_mask_surface_front_side_half.npy
occluded_finger_mask_surface_front_side_half_back_full.npy
```

`--surface_labels` is optional and defaults to
`<overlay_dir>/robot_finger_surface_labels.npy`; it is shown explicitly above
so the reviewed input is unambiguous. The comparison refuses to replace an
existing output directory unless `--overwrite` is supplied.

These outputs are suitable for checking contact, retargeting, approximate
placement, and local front/back compositing only. They are not calibrated
metric scene-depth results. Inspect the HaWoR projection, `rb5_preview.png`,
the object-mask debug video, and the contact-occlusion debug video before using
the full result. Set an explicit `APPROX_FOCAL_PX` when reusing a reviewed
approximate HaWoR cache.

The supplied checkerboard ZIP is **not** an 08_04 phone calibration. Its SHA-256 is the same source recorded by `configs/calibration/cameras_20260802.json`, and `configs/calibration/depth_registration_20260803.json` binds those views to a RealSense D455 and D435I. The rerun result is retained only as `08_04/calibration/realsense_checkerboard_not_iphone.json` so it cannot be selected accidentally.

All 48 phone clips consistently report:

- SH/camera 1: `iPhone 13 26mm`, 1280x720
- MH/camera 2: `iPhone 17 26mm`, 1280x720
- Blackmagic Cam 3.4, 24 sensor FPS

The MOVs do not contain a pixel focal length or camera matrix. The `26mm` lens label is a 35mm-equivalent description, not a calibrated `fx` for the cropped/stabilized 1280x720 stream. Capture a checkerboard with each actual phone using the same lens, zoom, resolution, stabilization, and app settings, then use the resulting pixel focal lengths. The runner deliberately has no fallback focal value.

Use at least 12 (preferably 15–25) sharp paired board poses covering the image
centre, edges, distances, and tilts. Arrange same-pose images as
`camera_1/<index>_Color.png` (SH) and `camera_2/<index>_Color.png` (MH), then
reuse the existing robust calibrator:

```bash
python src/calibration/calibrate_stereo_checkerboard.py \
  --input 08_04/calibration/iphone_checkerboard \
  --out 08_04/calibration/iphone_stereo.json \
  --qa-dir 08_04/calibration/iphone_qa \
  --pattern-cols 9 --pattern-rows 6 \
  --square-size-mm <measured_square_edge_mm>
```

Only use the result when the geometry quality checks pass. `SH_FOCAL` is
`camera_1.camera_matrix[0][0]`; `MH_FOCAL` is
`camera_2.camera_matrix[0][0]`.

The current HaWoR adapter consumes one scalar focal (`fx = fy`) and projects around the image centre. If the phone calibration has meaningful distortion or an off-centre principal point, undistort/recentre the frames first or treat this as an explicit approximation.

## HaWoR and HaCo

Run one pilot episode first:

```bash
SH_FOCAL=<iphone13_fx_px> MH_FOCAL=<iphone17_fx_px> \
  bash scripts/run_0804_stereo_hawor_haco.sh 1
```

After checking both `vts_projection.mp4` outputs, resume all episodes:

```bash
SH_FOCAL=<iphone13_fx_px> MH_FOCAL=<iphone17_fx_px> ALL=1 \
  bash scripts/run_0804_stereo_hawor_haco.sh
```

The script verifies the source lens metadata, resolution, full HaWoR schema and camera-space coordinate flag, frame ranges, cached focal, and every HaCo frame's source/index/focal contract. HaCo reads the focal directly from each HaWoR NPZ. Partial or stale caches are refused instead of being silently reused. `STAGES=haco` requires the same explicit focal values and cannot bypass this check.

The invalid episode-1 pilot caches are preserved as `rgb_hawor.invalid_realsense_focal/`; the active `rgb_hawor/` directories are clean.

## V-JEPA feature bundle

Only MH is used for V-JEPA. The local loader expects Meta's raw V-JEPA 2
ViT-L/16 256-pixel pretraining checkpoint (including `target_encoder`), not a
Hugging Face directory or a V-JEPA 2.1 checkpoint. Download the official
4.8-GiB file explicitly when ready; interrupted downloads resume from
`vitl.pt.part`:

```bash
bash scripts/download_vjepa2_vitl.sh
```

Once the checkpoint exists:

```bash
STAGES=vjepa VJEPA_CKPT=/path/to/vitl.pt ALL=1 \
  bash scripts/run_0804_stereo_hawor_haco.sh
```

Each episode receives `features.pt` containing MH V-JEPA tokens, MH MANO, and the shared GT labels.

The 08_04 runner explicitly uses the published V-JEPA 2 temporal/spatial
profile: MH is sampled from 24 FPS to 4 FPS, resized without aspect-ratio
distortion, and center-cropped to 256. Two sampled frames form one 2-Hz token.
The bundle records the exact original-frame pairs, centre frames, and a
`frame_to_token` lookup; MANO is selected at the same centre frame and a token
that crosses a GT segment boundary is marked `-1` instead of receiving a
mixed label. Episode 1 therefore produces 58 aligned tokens from 695 source
frames, not the old dense 347-token sequence. Classifier weights trained with
the old dense profile are intentionally incompatible and must be retrained.

The runner now requires a complete, camera-space MH HaWoR NPZ before V-JEPA
starts, so this bundle cannot silently omit MANO. The loader reads weights with
PyTorch's safe `weights_only` mode, removes only the known unused RoPE
`pos_embed` entry, and then strictly validates the rest of the EMA encoder.
Pass `VJEPA_CKPT` only when using a different compatible raw checkpoint.

After all 24 aligned bundles exist, train the provided V-JEPA+MANO baseline
(episodes 1–20 train, 21–24 validation):

```bash
bash scripts/train_0804_skill_classifier.sh
```

The MLP uses an 8-token/4-second context window and disables external W&B
logging by default. The dataset loader rejects feature/MANO/label length
mismatches and refuses to mix dense and 4-FPS sampling contracts. The saved
classifier checkpoint carries the sampling signature; long-horizon inference
uses each bundle's `frame_to_token` mapping to restore predictions over all
original MH frames.

## Dual-view visibility and MH assets

Finger-specific front/back evidence requires a SAM hand/arm mask from **both** views. Detector confidence alone is only whole-hand confidence and is not a substitute. Generate SH+MH visible masks, the MH annotated-object mask, and the MH inpainted background after HaWoR completes:

```bash
SH_FOCAL=<iphone13_fx_px> MH_FOCAL=<iphone17_fx_px> \
  STAGES=masks,object bash scripts/run_0804_stereo_visibility_assets.sh 1
```

Inspect these pilot outputs before inpainting:

```text
camera_1/visibility/processed/view/0/segmentation_processor/masks_arm.npy
camera_2/visibility/processed/view/0/segmentation_processor/masks_arm.npy
camera_2/visibility/processed/view/0/object_layer/debug_object_mask.mp4
```

Then make the protected MH background, or run all three asset stages together:

```bash
SH_FOCAL=<iphone13_fx_px> MH_FOCAL=<iphone17_fx_px> \
  STAGES=inpaint bash scripts/run_0804_stereo_visibility_assets.sh 1

SH_FOCAL=<iphone13_fx_px> MH_FOCAL=<iphone17_fx_px> ALL=1 \
  bash scripts/run_0804_stereo_visibility_assets.sh
```

The object mask is derived independently for every non-`Trans` annotation interval. It is also passed as a protection mask when the human is removed from MH.

The asset runner compares upstream modification times: replacing a HaWoR NPZ
or GT file invalidates dependent masks/backgrounds instead of silently reusing
them. Mask arrays and the inpaint video are published through temporary files,
so an interrupted producer does not replace a complete artifact.

## Front/back compositing

HaCo predicts contact, not direction. The directional RGB cue is:

```text
SH visible AND MH hidden
```

For the branch-parity contact compositor, dual-view HaCo uses MH geometry plus
SH same-finger confidence rescue as described above. The separate opt-in
`--include_visibility_haco` mode additionally implements a directional RGB
visibility experiment:

```text
strong stereo OR (assisted stereo AND dual-view HaCo active)
```

The selected robot pixels must also be MH semantic finger pixels inside the MH modal object mask. Palm, arm, and pixels outside the object are never removed. `--include_haco_only` can be emitted alongside it as a non-directional comparison baseline.

This RGB-only visibility decision does not use the rejected RealSense stereo extrinsics. Each phone still needs its own correct intrinsics for HaWoR/HaCo projection. Metric-depth modes additionally require valid camera-specific depth registration and are not enabled by the 08_04 phone videos alone.

Build the contact-aware MH robot trajectory and a six-frame placement preview:

```bash
bash scripts/run_0804_mh_robot_overlay.sh 1
```

Inspect `camera_2/visibility/processed/view/0/rb5_preview.png`. The default stops here deliberately so base placement or hand side can be adjusted before allocating full arrays. Once the preview is correct:

```bash
STAGES=render bash scripts/run_0804_mh_robot_overlay.sh 1
```

This MH robot renderer populates the following arrays under
`camera_2/visibility/processed/view/0/overlay_processor/`:

```text
manifest.json
robot_rgb.npy
robot_depth.npy
robot_mask.npy
robot_finger_labels.npy
robot_finger_surface_labels.npy
robot_finger_mask.npy
```

### Packed XHand anatomical surface labels

`robot_finger_surface_labels.npy` is a visible-surface segmentation with shape
`(T,H,W)` and dtype `uint8`. The classification is evaluated from triangle
normals in each XHand link frame, so it describes the anatomical finger face,
not the face currently pointing toward the camera. Only the frontmost visible
surface at each rendered pixel is stored.

| Surface ID | Anatomical face | Debug colour |
| ---: | --- | --- |
| 1 | palmar/front (finger-pad side) | red |
| 2 | lateral/side | yellow |
| 3 | dorsal/back (nail side) | blue |

The array packs both finger and surface into one value:

```text
packed_id = (finger_id - 1) * 3 + surface_id
0          = non-finger/background
1..3       = thumb:  front, side, back
4..6       = index:  front, side, back
7..9       = middle: front, side, back
10..12     = ring:   front, side, back
13..15     = pinky:  front, side, back
```

For every non-zero packed value, the original finger ID is recovered with
`((packed_id - 1) // 3) + 1`. The runner requires the recovered `(T,H,W)` map
to equal `robot_finger_labels.npy` exactly. It also rejects any overlay cache
whose surface array is not `uint8`, contains a value outside `0..15`, or lacks
the matching `manifest.json/finger_surface_labels` packing, normal-frame,
threshold, and per-hand normal-axis contract.

If any robot pose, side, render scale, or arm mode has changed, safely rebuild
the complete overlay directory atomically:

```bash
BACKGROUND_MODE=source STAGES=render FORCE=1 \
  bash scripts/run_0804_mh_robot_overlay.sh 1
```

For an already reviewed overlay, the surface-only mode backfills the new array
without rewriting its RGB, depth, mask, or existing finger-label files. This
command reads the original render settings from the manifest, validates every
new packed frame against the existing finger labels, then replaces only the
surface array and updates the manifest after the complete render succeeds:

```bash
EP=data/kitchen_dataset/26.08.04_stereo/1
PD="$EP/camera_2/visibility/processed/view/0"
read -r SIDE ARM_MODE SOURCE_W SOURCE_H RENDER_SCALE < <(python -c '
import json, sys
manifest = json.load(open(sys.argv[1]))
source_w, source_h = map(int, manifest["source_size"])
render_w, render_h = map(int, manifest["render_size"])
scale_x, scale_y = render_w / source_w, render_h / source_h
if abs(scale_x - scale_y) > 1e-9:
    raise SystemExit("surface backfill requires a uniform render scale")
print(manifest["side"], manifest["arm_mode"], source_w, source_h, scale_x)
' "$PD/overlay_processor/manifest.json")
PYOPENGL_PLATFORM=egl \
conda run -n inpaint-gpu --no-capture-output \
  python src/inpainting/render_rb5_pyrender_overlay.py \
  --data "$PD/rb5_overlay_input_${SIDE}.npz" \
  --jn "$PD/rb5_overlay_input_${SIDE}_jointnames.json" \
  --out "$PD" \
  --width "$SOURCE_W" --height "$SOURCE_H" \
  --render_scale "$RENDER_SCALE" \
  --arm_mode "$ARM_MODE" \
  --surface_labels_only --overwrite
```

Once those MH-only arrays exist, the checked wrapper runs the complete mapping:

```bash
bash scripts/run_0804_visibility_haco_composite.sh 1
```

Its direct equivalent is copy/paste runnable:

```bash
EP=data/kitchen_dataset/26.08.04_stereo/1
PD="$EP/camera_2/visibility/processed/view/0"
C1_OFFSET=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["temporal_alignment"]["camera1_frame_offset"])' \
  "$EP/stereo_manifest.json")
conda run -n inpaint --no-capture-output \
  python src/inpainting/composite_rb5_stereo_occlusion.py \
  --camera1_rgb_dir "$EP/camera_1/rgb" \
  --camera2_rgb_dir "$EP/camera_2/rgb" \
  --camera1_hawor "$EP/camera_1/rgb_hawor/retarget_input.npz" \
  --camera2_hawor "$EP/camera_2/rgb_hawor/retarget_input.npz" \
  --camera1_visible_mask "$EP/camera_1/visibility/processed/view/0/segmentation_processor/masks_arm.npy" \
  --camera2_visible_mask "$PD/segmentation_processor/masks_arm.npy" \
  --camera1_contact_dir "$EP/camera_1/contact" \
  --contact_dir "$EP/camera_2/contact" \
  --background "$PD/inpaint_processor/video_human_inpaint.mkv" \
  --overlay_dir "$PD/overlay_processor" \
  --object_mask "$PD/object_layer/object_mask_modal.npy" \
  --out_dir "$PD/stereo_occlusion" \
  --camera1_frame_offset "$C1_OFFSET" \
  --fps 24 \
  --include_visibility_haco \
  --include_haco_only
```

`--include_visibility_haco` now refuses to run unless both visible masks and the SH contact directory are supplied. The mapping is always:

```text
--camera1_*            <episode>/camera_1/...   # SH
--camera2_*            <episode>/camera_2/...   # MH
--camera1_contact_dir  <episode>/camera_1/contact
--contact_dir          <episode>/camera_2/contact
--background / --overlay_dir / --object_mask   # MH products only
--camera1_frame_offset  # manifest value: -1 for episodes 16/18, otherwise 0
--include_visibility_haco --include_haco_only
```

The checked wrapper reads this offset from `stereo_manifest.json` and records
the exact SH source-frame lookup in both `report.json` and
`stereo_evidence.npz`.
Existing composites made with a different offset are rejected as stale.

Primary outputs are:

```text
stereo_occlusion/video_overlay_visibility.mp4
stereo_occlusion/occluded_finger_mask_visibility.npy
stereo_occlusion/video_overlay_visibility_haco.mp4
stereo_occlusion/occluded_finger_mask_visibility_haco.npy
stereo_occlusion/video_overlay_haco_only.mp4
stereo_occlusion/stereo_evidence.npz
stereo_occlusion/report.json
```

`visibility` is the pure directional force experiment. `visibility_haco` is the
directional result gated by HaCo in ambiguous cases. `haco_only` is retained
only as a non-directional A/B baseline. The legacy three-panel comparison video
intentionally remains unchanged and does not include these opt-in modes.
