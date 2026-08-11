# Inpainting and layered robot overlay

This pipeline removes the visible human hand/arm, renders the retargeted robot as
RGB-D, and composites robot and object pixels with a depth-aware front/behind split.

The word `object` is deliberately generic. Kitchen recordings contain cups, a
chocolate milk carton, snack containers, lock boxes, a sponge and a trash bin; there
is no built-in single-shape assumption.

## Stages

```text
1  prepare_demo.py               input frames/video → normalized demo video
2  inject_hawor_data.py          HaWoR joints and hand boxes
3  annotate_arms.py              interactive SAM2 human-hand/arm mask
4  inpaint_hands.py              E2FGVI hand removal
5  robot renderer                robot RGB, depth and mask
6  estimate_depth.py             Depth Anything V2 scene depth       [object layer]
7  align_depth.py                metric alignment from HaWoR anchors [object layer]
8  segment_object.py             SAM2 modal object tracking          [object layer]
9  amodal_object.py              Diffusion-VAS amodal silhouette     [object layer]
10 composite_layered.py          final object/robot/background video
```

The compositor uses four ordered layers:

1. robot pixels in front of the HaWoR depth threshold
2. object pixels from the hand-removed background
3. robot pixels behind that threshold
4. inpainted background

This keeps front-facing robot pixels visible while sending the estimated rear portion
behind the restored object. The split is a 2.5D depth rule, not a complete 3D collision
solver.

## Dependencies and checkpoints

The code uses these Git submodules:

- `third_party/sam2`
- `third_party/E2FGVI`
- `third_party/Depth-Anything-V2`
- `third_party/diffusion-vas`

Expected checkpoints:

```text
third_party/sam2/checkpoints/sam2_hiera_large.pt
third_party/E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth
/result/skill2policy/ckpt/depth_anything/depth_anything_v2_vitl.pth
/result/skill2policy/ckpt/diffusion_vas/diffusion-vas-amodal-segmentation/
```

Use an environment with PyTorch, OpenCV, NumPy, SciPy, MediaPy, SAM2, E2FGVI,
PyRender and Depth Anything V2. Diffusion-VAS can run in a separate
`diffusion_vas` environment when its dependency versions conflict.

## Run

Baseline hand removal and robot overlay:

```bash
PYTHONPATH=$PWD/src python src/inpainting/run_layered.py \
  --input /path/to/episode/rgb \
  --hawor_npz /path/to/episode/rgb_hawor/retarget_input.npz \
  --right_pkl /path/to/episode/rgb_hawor/qpos_xhand_contact_right_smooth.pkl \
  --left_pkl /path/to/episode/rgb_hawor/qpos_xhand_contact_left_smooth.pkl \
  --data_root output/inpainting_raw \
  --processed_root output/inpainting
```

Add object segmentation, depth alignment and amodal occlusion handling:

```bash
PYTHONPATH=$PWD/src python src/inpainting/run_layered.py \
  --input /path/to/episode/rgb \
  --hawor_npz /path/to/episode/rgb_hawor/retarget_input.npz \
  --right_pkl /path/to/episode/rgb_hawor/qpos_xhand_contact_right_smooth.pkl \
  --left_pkl /path/to/episode/rgb_hawor/qpos_xhand_contact_left_smooth.pkl \
  --data_root output/inpainting_raw \
  --processed_root output/inpainting \
  --object_layer
```

Stage 3 opens the local annotation UI unless
`segmentation_processor/masks_arm.npy` already exists. Press **Save & finish** after
SAM2 propagation so the pipeline can continue.

The default robot backend invokes Isaac Sim for RB5-850 + XHand. Use
`--render_backend pyrender` for the legacy local RBY1 + XHand renderer.

## Outputs

For the default `demo_name=cam0`, `demo_num=0`:

```text
<processed_root>/cam0/0/
  video_L.mp4
  segmentation_processor/masks_arm.npy
  inpaint_processor/video_human_inpaint.mkv
  overlay_processor/
    robot_rgb.npy
    robot_depth.npy
    robot_mask.npy
  depth_processor/                         only with --object_layer
    depth_raw.npy
    depth_aligned.npy
    depth_align_params.npz
  object_layer/                            only with --object_layer
    object_mask_raw.npy
    object_mask_amodal.npy
    object_amodal_overlay.mp4
  overlay_processor_layered/
    video_overlay.mp4
    z_mcp.npy
```

Stages reuse existing outputs. Remove only the output of the stage you intend to
recompute.

## Main options

| Option | Meaning |
|---|---|
| `--object_layer` | enable object depth and amodal segmentation |
| `--render_backend {isaac,pyrender}` | choose robot renderer |
| `--hand {left,right,both}` | render selected hands |
| `--encoder {vits,vitb,vitl}` | Depth Anything V2 encoder |
| `--object_quantile` | depth quantile used to bootstrap the object prompt |
| `--object_overlap` | overlap between Diffusion-VAS windows |
| `--threshold_joint` | MANO joint used for front/behind depth split |
| `--zmcp_sigma_t` | temporal smoothing of the split depth |
| `--edge_sigma` | layer-edge feathering in pixels |

For multi-GPU amodal processing, `run_layered_parallel.py` shards Diffusion-VAS
windows across the devices passed through `--gpus`.
