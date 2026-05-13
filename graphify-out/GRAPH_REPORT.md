# Graph Report - /home/uhnam/workspaces/skill2policy/src  (2026-05-13)

## Corpus Check
- 56 files · ~129,323 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 364 nodes · 564 edges · 29 communities (19 shown, 10 thin omitted)
- Extraction: 92% EXTRACTED · 7% INFERRED · 1% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.88)
- Token cost: 39,000 input · 16,903 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Data Preprocessing|Data Preprocessing]]
- [[_COMMUNITY_Inpainting Data Injection|Inpainting Data Injection]]
- [[_COMMUNITY_Retargeting & URDF|Retargeting & URDF]]
- [[_COMMUNITY_Contact Estimation|Contact Estimation]]
- [[_COMMUNITY_Skill Classifier Models|Skill Classifier Models]]
- [[_COMMUNITY_Annotation Tool|Annotation Tool]]
- [[_COMMUNITY_E2FGVI Inpainting|E2FGVI Inpainting]]
- [[_COMMUNITY_Long-Horizon Inference|Long-Horizon Inference]]
- [[_COMMUNITY_Video & GT Tooling|Video & GT Tooling]]
- [[_COMMUNITY_Results Aggregation|Results Aggregation]]
- [[_COMMUNITY_Feature Extraction|Feature Extraction]]
- [[_COMMUNITY_xhand Render Overlay|xhand Render Overlay]]
- [[_COMMUNITY_V-JEPA Encoder|V-JEPA Encoder]]
- [[_COMMUNITY_Pose Utilities|Pose Utilities]]
- [[_COMMUNITY_SAM2 Arm Segmentation|SAM2 Arm Segmentation]]
- [[_COMMUNITY_URDF Subtree Extract|URDF Subtree Extract]]
- [[_COMMUNITY_Annotated Video Rendering|Annotated Video Rendering]]
- [[_COMMUNITY_Retargeting Paths|Retargeting Paths]]
- [[_COMMUNITY_Skill Labels|Skill Labels]]
- [[_COMMUNITY_Hand Kpts Loader|Hand Kpts Loader]]
- [[_COMMUNITY_GT Labels Loader|GT Labels Loader]]
- [[_COMMUNITY_GT Evaluation|GT Evaluation]]
- [[_COMMUNITY_V-JEPA Feature Extract|V-JEPA Feature Extract]]
- [[_COMMUNITY_MANO-only Classifier|MANO-only Classifier]]
- [[_COMMUNITY_TCN Forward Args|TCN Forward Args]]
- [[_COMMUNITY_SAM2 Path Constant|SAM2 Path Constant]]
- [[_COMMUNITY_E2FGVI Path Constant|E2FGVI Path Constant]]
- [[_COMMUNITY_xhand URDF Path|xhand URDF Path]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 87 edges
2. `infer_long_horizon.main` - 33 edges
3. `train.main` - 17 edges
4. `extract_hand_contact.main` - 17 edges
5. `inspect_combined.main` - 15 edges
6. `extract_for_retarget.main` - 13 edges
7. `SkillWindowDataset` - 12 edges
8. `inspect_wrist_axes.main` - 12 edges
9. `Retargeting README` - 12 edges
10. `collect_results.main` - 10 edges

## Surprising Connections (you probably didn't know these)
- `train.main` --references--> `mlp.yaml config`  [INFERRED]
  src/skill_classifier/train.py → src/skill_classifier/config/mlp.yaml
- `train.main` --references--> `transformer.yaml config`  [INFERRED]
  src/skill_classifier/train.py → src/skill_classifier/config/transformer.yaml
- `main()` --calls--> `get_bbox_from_kpts()`  [EXTRACTED]
  run.py → contact_estimation/extract_hand_contact.py
- `main()` --calls--> `build_contact_mesh()`  [EXTRACTED]
  run.py → contact_estimation/extract_hand_contact.py
- `main()` --calls--> `make_viz()`  [EXTRACTED]
  run.py → contact_estimation/extract_hand_contact.py

## Hyperedges (group relationships)
- **5-stage inpainting + overlay pipeline** — prepare_demo_main, inject_hawor_data_main, segment_arms_main, inpaint_hands_main, render_xhand_overlay_main [EXTRACTED 1.00]
- **HaWoR retarget_input.npz consumers** — inject_hawor_data_main, render_xhand_overlay_main, run_main [EXTRACTED 1.00]
- **MANO -> xhand coordinate alignment for overlay** — render_xhand_overlay_render_hands, _paths_r_mano_xhand, render_xhand_overlay_t_cv2gl [INFERRED 0.85]

## Communities (29 total, 10 thin omitted)

### Community 0 - "Data Preprocessing"
Cohesion: 0.05
Nodes (46): downsample_to_tokens(), extract_mano(), extract_vjepa(), labels_per_token(), Preprocess each recording into a single bundled `features.pt`.  For every record, Read result.json and return [num_frames, 96] axis-angle features., Average pairs of frames → [num_tokens, D]., Run V-JEPA over PNG frames in `image_dir`, return [T, D] features (T=F//tubelet) (+38 more)

### Community 1 - "Inpainting Data Injection"
Cohesion: 0.06
Nodes (37): ensure_e2fgvi_importable(), ensure_sam2_importable(), load_R_mano_xhand(), R_MANO_XHAND module constant, SAM2_CHECKPOINT path, _bbox_from_kpts (BBOX_PAD_RATIO=0.4), inject_hawor_data.main, HaWoR keypoints used to bypass Epic-Kitchens hand detector (top-down view failure) (+29 more)

### Community 2 - "Retargeting & URDF"
Cohesion: 0.09
Nodes (35): DexPilot vector retargeting (dex-retargeting), extract_urdf.collect_subtree_links, extract_urdf.extract_one, extract_urdf (STAR1 -> xhand subtree), inspect_combined.axis_arrows, inspect_combined.load_xhand_meshes_at_q0, inspect_combined.main, inspect_combined.mano_joints_canonical (+27 more)

### Community 3 - "Contact Estimation"
Cohesion: 0.07
Nodes (34): build_contact_mesh(), get_bbox_from_kpts(), make_viz(), predict_contact(), remove_small_contact_components(), Contact Estimation README, per-frame contact .npz, DROID-SLAM (lazy import) (+26 more)

### Community 4 - "Skill Classifier Models"
Cohesion: 0.1
Nodes (23): infer_long_horizon.run_classifier, SkillMLP, mlp.yaml config, MLP baseline for skill classification., Concatenates V-JEPA + hand features over the window, then MLP.      Input:  vjep, Model registry for skill classification architectures.  To add a new model:, Temporal Convolutional Network for skill classification., Conv1d with causal (left-only) padding. (+15 more)

### Community 5 - "Annotation Tool"
Cohesion: 0.08
Nodes (28): annotation_tool.build_app, gt_labels.json, annotation_tool.main, annotation_tool.save_gt, annotation_tool.validate_segments, Dataset, infer_long_horizon.load_frames, ACTION_LABELS (+20 more)

### Community 6 - "E2FGVI Inpainting"
Cohesion: 0.15
Nodes (18): E2FGVI_CHECKPOINT path, _clear_gpu_memory, _create_binary_masks, _get_ref_index, _inpaint_video (E2FGVI temporal-batch loop), _pad_images (reflect-pad to MOD_H/MOD_W), _read_frames, _read_masks (resize + dilate) (+10 more)

### Community 7 - "Long-Horizon Inference"
Cohesion: 0.12
Nodes (17): infer_long_horizon.discover_episodes, infer_long_horizon.evaluate_against_gt, infer_long_horizon.extract_vjepa_features, infer_long_horizon.main, infer_long_horizon.make_annotated_video, infer_long_horizon.plot_timeline, infer_long_horizon.print_eval_results, infer_long_horizon.render_prob_bar (+9 more)

### Community 8 - "Video & GT Tooling"
Cohesion: 0.15
Nodes (13): build_app(), build_rgb_video(), discover_episodes(), frames_to_ranges(), get_video_fps(), make_panel_html(), Web-based GT labeling tool for long-horizon skill sequences.  Usage:     conda a, Convert a sorted list of frame indices to compact range strings. (+5 more)

### Community 9 - "Results Aggregation"
Cohesion: 0.19
Nodes (13): collect_results.collect_experiments, collect_results.collect_long_horizon, collect_results.main, collect_experiments(), collect_long_horizon(), infer_ckpt(), infer_ext(), infer_variant() (+5 more)

### Community 10 - "Feature Extraction"
Cohesion: 0.24
Nodes (11): Wraps a pretrained V-JEPA encoder for feature extraction.      Input:  [B, C, T,, data_preprocess README, VJEPAFeatureExtractor, HaWoR hand pose extractor, infer_long_horizon.load_hand_kpts, infer_long_horizon.load_hand_pose_mano, preprocess.downsample_to_tokens, preprocess.extract_mano (+3 more)

### Community 11 - "xhand Render Overlay"
Cohesion: 0.27
Nodes (9): _make_T(), parse_urdf(), Render retargeted xhand hands onto the inpainted video (pyrender + trimesh).  Re, BFS forward kinematics from URDF joint tree., Return (joints, link_meshes) from a URDF. Manual parse to avoid urdfpy     (inco, render_hands(), compute_f1(), evaluate() (+1 more)

### Community 12 - "V-JEPA Encoder"
Cohesion: 0.33
Nodes (6): V-JEPA feature extractor for cube manipulation videos.  Loads the pretrained V-J, Build V-JEPA encoder matching the pretrain config., Load the target_encoder (EMA) weights from a V-JEPA pretrain checkpoint.      Th, build_vjepa_encoder, load_pretrained_encoder, V-JEPA2 vision encoder

### Community 13 - "Pose Utilities"
Cohesion: 0.29
Nodes (6): infer_long_horizon.load_mano_axis_angle, load_mano_axis_angle(), Load MANO params from result.json and convert to axis-angle.      Returns [F, 96, rotmat_to_axis_angle, Pipeline-wide utility functions., Convert 3x3 rotation matrix to axis-angle (3,) via Rodrigues.

### Community 14 - "SAM2 Arm Segmentation"
Cohesion: 0.4
Nodes (5): SAM2 arm/hand segmentation given precise HaWoR-derived bbox prompts.  Minimal po, Run forward+reverse SAM2 propagation for one hand. Returns (T,H,W) bool., One SAM2 propagation pass (forward or backward in time) from a single     seed f, _segment_hand(), _segment_one_pass()

### Community 15 - "URDF Subtree Extract"
Cohesion: 0.67
Nodes (3): collect_subtree_links(), extract_one(), Extract right/left hand subtree from star1 full-body URDF and emit standalone xh

### Community 16 - "Annotated Video Rendering"
Cohesion: 0.5
Nodes (4): make_annotated_video(), Render a vertical panel showing per-skill horizontal probability bars.     gt_id, Create video with skill label bar (bottom) + probability bars (right)., render_prob_bar()

### Community 17 - "Retargeting Paths"
Cohesion: 0.67
Nodes (3): retargeting._paths, CONFIG_DIR constant, URDF_ROOT constant

## Ambiguous Edges - Review These
- `overlay_on_rgb.main` → `render_hands (pyrender scene + T_CV2GL)`  [AMBIGUOUS]
  src/inpainting/render_xhand_overlay.py · relation: conceptually_related_to
- `retarget_from_npz.main` → `inject_hawor_data.main`  [AMBIGUOUS]
  src/inpainting/inject_hawor_data.py · relation: shares_data_with
- `retarget_from_npz.main` → `render_xhand_overlay.main`  [AMBIGUOUS]
  src/inpainting/render_xhand_overlay.py · relation: shares_data_with
- `extract_for_retarget.main` → `inject_hawor_data.main`  [AMBIGUOUS]
  src/inpainting/inject_hawor_data.py · relation: shares_data_with
- `extract_for_retarget.main` → `render_xhand_overlay.main`  [AMBIGUOUS]
  src/inpainting/render_xhand_overlay.py · relation: shares_data_with

## Knowledge Gaps
- **142 isolated node(s):** `Repo-relative paths for the retargeting module.  Importable by sibling scripts s`, `Extract right/left hand subtree from star1 full-body URDF and emit standalone xh`, `Show xhand mesh (q=0) + human MANO 21-joint skeleton (canonical frame) for both`, `Per-finger colored skeleton: one color per finger so connectivity is     legible`, `Visualize the xhand at q=0 with the wrist-link axis arrows so we can read off th` (+137 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `overlay_on_rgb.main` and `render_hands (pyrender scene + T_CV2GL)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `retarget_from_npz.main` and `inject_hawor_data.main`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `retarget_from_npz.main` and `render_xhand_overlay.main`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `extract_for_retarget.main` and `inject_hawor_data.main`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `extract_for_retarget.main` and `render_xhand_overlay.main`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `main()` connect `Data Preprocessing` to `Inpainting Data Injection`, `Retargeting & URDF`, `Contact Estimation`, `Skill Classifier Models`, `Annotation Tool`, `E2FGVI Inpainting`, `Long-Horizon Inference`, `Video & GT Tooling`, `Results Aggregation`, `Feature Extraction`, `xhand Render Overlay`, `V-JEPA Encoder`, `Pose Utilities`, `SAM2 Arm Segmentation`, `Annotated Video Rendering`, `Hand Kpts Loader`, `GT Labels Loader`, `GT Evaluation`, `V-JEPA Feature Extract`, `MANO-only Classifier`?**
  _High betweenness centrality (0.730) - this node is a cross-community bridge._
- **Why does `infer_long_horizon.main` connect `Long-Horizon Inference` to `Data Preprocessing`, `Skill Classifier Models`, `Annotation Tool`, `Video & GT Tooling`, `Results Aggregation`, `Feature Extraction`, `V-JEPA Encoder`, `Pose Utilities`, `Annotated Video Rendering`, `Hand Kpts Loader`, `GT Labels Loader`, `GT Evaluation`, `V-JEPA Feature Extract`, `MANO-only Classifier`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._