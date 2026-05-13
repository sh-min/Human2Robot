# Graph Report - src  (2026-05-13)

## Corpus Check
- 54 files · ~127,696 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 432 nodes · 644 edges · 23 communities (18 shown, 5 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 61 edges (avg confidence: 0.89)
- Token cost: 192,855 input · 34,034 output

## Community Hubs (Navigation)
- [[_COMMUNITY_InpaintingRetargeting Path Shims|Inpainting/Retargeting Path Shims]]
- [[_COMMUNITY_URDF + xhand Inspection Viewers|URDF + xhand Inspection Viewers]]
- [[_COMMUNITY_Retargeting Visualization|Retargeting Visualization]]
- [[_COMMUNITY_Hand Estimation & Contact Pipeline|Hand Estimation & Contact Pipeline]]
- [[_COMMUNITY_Retargeting Core (DexPilot + Chamfer)|Retargeting Core (DexPilot + Chamfer)]]
- [[_COMMUNITY_Inpainting Orchestrator + Contact Retarget|Inpainting Orchestrator + Contact Retarget]]
- [[_COMMUNITY_Skill Dataset & Sliding Window|Skill Dataset & Sliding Window]]
- [[_COMMUNITY_Skill Classifier Models (MLP  TCN  Transformer)|Skill Classifier Models (MLP / TCN / Transformer)]]
- [[_COMMUNITY_Long-Horizon Inference|Long-Horizon Inference]]
- [[_COMMUNITY_Annotation Tool|Annotation Tool]]
- [[_COMMUNITY_Data Preprocess & MANO Utils|Data Preprocess & MANO Utils]]
- [[_COMMUNITY_Forward Tensor Doc|Forward Tensor Doc]]
- [[_COMMUNITY_Skill Label Definitions|Skill Label Definitions]]
- [[_COMMUNITY_Retargeting Path Constants|Retargeting Path Constants]]
- [[_COMMUNITY_Stage Comparison & R_MANO_XHAND Procrustes|Stage Comparison & R_MANO_XHAND Procrustes]]
- [[_COMMUNITY_E2FGVI Inpainting (inpaint_hands)|E2FGVI Inpainting (inpaint_hands)]]
- [[_COMMUNITY_SAM2 JPEG Dump Helper|SAM2 JPEG Dump Helper]]
- [[_COMMUNITY_Arm Mask Dilation|Arm Mask Dilation]]
- [[_COMMUNITY_skill_classifier.models init|skill_classifier.models init]]
- [[_COMMUNITY_MANO Palmar Mask Assets|MANO Palmar Mask Assets]]
- [[_COMMUNITY_Frame-10 2D3D Vis Overlay|Frame-10 2D/3D Vis Overlay]]
- [[_COMMUNITY_MANO Frame-10 Visualization|MANO Frame-10 Visualization]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 26 edges
2. `visualize_contact_retarget.main()` - 25 edges
3. `infer_long_horizon.main` - 23 edges
4. `retarget_from_npz.retarget_one_hand` - 22 edges
5. `train.main` - 17 edges
6. `extract_hand_contact.main` - 14 edges
7. `overlay_on_rgb.main` - 12 edges
8. `Retargeting README` - 12 edges
9. `SkillWindowDataset` - 11 edges
10. `inspect_combined.main` - 11 edges

## Surprising Connections (you probably didn't know these)
- `transformer.yaml config` --references--> `SkillTransformer`  [INFERRED]
  src/skill_classifier/config/transformer.yaml → src/skill_classifier/models/transformer.py
- `mlp.yaml config` --references--> `SkillMLP`  [INFERRED]
  src/skill_classifier/config/mlp.yaml → src/skill_classifier/models/mlp.py
- `frame10 MANO+xhand 3D Plotly viz` --references--> `save_xhand_3d()`  [INFERRED]
  src/retargeting/vis/frame10_xhand_3d.html → src/retargeting/visualize_contact_retarget.py
- `train.main` --references--> `mlp.yaml config`  [INFERRED]
  src/skill_classifier/train.py → src/skill_classifier/config/mlp.yaml
- `train.main` --references--> `transformer.yaml config`  [INFERRED]
  src/skill_classifier/train.py → src/skill_classifier/config/transformer.yaml

## Hyperedges (group relationships)
- **features.pt produced by preprocess, consumed by classifier and inference** — preprocess_main, skill_dataset_load_recordings, infer_long_horizon_main, skill_dataset_features_pt_bundle [INFERRED 0.85]
- **xhand URDF consumed by retargeting + visualization** — xhand_right_urdf, xhand_left_urdf, yml_xhand_right_dexpilot, yml_xhand_left_dexpilot, overlay_on_rgb_main, play_sequence_main, inspect_combined_main, inspect_wrist_axes_main [EXTRACTED 1.00]
- **gt_labels.json flows from annotator to preprocess and inference** — annotation_tool_save_gt, annotation_tool_gt_labels_json, preprocess_labels_per_token, infer_long_horizon_load_gt_labels [INFERRED 0.85]
- **All skill classifier models register via registry decorator** — registry_register_model, transformer_skilltransformer, mlp_skillmlp, tcn_skilltcn, tcn_supcon_skilltcnsupcon [EXTRACTED 1.00]
- **Five-stage inpainting + overlay pipeline** — prepare_demo__build_from_video, inject_hawor_data__bbox_from_kpts, segment_arms__segment_hand, inpaint_hands__inpaint_video, render_xhand_overlay_render_hands [EXTRACTED 1.00]
- **Modules consuming HaWoR retarget_input.npz** — inject_hawor_data__project_3d_to_2d, render_xhand_overlay_render_hands, extract_hand_contact_predict_contact, hawor_retarget_input_npz [EXTRACTED 1.00]
- **MANO->xhand coordinate alignment trio** — _paths_r_mano_xhand, render_xhand_overlay_t_cv2gl, render_xhand_overlay_render_hands [EXTRACTED 1.00]
- **Two-stage retargeting pipeline (DexPilot vector + normal-aware Chamfer)** — retarget_from_npz_retarget_one_hand, retarget_from_npz_dexpilot_stage1, retarget_from_npz_build_contact_refiner, retarget_from_npz_stage2_normal_chamfer [INFERRED 0.95]
- **One-time asset generation: R_MANO_XHAND, palmar mask, finger parts** — compute_r_mano_xhand_main, build_palmar_mask_main, build_finger_parts_main, _paths_load_r_mano_xhand [INFERRED 0.85]
- **Stage 1 vs Stage 2 comparison/visualization tools** — compare_stages_main, visualize_contact_retarget_main, overlay_on_rgb_main [INFERRED 0.85]

## Communities (23 total, 5 thin omitted)

### Community 13 - "Inpainting/Retargeting Path Shims"
Cohesion: 0.2
Nodes (9): Repo-relative paths for the retargeting module.  Importable by sibling scripts s, ensure_sam2_importable(), ensure_e2fgvi_importable(), load_R_mano_xhand(), Repo-relative paths and import shims for the inpainting module.  Vendored deps:, Make `import sam2` work without installing the submodule.      SAM2 ships as a p, Make `from model.e2fgvi_hq import InpaintGenerator` work.      Upstream E2FGVI h, Procrustes-fit rotation aligning xhand root frame with MANO wrist frame.      Bu (+1 more)

### Community 1 - "URDF + xhand Inspection Viewers"
Cohesion: 0.06
Nodes (43): Extract right/left hand subtree from star1 full-body URDF and emit standalone xh, Show xhand mesh (q=0) + human MANO 21-joint skeleton (canonical frame) for both, Per-finger colored skeleton: one color per finger so connectivity is     legible, Visualize the xhand at q=0 with the wrist-link axis arrows so we can read off th, Returns list of (verts_world, faces) for all visual meshes at q=0., Overlay both retargeted xhands onto the original RGB at HaWoR's estimated wrist, joint_color(), Interactive 3D playback (trimesh viewer): xhand robots + MANO skeletons + camera (+35 more)

### Community 8 - "Retargeting Visualization"
Cohesion: 0.13
Nodes (26): style(), _eq_axes(), Retargeting visualization — two separate outputs per run:    frame{N}_mano_2d.pn, Load qpos for a single frame from a retargeting pkl file., Yellow xhand skeleton + DexPilot vectors (color=col)., MANO hand (no contact points) + centroids + DexPilot vecs + xhand., vertex_finger_labels(), to_canonical() (+18 more)

### Community 0 - "Hand Estimation & Contact Pipeline"
Cohesion: 0.05
Nodes (51): pts: (N, 3) cam-frame 3D. Returns (N, 2) image pixel coords + depth., HaWoR -> npz extractor for retargeting.  Wraps the HaWoR repo (vendored at <repo, Project per-frame MANO mesh vertices onto RGB and write mp4., extract_hand_contact.main, extract_hand_contact.predict_contact, extract_hand_contact.remove_small_contact_components, extract_hand_contact.get_bbox_from_kpts, extract_hand_contact.build_contact_mesh (+43 more)

### Community 4 - "Retargeting Core (DexPilot + Chamfer)"
Cohesion: 0.08
Nodes (33): HaWoR npz -> xhand qpos sequence (DexPilot retargeting).  Usage:     conda activ, retarget_from_npz.retarget_one_hand, Ordered to match _MANO_JOINT_TO_FINGER values: [thumb, index, mid, ring, pinky]., Stage-2 refiner: normal-aware vertex matching between human contact     verts an, Build a palmar (manipulation-relevant) vertex mask for MANO right/left.  Loads e, MANO pkl stores chumpy arrays; cast to numpy., Build per-vertex finger-part labels for MANO right/left.  Uses the MANO skinning, _tip_link_names() (+25 more)

### Community 9 - "Inpainting Orchestrator + Contact Retarget"
Cohesion: 0.09
Nodes (15): HaWoR npz + HACO contact -> xhand qpos sequence (DexPilot retargeting).  Same as, _run(), main(), End-to-end inpainting + xhand overlay pipeline.  All stages are local scripts in, Inject HaWoR keypoints + derived bboxes into a phantom demo folder.  Phantom's p, _dump_frames_as_jpegs(), _segment_one_pass(), _segment_hand() (+7 more)

### Community 2 - "Skill Dataset & Sliding Window"
Cohesion: 0.06
Nodes (38): Dataset, Windowed dataset for skill classification from per-recording bundled features., Load all recording bundles under {data_root}/{recording_glob}/features.pt., Sliding-window dataset over per-recording bundled features.      Sample = (rec_i, Sampler, Train skill classifier on precomputed V-JEPA + hand pose features.  Usage:     c, Compute macro and weighted F1 without sklearn., Save a confusion matrix figure with raw counts and normalized (recall) views. (+30 more)

### Community 5 - "Skill Classifier Models (MLP / TCN / Transformer)"
Cohesion: 0.11
Nodes (22): Run skill classifier on every token position.     Returns per-token predictions, Temporal Transformer for skill classification., Small temporal transformer over the feature window.      Input:  vjepa [B, W, D_, MLP baseline for skill classification., Concatenates V-JEPA + hand features over the window, then MLP.      Input:  vjep, Model registry for skill classification architectures.  To add a new model:, Temporal Convolutional Network for skill classification., Conv1d with causal (left-only) padding. (+14 more)

### Community 6 - "Long-Horizon Inference"
Cohesion: 0.08
Nodes (28): run_classifier_mano_only(), load_gt_labels(), Inference on long-horizon multi-skill episodes.  Given a long video with multipl, Extract per-token V-JEPA features from frames., Run hand-only classifier per frame (no V-JEPA, no token downsampling)., Plot skill prediction timeline., Render a vertical panel showing per-skill horizontal probability bars.     gt_id, Create video with skill label bar (bottom) + probability bars (right). (+20 more)

### Community 10 - "Annotation Tool"
Cohesion: 0.13
Nodes (17): Find all episode directories in data_dir., discover_episodes(), get_video_fps(), build_rgb_video(), frames_to_ranges(), make_panel_html(), Web-based GT labeling tool for long-horizon skill sequences.  Usage:     conda a, Extract FPS from an existing video file. (+9 more)

### Community 3 - "Data Preprocess & MANO Utils"
Cohesion: 0.07
Nodes (33): Load hand keypoints. JSON keys are frame names like 'rgb_frame00000'., Load global_orient + hand_pose + kpts_3d from result.json. Returns [F, 2, 207]., Load MANO params from result.json and convert to axis-angle.      Returns [F, 96, Preprocess each recording into a single bundled `features.pt`.  For every record, Run V-JEPA over PNG frames in `image_dir`, return [T, D] features (T=F//tubelet), Read result.json and return [num_frames, 96] axis-angle features., Average pairs of frames → [num_tokens, D]., V-JEPA feature extractor for cube manipulation videos.  Loads the pretrained V-J (+25 more)

### Community 16 - "Retargeting Path Constants"
Cohesion: 0.67
Nodes (3): retargeting._paths, URDF_ROOT constant, CONFIG_DIR constant

### Community 7 - "Stage Comparison & R_MANO_XHAND Procrustes"
Cohesion: 0.09
Nodes (27): retarget_from_npz.main, Procrustes-fit R_MANO_XHAND (the rotation that takes a vector in MANO canonical, MANO at zero pose (T-pose) — return wrist-relative MCP knuckles (5, 3)., xhand at q=0: MCP link origins in wrist link frame (5, 3).      Each MCP link's, Find R minimizing ||A @ R - B||_F, ensuring proper rotation det=+1., Open3D 3D viewer that overlays three things in the same cam-frame so stage-1 vs, Load every visual mesh in the xhand URDF; return list of     (link_name, vertice, Pre-build a single TriangleMesh that concatenates all links. Returns     (mesh, (+19 more)

### Community 11 - "E2FGVI Inpainting (inpaint_hands)"
Cohesion: 0.19
Nodes (12): _read_masks(), _create_binary_masks(), _pad_images(), _get_ref_index(), _clear_gpu_memory(), _inpaint_video(), E2FGVI video inpainting over SAM2 arm masks.  Minimal port of phantom's `HandInp, Load arm masks, resize, and dilate (matches phantom's read_mask). (+4 more)

### Community 15 - "MANO Palmar Mask Assets"
Cohesion: 0.38
Nodes (7): palmar_mask_right.png (MANO right-hand palmar vertex mask visualization), MANO right hand mesh model, Palmar vertex subset (421 of 778 MANO vertices), Dorsal vertex subset (357 of 778 MANO vertices), Palm direction vector [0.0, -1.0, 0.0] with threshold 0.0, Four-view scatter visualization (palm, dorsal, side, isometric), Retargeting pipeline (contact-aware Chamfer)

### Community 12 - "Frame-10 2D/3D Vis Overlay"
Cohesion: 0.24
Nodes (11): Frame 10 MANO + xhand 2D/3D Visualization Overlay, RGB Frame 10 Reference Image (Rubik's Cube in Hands), Left Hand - Original Retarget (retarget_from_npz), Left Hand - Contact-Adjusted Retarget (retarget_from_npz_contact), Right Hand - Original Retarget (retarget_from_npz), Right Hand - Contact-Adjusted Retarget (retarget_from_npz_contact), Frame Index 10, MANO Skeleton/Vector (blue/orange) (+3 more)

### Community 14 - "MANO Frame-10 Visualization"
Cohesion: 0.32
Nodes (8): MANO Retargeting Visualization - Frame 10, RGB frame 10 panel (hands holding Rubik's cube on wooden table), Left hand - Original MANO 3D plot, Left hand - Contact-adjusted MANO 3D plot, Right hand - Original MANO 3D plot, MANO hand model (human source representation), Contact-aware retargeting (pre-retargeting source comparison), Finger legend: thumb, index, middle, ring, pinky (both hands)

## Knowledge Gaps
- **162 isolated node(s):** `Repo-relative paths for the retargeting module.  Importable by sibling scripts s`, `Extract right/left hand subtree from star1 full-body URDF and emit standalone xh`, `Show xhand mesh (q=0) + human MANO 21-joint skeleton (canonical frame) for both`, `Per-finger colored skeleton: one color per finger so connectivity is     legible`, `Visualize the xhand at q=0 with the wrist-link axis arrows so we can read off th` (+157 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Inpainting Orchestrator + Contact Retarget` to `Hand Estimation & Contact Pipeline`, `URDF + xhand Inspection Viewers`, `Skill Dataset & Sliding Window`, `Data Preprocess & MANO Utils`, `Retargeting Core (DexPilot + Chamfer)`, `Long-Horizon Inference`, `Stage Comparison & R_MANO_XHAND Procrustes`, `Retargeting Visualization`, `Annotation Tool`, `E2FGVI Inpainting (inpaint_hands)`?**
  _High betweenness centrality (0.554) - this node is a cross-community bridge._
- **Why does `infer_long_horizon.main` connect `Long-Horizon Inference` to `Skill Dataset & Sliding Window`, `Data Preprocess & MANO Utils`, `Skill Classifier Models (MLP / TCN / Transformer)`, `Inpainting Orchestrator + Contact Retarget`, `Annotation Tool`?**
  _High betweenness centrality (0.225) - this node is a cross-community bridge._
- **Why does `visualize_contact_retarget.main()` connect `Retargeting Visualization` to `Inpainting Orchestrator + Contact Retarget`, `Stage Comparison & R_MANO_XHAND Procrustes`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `retarget_from_npz.retarget_one_hand` (e.g. with `overlay_on_rgb.main` and `retarget_from_npz_contact.retarget_one_hand`) actually correct?**
  _`retarget_from_npz.retarget_one_hand` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `train.main` (e.g. with `collect_results.collect_experiments` and `mlp.yaml config`) actually correct?**
  _`train.main` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Repo-relative paths for the retargeting module.  Importable by sibling scripts s`, `Extract right/left hand subtree from star1 full-body URDF and emit standalone xh`, `Show xhand mesh (q=0) + human MANO 21-joint skeleton (canonical frame) for both` to the rest of the system?**
  _162 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `URDF + xhand Inspection Viewers` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._