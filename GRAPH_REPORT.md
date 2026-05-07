# Graph Report - .  (2026-05-07)

## Corpus Check
- Corpus is ~18,035 words - fits in a single context window. You may not need a graph.

## Summary
- 287 nodes · 449 edges · 16 communities (13 shown, 3 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Hand Contact Extraction|Hand Contact Extraction]]
- [[_COMMUNITY_XHand Retargeting Pipeline|XHand Retargeting Pipeline]]
- [[_COMMUNITY_MLPTCN Skill Classifier|MLP/TCN Skill Classifier]]
- [[_COMMUNITY_GT Annotation Tool|GT Annotation Tool]]
- [[_COMMUNITY_Long-Horizon Inference|Long-Horizon Inference]]
- [[_COMMUNITY_V-JEPA Feature Extractor|V-JEPA Feature Extractor]]
- [[_COMMUNITY_HaWoR Frame Preparation|HaWoR Frame Preparation]]
- [[_COMMUNITY_Annotation Tool Helpers|Annotation Tool Helpers]]
- [[_COMMUNITY_Results Aggregation|Results Aggregation]]
- [[_COMMUNITY_MANO-to-XHand Retargeting|MANO-to-XHand Retargeting]]
- [[_COMMUNITY_URDF Subtree Extraction|URDF Subtree Extraction]]
- [[_COMMUNITY_Path Constants|Path Constants]]
- [[_COMMUNITY_Retargeting Path Module|Retargeting Path Module]]
- [[_COMMUNITY_Skill Label Definitions|Skill Label Definitions]]
- [[_COMMUNITY_Feature Extractor Notes|Feature Extractor Notes]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 70 edges
2. `infer_long_horizon.main` - 33 edges
3. `train.main` - 17 edges
4. `extract_hand_contact.main` - 17 edges
5. `inspect_combined.main` - 15 edges
6. `SkillWindowDataset` - 12 edges
7. `inspect_wrist_axes.main` - 12 edges
8. `Retargeting README` - 12 edges
9. `extract_for_retarget.main` - 11 edges
10. `collect_results.main` - 10 edges

## Surprising Connections (you probably didn't know these)
- `train.main` --references--> `mlp.yaml config`  [INFERRED]
  src/skill_classifier/train.py → src/skill_classifier/config/mlp.yaml
- `train.main` --references--> `transformer.yaml config`  [INFERRED]
  src/skill_classifier/train.py → src/skill_classifier/config/transformer.yaml
- `main()` --calls--> `load_xhand_meshes_at_q0()`  [EXTRACTED]
  skill_classifier/train.py → retargeting/inspect_combined.py
- `main()` --calls--> `mano_joints_canonical()`  [EXTRACTED]
  skill_classifier/train.py → retargeting/inspect_combined.py
- `main()` --calls--> `axis_arrows()`  [EXTRACTED]
  skill_classifier/train.py → retargeting/inspect_combined.py

## Hyperedges (group relationships)
- **features.pt produced by preprocess, consumed by classifier and inference** — preprocess_main, skill_dataset_load_recordings, infer_long_horizon_main, skill_dataset_features_pt_bundle [INFERRED 0.85]
- **All skill classifier models register via registry decorator** — registry_register_model, transformer_skilltransformer, mlp_skillmlp, tcn_skilltcn, tcn_supcon_skilltcnsupcon [EXTRACTED 1.00]
- **gt_labels.json flows from annotator to preprocess and inference** — annotation_tool_save_gt, annotation_tool_gt_labels_json, preprocess_labels_per_token, infer_long_horizon_load_gt_labels [INFERRED 0.85]
- **HaWoR -> contact -> xhand qpos pipeline** — extract_for_retarget_main, extract_hand_contact_main, retarget_from_npz_contact_main, retarget_input_npz, contact_npz, qpos_contact_pkl [EXTRACTED 1.00]
- **MANO->xhand wrist-frame alignment used across modules** — r_mano_xhand_const, retarget_from_npz_retarget_one_hand, retarget_from_npz_contact_retarget_one_hand, overlay_on_rgb_main, play_sequence_main [EXTRACTED 1.00]
- **xhand URDF consumed by retargeting + visualization** — xhand_right_urdf, xhand_left_urdf, yml_xhand_right_dexpilot, yml_xhand_left_dexpilot, overlay_on_rgb_main, play_sequence_main, inspect_combined_main, inspect_wrist_axes_main [EXTRACTED 1.00]

## Communities (16 total, 3 thin omitted)

### Community 0 - "Hand Contact Extraction"
Cohesion: 0.05
Nodes (52): build_contact_mesh(), get_bbox_from_kpts(), make_viz(), predict_contact(), remove_small_contact_components(), render_hand(), downsample_to_tokens(), extract_mano() (+44 more)

### Community 1 - "XHand Retargeting Pipeline"
Cohesion: 0.09
Nodes (36): DexPilot vector retargeting (dex-retargeting), extract_urdf.collect_subtree_links, extract_urdf.extract_one, extract_urdf (STAR1 -> xhand subtree), inspect_combined.axis_arrows, inspect_combined.load_xhand_meshes_at_q0, inspect_combined.main, inspect_combined.mano_joints_canonical (+28 more)

### Community 2 - "MLP/TCN Skill Classifier"
Cohesion: 0.1
Nodes (23): infer_long_horizon.run_classifier, SkillMLP, mlp.yaml config, MLP baseline for skill classification., Concatenates V-JEPA + hand features over the window, then MLP.      Input:  vjep, Model registry for skill classification architectures.  To add a new model:, Temporal Convolutional Network for skill classification., Conv1d with causal (left-only) padding. (+15 more)

### Community 3 - "GT Annotation Tool"
Cohesion: 0.08
Nodes (28): annotation_tool.build_app, gt_labels.json, annotation_tool.main, annotation_tool.save_gt, annotation_tool.validate_segments, Dataset, infer_long_horizon.load_frames, ACTION_LABELS (+20 more)

### Community 4 - "Long-Horizon Inference"
Cohesion: 0.07
Nodes (31): infer_long_horizon.discover_episodes, infer_long_horizon.evaluate_against_gt, infer_long_horizon.extract_vjepa_features, infer_long_horizon.main, infer_long_horizon.make_annotated_video, infer_long_horizon.plot_timeline, infer_long_horizon.print_eval_results, infer_long_horizon.render_prob_bar (+23 more)

### Community 5 - "V-JEPA Feature Extractor"
Cohesion: 0.1
Nodes (23): V-JEPA feature extractor for cube manipulation videos.  Loads the pretrained V-J, Build V-JEPA encoder matching the pretrain config., Load the target_encoder (EMA) weights from a V-JEPA pretrain checkpoint.      Th, Wraps a pretrained V-JEPA encoder for feature extraction.      Input:  [B, C, T,, data_preprocess README, build_vjepa_encoder, load_pretrained_encoder, VJEPAFeatureExtractor (+15 more)

### Community 6 - "HaWoR Frame Preparation"
Cohesion: 0.09
Nodes (23): Contact Estimation README, per-frame contact .npz, DROID-SLAM (lazy import), extract_for_retarget.main, extract_for_retarget.prepare_jpg_frames, extract_for_retarget.project_vertices_to_rgb, extract_hand_contact.predict_contact, extract_hand_contact.remove_small_contact_components (+15 more)

### Community 7 - "Annotation Tool Helpers"
Cohesion: 0.15
Nodes (13): build_app(), build_rgb_video(), discover_episodes(), frames_to_ranges(), get_video_fps(), make_panel_html(), Web-based GT labeling tool for long-horizon skill sequences.  Usage:     conda a, Convert a sorted list of frame indices to compact range strings. (+5 more)

### Community 8 - "Results Aggregation"
Cohesion: 0.19
Nodes (13): collect_results.collect_experiments, collect_results.collect_long_horizon, collect_results.main, collect_experiments(), collect_long_horizon(), infer_ckpt(), infer_ext(), infer_variant() (+5 more)

### Community 9 - "MANO-to-XHand Retargeting"
Cohesion: 0.4
Nodes (4): HaWoR npz + HACO contact -> xhand qpos sequence (DexPilot retargeting).  Same as, _vertex_finger_labels(), HaWoR npz -> xhand qpos sequence (DexPilot retargeting).  Usage:     conda activ, retarget_one_hand()

### Community 10 - "URDF Subtree Extraction"
Cohesion: 0.67
Nodes (3): collect_subtree_links(), extract_one(), Extract right/left hand subtree from star1 full-body URDF and emit standalone xh

### Community 11 - "Path Constants"
Cohesion: 0.67
Nodes (3): retargeting._paths, CONFIG_DIR constant, URDF_ROOT constant

## Knowledge Gaps
- **107 isolated node(s):** `Repo-relative paths for the retargeting module.  Importable by sibling scripts s`, `Extract right/left hand subtree from star1 full-body URDF and emit standalone xh`, `Show xhand mesh (q=0) + human MANO 21-joint skeleton (canonical frame) for both`, `Per-finger colored skeleton: one color per finger so connectivity is     legible`, `Visualize the xhand at q=0 with the wrist-link axis arrows so we can read off th` (+102 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Hand Contact Extraction` to `XHand Retargeting Pipeline`, `MLP/TCN Skill Classifier`, `GT Annotation Tool`, `Long-Horizon Inference`, `V-JEPA Feature Extractor`, `HaWoR Frame Preparation`, `Annotation Tool Helpers`, `Results Aggregation`, `MANO-to-XHand Retargeting`?**
  _High betweenness centrality (0.682) - this node is a cross-community bridge._
- **Why does `infer_long_horizon.main` connect `Long-Horizon Inference` to `Hand Contact Extraction`, `MLP/TCN Skill Classifier`, `GT Annotation Tool`, `V-JEPA Feature Extractor`, `Annotation Tool Helpers`, `Results Aggregation`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `extract_hand_contact.main` connect `Hand Contact Extraction` to `XHand Retargeting Pipeline`, `HaWoR Frame Preparation`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `main()` (e.g. with `load_recordings` and `SkillWindowDataset`) actually correct?**
  _`main()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `train.main` (e.g. with `collect_results.collect_experiments` and `mlp.yaml config`) actually correct?**
  _`train.main` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Repo-relative paths for the retargeting module.  Importable by sibling scripts s`, `Extract right/left hand subtree from star1 full-body URDF and emit standalone xh`, `Show xhand mesh (q=0) + human MANO 21-joint skeleton (canonical frame) for both` to the rest of the system?**
  _107 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Hand Contact Extraction` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._