---
type: "query"
date: "2026-05-20T16:02:47.858228+00:00"
question: "Why does retarget_input.npz connect so many communities across the codebase?"
contributor: "graphify"
source_nodes: ["retarget_input.npz (HaWoR output)", "retarget_one_hand", "extract_hand_contact.main", "Depth alignment per-frame (a,b) LSQ via HaWoR anchors", "optimize_cube_pose.main", "export (final_pose)"]
---

# Q: Why does retarget_input.npz connect so many communities across the codebase?

## Answer

retarget_input.npz is a pure data sink (16 incoming edges, 0 outgoing). It is produced once by extract_for_retarget.main (HaWoR stage 1) and read by every downstream consumer: stage 2/3 retargeting (retarget_one_hand, retarget_from_npz_contact, npz_to_result_json), contact estimation (extract_hand_contact, inspect_contact_3d), inpainting depth alignment (HaWoR anchors), visualization (overlay_on_rgb, inspect_combined, play_sequence, compare_stages), cube pose optimization (optimize_cube_pose, inspect_cube_pose), and indirectly the LeRobot dataset conversion (via final_pose.pkl). It is the load-bearing artifact of the pipeline: corrupting it breaks every downstream stage. It crosses 9+ communities so graphify surfaces it as the highest-betweenness cross-cutting bridge in the codebase.

## Source Nodes

- retarget_input.npz (HaWoR output)
- retarget_one_hand
- extract_hand_contact.main
- Depth alignment per-frame (a,b) LSQ via HaWoR anchors
- optimize_cube_pose.main
- export (final_pose)