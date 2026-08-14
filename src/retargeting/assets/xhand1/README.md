# xhand1 — RobotEra XHAND1 left hand

Source: `xhand_left_package` (RobotEra XHAND1, 12-DOF left hand), uploaded 2026-08-14.

Kinematically identical to `../xhand/xhand_left.urdf`: same 12 actuated joints,
same names, same order, same origins. A pkl retargeted for `xhand` drives this
model unchanged. Two things differ.

**Frame.** The release expresses the hand rotated 180 deg about (1,-1,0) from
ours (`T_new = R T_old R`, `R = [[0,-1,0],[-1,0,0],[0,0,-1]]`). Rather than
rewrite every origin, a `left_hand_base_link` root was added with a fixed mount
joint carrying that rotation, so the root frame matches ours and
`../R_mano_xhand_left.npy` still applies. `left_hand_index_bend_joint`'s axis
was flipped back to `-1 0 0` — the release is the only joint of the twelve
whose sign convention is opposite to ours.

**Model.** 30 links instead of 21: the finger linkage bars (`*back_link*`) are
separate here, fused into the parent link mesh in the older export. Visuals are
per-vertex-coloured PLYs carved out of the release's
`glb/xhand_left_colored_scene.glb` — white shell, dark charcoal back, the real
two-tone. The older export had no such colours, so it was tinted from a MuJoCo
per-link `rgba` table that had the base link orange and the index bend link
peach; `mjcf` is `None` for this embodiment because a per-link colour cannot
express a two-tone link.

    meshes/           original STL, as released
    meshes_colored/   per-link visual PLY, vertex-coloured (referenced by the URDF)

Left hand only — the release ships no right hand.
