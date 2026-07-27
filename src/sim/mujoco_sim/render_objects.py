"""Shoot every tabletop object on a table, one 3/4 view each, and write a
captioned contact sheet next to them.

This is what regenerates docs/objects/. Framing is driven by each object's
own measured extent, so a 55 mm carton and a 197 mm bin both fill the frame.
The measured size and mass are printed and captioned, which makes this a
cheap check that an asset still matches what build_objects.py claims.

Usage:
    python -m mujoco_sim.render_objects
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "src/sim/mujoco_sim/assets"
OUT = REPO / "docs/objects"

TABLE_Z = 0.75
RES = 720

NAMES = (
    "cup_blue", "cup_green", "lock_box_large", "lock_box_small",
    "milk_carton", "pringles", "sponge", "trash_bin", "cup_holder",
)

SCENE = f"""<mujoco model="object_shot">
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{RES}" offheight="{RES}"/>
    <headlight ambient="0.40 0.40 0.40" diffuse="0.45 0.45 0.45"
               specular="0.30 0.30 0.30"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.01" zfar="30"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.93 0.91 0.88" rgb2="0.80 0.79 0.77"
             width="256" height="256"/>
    <texture name="ttex" type="2d" builtin="flat" rgb1="0.17 0.17 0.18"
             width="8" height="8"/>
    <material name="table" texture="ttex" reflectance="0.15"
              specular="0.30" shininess="0.40"/>
    <material name="wall" rgba="0.93 0.91 0.88 1" specular="0.05"/>
    <material name="floor" rgba="0.85 0.84 0.81 1" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="-1.2 -1.1 2.4" dir="0.42 0.38 -1" directional="true"
           diffuse="0.62 0.62 0.62" specular="0.85 0.85 0.85"
           castshadow="true"/>
    <light pos="1.4 0.9 2.2" dir="-0.45 -0.3 -1" directional="true"
           diffuse="0.32 0.32 0.32" specular="0.55 0.55 0.55"
           castshadow="false"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor"/>
    <geom name="wall" type="box" pos="1.15 0 1.3" size="0.05 3 1.3"
          material="wall"/>
    <geom name="table" type="box" pos="0.05 0 {TABLE_Z/2:.4f}"
          size="0.80 0.75 {TABLE_Z/2:.4f}" material="table"/>
  </worldbody>
</mujoco>
"""


def extent(path):
    """Exact axis-aligned size of one object. geom_rbound is a bounding
    sphere and would badly overstate thin wire, so each primitive gets its
    own AABB."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for geom in range(model.ngeom):
        pos = data.geom_xpos[geom]
        mat = data.geom_xmat[geom].reshape(3, 3)
        size, geom_type = model.geom_size[geom], model.geom_type[geom]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half = np.abs(mat) @ size[:3]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = np.abs(mat[:, 2]) * size[1] + size[0]
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis = mat[:, 2]
            half = (np.abs(axis) * size[1]
                    + size[0] * np.sqrt(np.maximum(0.0, 1.0 - axis**2)))
        else:                                    # sphere and friends
            half = np.full(3, model.geom_rbound[geom])
        lo = np.minimum(lo, pos - half)
        hi = np.maximum(hi, pos + half)
    return hi - lo, float(sum(model.body_mass))


def _cam_quat(eye, target):
    """MuJoCo cameras look down their own -z with +y up, so build that frame
    and hand it over as a quaternion -- MjsCamera has no xyaxes field."""
    fwd = np.array(target, float) - np.array(eye, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up, -fwd]).flatten())
    return quat


def shoot(name, path, size):
    spec = mujoco.MjSpec.from_string(SCENE)
    obj = mujoco.MjSpec.from_file(str(path))
    frame = spec.worldbody.add_frame()
    frame.pos = [0.0, 0.0, TABLE_Z]
    spec.attach(obj, prefix=f"{name}_", frame=frame)

    target = (0.0, 0.0, TABLE_Z + size[2] * 0.45)
    dist = 2.3 * float(max(size)) + 0.06
    eye = (-0.62 * dist, -0.62 * dist, target[2] + 0.48 * dist)
    cam = spec.worldbody.add_camera()
    cam.name = "shot"
    cam.pos = eye
    cam.fovy = 42
    cam.quat = _cam_quat(eye, target)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=RES, width=RES) as renderer:
        renderer.update_scene(data, camera="shot")
        Image.fromarray(renderer.render()).save(OUT / f"{name}.png")


def contact_sheet(rows, cols=3, tile=340, pad=46):
    try:
        bold = ImageFont.truetype("arialbd.ttf", 20)
        plain = ImageFont.truetype("arial.ttf", 17)
    except OSError:                              # no Arial off Windows
        bold = plain = ImageFont.load_default()

    n_rows = -(-len(rows) // cols)
    sheet = Image.new("RGB", (tile * cols, (tile + pad) * n_rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, size, mass) in enumerate(rows):
        sheet.paste(Image.open(OUT / f"{name}.png").resize((tile, tile)),
                    ((i % cols) * tile, (i // cols) * (tile + pad)))
        x, y = (i % cols) * tile, (i // cols) * (tile + pad) + tile
        draw.text((x + 8, y + 3), name, fill="black", font=bold)
        draw.text((x + 8, y + 25),
                  f"{size[0]*1000:.0f} x {size[1]*1000:.0f} x "
                  f"{size[2]*1000:.0f} mm   {mass*1000:.0f} g",
                  fill="#444", font=plain)
    sheet.save(OUT / "all_objects.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NAMES:
        path = ASSETS / name / f"{name}.xml"
        size, mass = extent(path)
        shoot(name, path, size)
        rows.append((name, size, mass))
        print(f"{name:<15} {size[0]*1000:6.1f} x {size[1]*1000:6.1f} x "
              f"{size[2]*1000:6.1f} mm   {mass*1000:6.1f} g")
    contact_sheet(rows)
    print(f"wrote {OUT / 'all_objects.png'}")


if __name__ == "__main__":
    main()
