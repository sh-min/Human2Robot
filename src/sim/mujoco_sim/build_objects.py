"""Generate tabletop manipulation objects as pure-primitive MJCF.

No mesh files: every object is assembled from MuJoCo boxes, cylinders and
capsules.

  milk_carton      55 x 55 x 115 mm gable-top carton (height includes the
                   sloped roof and the sealed fin on top)
  pringles         O65 x 90 mm Pringles can
  cup_green        O80 x 105 mm cup with a handle
  cup_blue         O75 x 85 mm cup with a handle
  lock_box_large   160 x 120 x 50 mm open food container
  lock_box_small   130 x 90 x 55 mm open food container
  sponge           130 x 80 x 30 mm dish sponge with a scrub pad
  trash_bin        137 x 137 x 197 mm white open bin

Cups are hollow (ring of boxes -- MuJoCo has no hollow primitive) and so
are the containers and the bin (floor slab + four walls).

Each object's origin is its bottom center, so placing one on a table is
just ``pos = [x, y, table_top_z]``.

Mass: rigid plastic and ceramic parts use ``density``; objects dominated
by their contents (milk, chips, sponge foam) set mass explicitly.

Usage:
    python -m mujoco_sim.build_objects
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "src/sim/mujoco_sim/assets"

# Contact params shared by every object.
CONTACT = 'condim="4" friction="1.2 0.05 0.001" solref="0.01 1"'

DENSITY_CERAMIC = 2400
DENSITY_PP = 900  # polypropylene, for the containers and the bin

# --- milk carton -----------------------------------------------------
MILK_W = 0.055        # square base
MILK_BODY_H = 0.088   # rectangular part
MILK_GABLE_H = 0.020  # sloped roof
MILK_ROOF_T = 0.009   # roof slab thickness (thick enough to fill the gable)
MILK_FIN_H = 0.007    # sealed fin on the ridge  -> 115 mm total

# --- pringles can ----------------------------------------------------
PR_R = 0.0325
PR_BODY_H = 0.084
PR_LID_H = 0.006      # -> 90 mm total

# --- cups ------------------------------------------------------------
# ``arc`` is the handle centerline radius and ``rod`` its thickness; the
# finger gap between wall and handle is arc - rod - 3 mm (the handle ends
# are sunk 3 mm into the wall so the arc is welded on, not just touching).
CUP_WALL = 0.004
CUP_BASE = 0.006
CUP_SEGS = 24
CUP_HANDLE_SEGS = 8
CUP_HANDLE_DIR = -1.0  # handle bulges toward -x, i.e. toward the robot
CUPS = {
    "cup_green": dict(r=0.0400, h=0.105, arc=0.032, rod=0.0048,
                      rgb=(0.20, 0.62, 0.34)),
    "cup_blue": dict(r=0.0375, h=0.085, arc=0.030, rod=0.0048,
                     rgb=(0.15, 0.40, 0.75)),
}

# --- open containers -------------------------------------------------
BOXES = {
    "lock_box_large": dict(w=0.160, d=0.120, h=0.050, t=0.0025, base=0.004,
                           rgb=(0.88, 0.92, 0.95), rim=(0.10, 0.45, 0.80)),
    "lock_box_small": dict(w=0.130, d=0.090, h=0.055, t=0.0025, base=0.004,
                           rgb=(0.88, 0.92, 0.95), rim=(0.10, 0.45, 0.80)),
    "trash_bin": dict(w=0.137, d=0.137, h=0.197, t=0.002, base=0.003,
                      rgb=(0.95, 0.95, 0.95), rim=None),
}

# --- sponge ----------------------------------------------------------
SPONGE_W, SPONGE_D = 0.130, 0.080
SPONGE_FOAM_H = 0.022
SPONGE_PAD_H = 0.008  # -> 30 mm total


def _quat_z(rad):
    return (math.cos(rad / 2), 0.0, 0.0, math.sin(rad / 2))


def _quat_y(rad):
    return (math.cos(rad / 2), 0.0, math.sin(rad / 2), 0.0)


def _q(quat) -> str:
    return " ".join(f"{v:+.6f}" for v in quat)


def _v(*vals) -> str:
    return " ".join(f"{v:+.5f}" for v in vals)


def _rgba(rgb) -> str:
    return f'rgba="{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} 1"'


def _wall_ring(w, d, z_lo, z_hi, t, cls, rgba="", outset=0.0) -> list[str]:
    """Four boxes forming the side walls between z_lo and z_hi. The x walls
    span the full depth; the y walls are inset so corners don't double up."""
    hz = (z_hi - z_lo) / 2
    zc = (z_lo + z_hi) / 2
    hw, hd = w / 2 + outset, d / 2 + outset
    out = []
    for sx in (+1.0, -1.0):
        out.append(
            f'      <geom class="{cls}" type="box" {rgba} '
            f'pos="{_v(sx * (hw - t / 2), 0, zc)}" '
            f'size="{_v(t / 2, hd, hz)}"/>'
        )
    for sy in (+1.0, -1.0):
        out.append(
            f'      <geom class="{cls}" type="box" {rgba} '
            f'pos="{_v(0, sy * (hd - t / 2), zc)}" '
            f'size="{_v(hw - t, t / 2, hz)}"/>'
        )
    return out


def build_milk_carton() -> str:
    half = MILK_W / 2
    # Roof: two slabs from the top edge of the body up to the ridge. Local
    # x runs along the slope, local y along the ridge, local z is the
    # outward normal. They are thick enough to meet at the peak (so the
    # gable reads solid) and shifted half a thickness inward so their outer
    # faces land exactly on the roof plane.
    slope = math.hypot(half, MILK_GABLE_H)
    beta = math.atan2(MILK_GABLE_H, half)
    ht = MILK_ROOF_T / 2
    roof = []
    for sx in (+1.0, -1.0):
        nx, nz = math.sin(sx * beta), math.cos(sx * beta)
        roof.append(
            f'      <geom class="milk" type="box" '
            f'pos="{_v(sx * half / 2 - ht * nx, 0, MILK_BODY_H + MILK_GABLE_H / 2 - ht * nz)}" '
            f'quat="{_q(_quat_y(sx * beta))}" '
            f'size="{_v(slope / 2, half, ht)}" mass="0.005"/>'
        )

    total = MILK_BODY_H + MILK_GABLE_H + MILK_FIN_H
    return f"""<mujoco model="milk_carton">
  <!-- AUTO-GENERATED by build_objects.py. Do not edit by hand.
       {MILK_W*1000:.0f} x {MILK_W*1000:.0f} x {total*1000:.0f} mm gable-top carton. The body holds
       {MILK_W*MILK_W*MILK_BODY_H*1e6:.0f} ml, so mass is 0.282 kg = {MILK_W*MILK_W*MILK_BODY_H*1e6:.0f} g of milk + ~12 g of
       paper. Origin at bottom center. -->

  <default>
    <default class="milk">
      <geom {CONTACT} rgba="0.96 0.96 0.94 1"/>
    </default>
    <default class="milk_print">
      <geom contype="0" conaffinity="0" group="1" mass="0"/>
    </default>
  </default>

  <worldbody>
    <body name="root">
      <freejoint name="free"/>
      <geom name="body" class="milk" type="box"
            pos="{_v(0, 0, MILK_BODY_H / 2)}"
            size="{_v(half, half, MILK_BODY_H / 2)}" mass="0.270"/>

      <!-- Gable roof + sealed top fin. -->
{chr(10).join(roof)}
      <geom name="fin" class="milk" type="box"
            pos="{_v(0, 0, MILK_BODY_H + MILK_GABLE_H + MILK_FIN_H / 2)}"
            size="{_v(0.0015, half, MILK_FIN_H / 2)}" mass="0.002"/>

      <!-- Blue label band (visual only, 0.1 mm proud of the body). -->
      <geom class="milk_print" type="box"
            pos="{_v(0, 0, 0.035)}" size="{_v(half + 0.0001, half + 0.0001, 0.014)}"
            rgba="0.05 0.35 0.70 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_pringles() -> str:
    return f"""<mujoco model="pringles">
  <!-- AUTO-GENERATED by build_objects.py. Do not edit by hand.
       O{PR_R*2000:.0f} x {(PR_BODY_H+PR_LID_H)*1000:.0f} mm can: cardboard tube plus a plastic lid.
       Mass 0.065 kg = ~40 g of chips + tube and lid.
       Origin at bottom center. -->

  <default>
    <default class="pringles">
      <geom {CONTACT}/>
    </default>
    <default class="pringles_print">
      <geom contype="0" conaffinity="0" group="1" mass="0"/>
    </default>
  </default>

  <worldbody>
    <body name="root">
      <freejoint name="free"/>
      <geom name="tube" class="pringles" type="cylinder"
            pos="{_v(0, 0, PR_BODY_H / 2)}" size="{_v(PR_R, PR_BODY_H / 2)}"
            mass="0.055" rgba="0.78 0.11 0.15 1"/>
      <geom name="lid" class="pringles" type="cylinder"
            pos="{_v(0, 0, PR_BODY_H + PR_LID_H / 2)}"
            size="{_v(PR_R, PR_LID_H / 2)}"
            mass="0.010" rgba="0.90 0.90 0.88 1"/>

      <!-- White band under the lid (visual only). -->
      <geom class="pringles_print" type="cylinder"
            pos="{_v(0, 0, PR_BODY_H - 0.009)}" size="{_v(PR_R + 0.0001, 0.008)}"
            rgba="0.95 0.95 0.93 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_cup(name, *, r, h, arc, rod, rgb) -> str:
    r_mid = r - CUP_WALL / 2
    # 2% tangential overlap: enough to close the seams, small enough that
    # the interpenetration doesn't notch the rim.
    chord = r_mid * math.tan(math.pi / CUP_SEGS) * 1.02
    wall_h = h - CUP_BASE
    z_mid = CUP_BASE + wall_h / 2

    wall = []
    for i in range(CUP_SEGS):
        th = 2 * math.pi * i / CUP_SEGS
        wall.append(
            f'      <geom class="cup" type="box" '
            f'pos="{_v(r_mid * math.cos(th), r_mid * math.sin(th), z_mid)}" '
            f'quat="{_q(_quat_z(th))}" '
            f'size="{_v(CUP_WALL / 2, chord, wall_h / 2)}"/>'
        )

    cx = CUP_HANDLE_DIR * (r - 0.003)
    zc = h / 2
    pts = []
    for i in range(CUP_HANDLE_SEGS + 1):
        t = -math.pi / 2 + math.pi * i / CUP_HANDLE_SEGS
        pts.append((cx + CUP_HANDLE_DIR * arc * math.cos(t), 0.0, zc + arc * math.sin(t)))
    handle = [
        f'      <geom class="cup" type="capsule" '
        f'fromto="{_v(*a, *b)}" size="{rod:.5f}"/>'
        for a, b in zip(pts, pts[1:])
    ]

    gap = (arc - rod - 0.003) * 1000
    return f"""<mujoco model="{name}">
  <!-- AUTO-GENERATED by build_objects.py. Do not edit by hand.
       O{r*2000:.0f} x {h*1000:.0f} mm cup: {CUP_SEGS}-segment ring wall on a disc base,
       plus a capsule-arc handle with ~{gap:.0f} mm of finger clearance
       between the handle and the wall. Origin at bottom center. -->

  <default>
    <default class="cup">
      <geom {CONTACT} density="{DENSITY_CERAMIC}" {_rgba(rgb)}/>
    </default>
  </default>

  <worldbody>
    <body name="root">
      <freejoint name="free"/>
      <geom name="base" class="cup" type="cylinder"
            pos="{_v(0, 0, CUP_BASE / 2)}" size="{_v(r, CUP_BASE / 2)}"/>

      <!-- Ring wall. -->
{chr(10).join(wall)}

      <!-- Handle. -->
{chr(10).join(handle)}
    </body>
  </worldbody>
</mujoco>
"""


def build_open_box(name, *, w, d, h, t, base, rgb, rim) -> str:
    walls = _wall_ring(w, d, base, h, t, "box_geom")
    rim_lines = []
    if rim is not None:
        rim_lines = _wall_ring(
            w, d, h - 0.009, h, t, "box_print", _rgba(rim), outset=0.0001
        )
        rim_lines = ["", "      <!-- Coloured rim (visual only). -->"] + rim_lines

    return f"""<mujoco model="{name}">
  <!-- AUTO-GENERATED by build_objects.py. Do not edit by hand.
       {w*1000:.0f} x {d*1000:.0f} x {h*1000:.0f} mm open container: floor slab plus four
       {t*1000:.1f} mm walls. Origin at bottom center. -->

  <default>
    <default class="box_geom">
      <geom {CONTACT} density="{DENSITY_PP}" {_rgba(rgb)}/>
    </default>
    <default class="box_print">
      <geom contype="0" conaffinity="0" group="1" mass="0"/>
    </default>
  </default>

  <worldbody>
    <body name="root">
      <freejoint name="free"/>
      <geom name="floor" class="box_geom" type="box"
            pos="{_v(0, 0, base / 2)}" size="{_v(w / 2, d / 2, base / 2)}"/>

      <!-- Side walls. -->
{chr(10).join(walls)}{chr(10).join(rim_lines)}
    </body>
  </worldbody>
</mujoco>
"""


def build_sponge() -> str:
    total = SPONGE_FOAM_H + SPONGE_PAD_H
    return f"""<mujoco model="sponge">
  <!-- AUTO-GENERATED by build_objects.py. Do not edit by hand.
       {SPONGE_W*1000:.0f} x {SPONGE_D*1000:.0f} x {total*1000:.0f} mm dish sponge: foam block with an
       abrasive pad bonded on top. Modelled rigid; MuJoCo primitives
       don't deform, so this is a stand-in for a compliant object.
       Origin at bottom center. -->

  <default>
    <default class="sponge">
      <geom condim="4" friction="1.6 0.08 0.002" solref="0.02 1"/>
    </default>
  </default>

  <worldbody>
    <body name="root">
      <freejoint name="free"/>
      <geom name="foam" class="sponge" type="box"
            pos="{_v(0, 0, SPONGE_FOAM_H / 2)}"
            size="{_v(SPONGE_W / 2, SPONGE_D / 2, SPONGE_FOAM_H / 2)}"
            mass="0.012" rgba="0.98 0.85 0.25 1"/>
      <geom name="pad" class="sponge" type="box"
            pos="{_v(0, 0, SPONGE_FOAM_H + SPONGE_PAD_H / 2)}"
            size="{_v(SPONGE_W / 2, SPONGE_D / 2, SPONGE_PAD_H / 2)}"
            mass="0.005" rgba="0.15 0.45 0.22 1"/>
    </body>
  </worldbody>
</mujoco>
"""


BUILDERS = {
    "milk_carton": build_milk_carton,
    "pringles": build_pringles,
    "sponge": build_sponge,
    **{n: partial(build_cup, n, **p) for n, p in CUPS.items()},
    **{n: partial(build_open_box, n, **p) for n, p in BOXES.items()},
}


def main():
    for name, builder in sorted(BUILDERS.items()):
        out = ASSETS / name / f"{name}.xml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(builder())
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
