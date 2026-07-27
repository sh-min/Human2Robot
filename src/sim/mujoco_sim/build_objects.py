"""Generate tabletop manipulation objects as pure-primitive MJCF.

No mesh files: every object is assembled from MuJoCo boxes, cylinders and
capsules.

  milk_carton      55 x 55 x 115 mm gable-top carton (height includes the
                   sloped roof and the sealed fin on top)
  pringles         O65 x 90 mm Pringles can
  cup_green        O80 x 105 mm cup with a handle
  cup_blue         O75 x 85 mm cup with a handle
  lock_box_large   160 x 120 x 50 mm open food container, blue rim,
                   14 mm rounded corners
  lock_box_small   130 x 90 x 55 mm open food container, green rim,
                   12 mm rounded corners
  sponge           130 x 80 x 30 mm dish sponge with a scrub pad
  trash_bin        137 x 137 x 197 mm white open bin, square corners
  cup_holder       130 x 125 x 175 mm chrome wire cup/plate rack: four
                   arches on two base rails, plus two ball-tipped hooks

Cups are hollow (ring of boxes -- MuJoCo has no hollow primitive) and so
are the containers and the bin (floor slab + walls). Rounded corners are
quarter-arcs of the same box segments, with cylinders filling the floor's
corners.

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
# ``corner`` is the radius of the rounded vertical corners, 0 for a box with
# sharp ones. The two lock boxes are deliberately different colours so they
# are easy to tell apart in a render or a policy rollout.
BOXES = {
    "lock_box_large": dict(w=0.160, d=0.120, h=0.050, t=0.0025, base=0.004,
                           corner=0.014,
                           rgb=(0.86, 0.90, 0.94), rim=(0.10, 0.40, 0.78)),
    "lock_box_small": dict(w=0.130, d=0.090, h=0.055, t=0.0025, base=0.004,
                           corner=0.012,
                           rgb=(0.95, 0.93, 0.87), rim=(0.20, 0.60, 0.34)),
    "trash_bin": dict(w=0.137, d=0.137, h=0.197, t=0.002, base=0.003,
                      corner=0.0,
                      rgb=(0.95, 0.95, 0.95), rim=None),
}

# --- sponge ----------------------------------------------------------
SPONGE_W, SPONGE_D = 0.130, 0.080
SPONGE_FOAM_H = 0.022
SPONGE_PAD_H = 0.008  # -> 30 mm total

# --- chrome wire cup/plate rack --------------------------------------
# Built in layers, bottom up: two base rails -> two semicircle arches
# standing on them -> two side bars along the rails, riding the flanks of
# those semicircles -> two elongated arches standing on the side bars (the
# plate slot) -> two short bars across the semicircle crowns -> a post off
# each -> a rod -> a shepherd's-crook cup hook at each end.
RACK_DEPTH = 0.130        # 세로: rail length
RACK_RAIL_GAP = 0.125     # 가로: rail spacing
RACK_H = 0.175            # 높이: crest of the crook
# The four arches, spaced 45 / 45 / 40 mm -- exactly the 130 mm depth.
RACK_X_SEMI = (-0.065, -0.020)
RACK_X_ELONG = (+0.025, +0.065)
RACK_RAIL_R = 0.0035      # 발받침대 O7 mm
RACK_WIRE_R = 0.0015      # O3 mm wire
RACK_BALL_R = 0.0040
RACK_CAP_R = 0.0035       # black end cap, flush so the rail sits on z=0
RACK_CAP_L = 0.008
RACK_SEMI_R = RACK_RAIL_GAP / 2   # a foot on each rail -> a true semicircle
RACK_RAIL_TOP = 2 * RACK_RAIL_R   # everything stands on top of the rails
RACK_SIDE_T = math.radians(45)    # where the side bars ride the semicircle
RACK_ELONG_TOP = 0.135
RACK_MID_Y = 0.020        # the two short bars, near the centre line
RACK_ROD_X = -0.020       # posts and rod stand over the rear semicircle
RACK_ROD_Y = 0.062        # rod half-length before the crooks
RACK_SEGS = 10
# The crook leaves the rod going up, arcs forward over a crest and back
# down, then reverses curvature and flicks up to the ball. Two arcs turning
# opposite ways, so the whole thing is an S; the cup hangs in the descending
# bend and the upturned tip is what keeps it there.
RACK_CROOK_R = 0.020
RACK_TIP_R = 0.006
RACK_CROOK_SWEEP = math.radians(150)


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


def _rounded_ring(w, d, z_lo, z_hi, t, r, cls, rgba="", outset=0.0,
                  segs=4) -> list[str]:
    """Side walls of a rounded rectangle: four straight runs, each stopping r
    short of the ends, plus a quarter-arc of box segments at every corner.
    Same trick as the cup's ring wall -- MuJoCo has no rounded box."""
    hz = (z_hi - z_lo) / 2
    zc = (z_lo + z_hi) / 2
    hw, hd = w / 2 + outset, d / 2 + outset
    out = []
    for sx in (+1.0, -1.0):
        out.append(
            f'      <geom class="{cls}" type="box" {rgba} '
            f'pos="{_v(sx * (hw - t / 2), 0, zc)}" '
            f'size="{_v(t / 2, hd - r, hz)}"/>'
        )
    for sy in (+1.0, -1.0):
        out.append(
            f'      <geom class="{cls}" type="box" {rgba} '
            f'pos="{_v(0, sy * (hd - t / 2), zc)}" '
            f'size="{_v(hw - r, t / 2, hz)}"/>'
        )

    r_mid = r - t / 2
    # 2% tangential overlap closes the seams without notching the rim.
    chord = r_mid * math.tan(math.pi / (4 * segs)) * 1.02
    for sx in (+1.0, -1.0):
        for sy in (+1.0, -1.0):
            cx, cy = sx * (hw - r), sy * (hd - r)
            mid = math.atan2(sy, sx)          # +-45 or +-135 deg
            for i in range(segs):
                th = mid - math.pi / 4 + math.pi / 2 * (i + 0.5) / segs
                out.append(
                    f'      <geom class="{cls}" type="box" {rgba} '
                    f'pos="{_v(cx + r_mid * math.cos(th), cy + r_mid * math.sin(th), zc)}" '
                    f'quat="{_q(_quat_z(th))}" '
                    f'size="{_v(t / 2, chord, hz)}"/>'
                )
    return out


def _capsule(a, b, r, cls) -> str:
    return (f'      <geom class="{cls}" type="capsule" '
            f'fromto="{_v(*a, *b)}" size="{r:.5f}"/>')


def _wire_chain(pts, r, cls) -> list[str]:
    """Capsules through pts, skipping coincident pairs -- an arc often starts
    exactly on the endpoint of the segment feeding it."""
    return [_capsule(a, b, r, cls) for a, b in zip(pts, pts[1:])
            if math.dist(a, b) > 1e-6]


def _bend(start, heading, r, sweep, sign, segs):
    """Circular arc in the x-z plane at constant y, starting at ``start`` and
    travelling along ``heading``. ``sign`` is +1 to curve left of heading, -1
    to curve right. Returns the points after ``start``, plus the heading you
    leave on."""
    cx = start[0] - sign * r * math.sin(heading)
    cz = start[2] + sign * r * math.cos(heading)
    a0 = math.atan2(start[2] - cz, start[0] - cx)
    pts = [(cx + r * math.cos(a0 + sign * sweep * i / segs), start[1],
            cz + r * math.sin(a0 + sign * sweep * i / segs))
           for i in range(1, segs + 1)]
    return pts, heading + sign * sweep


def _rounded_slab(w, d, z_lo, z_hi, r, cls) -> list[str]:
    """Rounded-rectangle slab: a cross of two boxes plus a cylinder at each
    corner. The pieces overlap inside, which is what you want for a solid
    floor."""
    hz = (z_hi - z_lo) / 2
    zc = (z_lo + z_hi) / 2
    hw, hd = w / 2, d / 2
    out = [
        f'      <geom class="{cls}" type="box" '
        f'pos="{_v(0, 0, zc)}" size="{_v(hw, hd - r, hz)}"/>',
        f'      <geom class="{cls}" type="box" '
        f'pos="{_v(0, 0, zc)}" size="{_v(hw - r, hd, hz)}"/>',
    ]
    for sx in (+1.0, -1.0):
        for sy in (+1.0, -1.0):
            out.append(
                f'      <geom class="{cls}" type="cylinder" '
                f'pos="{_v(sx * (hw - r), sy * (hd - r), zc)}" '
                f'size="{_v(r, hz)}"/>'
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


def build_open_box(name, *, w, d, h, t, base, rgb, rim, corner=0.0) -> str:
    if corner > 0:
        floor = _rounded_slab(w, d, 0.0, base, corner, "box_geom")
        walls = _rounded_ring(w, d, base, h, t, corner, "box_geom")
    else:
        floor = [
            f'      <geom name="floor" class="box_geom" type="box" '
            f'pos="{_v(0, 0, base / 2)}" size="{_v(w / 2, d / 2, base / 2)}"/>'
        ]
        walls = _wall_ring(w, d, base, h, t, "box_geom")

    rim_lines = []
    if rim is not None:
        ring = (
            _rounded_ring(w, d, h - 0.009, h, t, corner, "box_print",
                          _rgba(rim), outset=0.0001)
            if corner > 0 else
            _wall_ring(w, d, h - 0.009, h, t, "box_print", _rgba(rim),
                       outset=0.0001)
        )
        rim_lines = ["", "      <!-- Coloured rim (visual only). -->"] + ring

    shape = (f"{corner*1000:.0f} mm rounded corners"
             if corner > 0 else "square corners")
    return f"""<mujoco model="{name}">
  <!-- AUTO-GENERATED by build_objects.py. Do not edit by hand.
       {w*1000:.0f} x {d*1000:.0f} x {h*1000:.0f} mm open container with {shape}:
       floor slab plus {t*1000:.1f} mm walls. Origin at bottom center. -->

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
      <!-- Floor. -->
{chr(10).join(floor)}

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


def build_cup_holder() -> str:
    half = RACK_SEMI_R
    side_y = half * math.cos(RACK_SIDE_T)
    side_z = RACK_RAIL_TOP + half * math.sin(RACK_SIDE_T)
    elong_shoulder = RACK_ELONG_TOP - side_y
    # The crest of the crook is the highest thing on the rack, so the rod
    # height falls out of the target 175 mm.
    rod_z = RACK_H - RACK_WIRE_R - RACK_CROOK_R

    out = ["      <!-- 1. Base rails with black end caps. -->"]
    for sy in (-1.0, +1.0):
        y = sy * half
        out.append(_capsule((-RACK_DEPTH / 2 + RACK_CAP_L, y, RACK_RAIL_R),
                            (+RACK_DEPTH / 2 - RACK_CAP_L, y, RACK_RAIL_R),
                            RACK_RAIL_R, "rack_rail"))
        for sx in (-1.0, +1.0):
            out.append(_capsule(
                (sx * (RACK_DEPTH / 2 - RACK_CAP_L), y, RACK_RAIL_R),
                (sx * (RACK_DEPTH / 2 - RACK_CAP_R), y, RACK_RAIL_R),
                RACK_CAP_R, "rack_cap"))

    out += ["", "      <!-- 2. Semicircle arches on the rails. -->"]
    for x in RACK_X_SEMI:
        out += _wire_chain(
            [(x, half * math.cos(t), RACK_RAIL_TOP + half * math.sin(t))
             for t in (math.pi * i / 18 for i in range(19))],
            RACK_WIRE_R, "rack")

    out += ["", "      <!-- 3. Side bars along the rails, on the "
                "semicircles' flanks. -->"]
    for sy in (-1.0, +1.0):
        out.append(_capsule((RACK_X_SEMI[0], sy * side_y, side_z),
                            (RACK_X_ELONG[-1], sy * side_y, side_z),
                            RACK_WIRE_R, "rack"))

    out += ["", "      <!-- 4. Elongated arches on the side bars: the plate "
                "slot. -->"]
    for x in RACK_X_ELONG:
        pts = [(x, -side_y, side_z), (x, -side_y, elong_shoulder)]
        for i in range(15):
            t = math.pi - math.pi * i / 14
            pts.append((x, side_y * math.cos(t),
                        elong_shoulder + side_y * math.sin(t)))
        pts += [(x, +side_y, elong_shoulder), (x, +side_y, side_z)]
        out += _wire_chain(pts, RACK_WIRE_R, "rack")

    out += ["", "      <!-- 5. Two short bars across the semicircle crowns, "
                "and a post off each. -->"]
    mid_z = RACK_RAIL_TOP + math.sqrt(half**2 - RACK_MID_Y**2)
    for sy in (-1.0, +1.0):
        y = sy * RACK_MID_Y
        out.append(_capsule((RACK_X_SEMI[0], y, mid_z),
                            (RACK_X_SEMI[-1], y, mid_z), RACK_WIRE_R, "rack"))
        out.append(_capsule((RACK_ROD_X, y, mid_z), (RACK_ROD_X, y, rod_z),
                            RACK_WIRE_R, "rack"))

    out += ["", "      <!-- 6-7. Rod, and a shepherd's-crook cup hook at "
                "each end. -->"]
    out.append(_capsule((RACK_ROD_X, -RACK_ROD_Y, rod_z),
                        (RACK_ROD_X, +RACK_ROD_Y, rod_z), RACK_WIRE_R, "rack"))
    for sy in (-1.0, +1.0):
        start = (RACK_ROD_X, sy * RACK_ROD_Y, rod_z)
        over, heading = _bend(start, math.pi / 2, RACK_CROOK_R,
                              RACK_CROOK_SWEEP, +1, RACK_SEGS)
        flick, _ = _bend(over[-1], heading, RACK_TIP_R, RACK_CROOK_SWEEP,
                         -1, RACK_SEGS)
        out += _wire_chain([start] + over + flick, RACK_WIRE_R, "rack")
        out.append(f'      <geom class="rack" type="sphere" '
                   f'pos="{_v(*flick[-1])}" size="{RACK_BALL_R:.5f}"/>')

    return f"""<mujoco model="cup_holder">
  <!-- AUTO-GENERATED by build_objects.py. Do not edit by hand.
       Chrome wire cup/plate rack, {RACK_DEPTH*1000:.0f} mm (세로) footprint x {RACK_RAIL_GAP*1000:.0f} mm
       (가로) rail spacing x {RACK_H*1000:.0f} mm (높이). Four arches -- two semicircles
       then two elongated ones forming the plate slot -- spaced 45 / 45 / 40
       mm along the rails. Ball-tipped shepherd's crooks hang the cups.
       Origin at bottom center. -->

  <asset>
    <material name="chrome" rgba="0.56 0.59 0.63 1"
              specular="1" shininess="1" reflectance="0.7"/>
    <material name="rack_endcap" rgba="0.10 0.10 0.11 1"
              specular="0.3" shininess="0.3"/>
  </asset>

  <default>
    <default class="rack">
      <geom {CONTACT} material="chrome" density="7800"/>
    </default>
    <default class="rack_rail">
      <geom {CONTACT} material="chrome" density="2000"/>
    </default>
    <default class="rack_cap">
      <geom {CONTACT} material="rack_endcap" density="1100"/>
    </default>
  </default>

  <worldbody>
    <body name="root">
      <freejoint name="free"/>
{chr(10).join(out)}
    </body>
  </worldbody>
</mujoco>
"""


BUILDERS = {
    "milk_carton": build_milk_carton,
    "pringles": build_pringles,
    "sponge": build_sponge,
    "cup_holder": build_cup_holder,
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
