"""Surface treatment for the STL-only robot hand meshes.

The xhand assets are 40 STL files. STL stores triangles and nothing else -- no
UVs, no materials, no vertex colours -- so there is no texture map to load. The
only colour information anywhere is the per-link ``rgba`` in the MuJoCo MJCF,
and for the left hand those are nearly all white (``1 1 1`` /
``0.87 0.87 0.89``). The render therefore comes out 99.8 % desaturated and
reads as one smooth blob.

Glossiness is not the cause: pyrender already derives metallicFactor 0.2 /
roughnessFactor 0.8 from the trimesh visual, which is matte. What is missing is
*tonal variation* -- the seams between shells, screw bosses and panel gaps are
present in the geometry but flat lighting gives them no shading.

Cavity shading recovers exactly that. A vertex whose neighbours sit in front of
its tangent plane is inside a crevice; darkening those makes existing geometry
legible. It is a per-vertex quantity, so the missing UVs do not matter, and it
multiplies the mesh's own colour, so a link meant to be orange stays orange.

An explicit pyrender material must NOT be used here: passing one replaces the
material derived from the mesh, which discards the MJCF link colours and the
RB5 arm's .dae texture and flattens everything to white.
"""
from __future__ import annotations

import numpy as np
import pyrender
import trimesh

SURFACE_MODES = ("default", "cavity")


def cavity(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex concavity in [0, 1]; 0 on flat/convex, 1 deep in a crevice.

    For vertex ``v`` with one-ring neighbours ``N``, the mean unit direction to
    the neighbours is compared against the vertex normal. A convex surface puts
    the neighbours behind the tangent plane (negative dot), a concave one in
    front.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0 or len(vertices) < 4:
        return np.zeros(len(vertices), dtype=np.float32)

    direction = np.zeros_like(vertices)
    count = np.zeros(len(vertices), dtype=np.float64)
    for a, b in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
        delta = vertices[b] - vertices[a]
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        np.divide(delta, length, out=delta, where=length > 1e-12)
        np.add.at(direction, a, delta)
        np.add.at(count, a, 1.0)

    valid = count > 0
    direction[valid] /= count[valid, None]
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    np.divide(direction, length, out=direction, where=length > 1e-12)
    return np.clip((normals * direction).sum(axis=1), 0.0, 1.0).astype(np.float32)


def bake_cavity(mesh: trimesh.Trimesh, strength: float = 0.55,
                gamma: float = 0.6) -> trimesh.Trimesh:
    """Multiply the mesh's own colour by its cavity term.

    *gamma* below 1 widens the darkened band away from the very bottom of each
    crease, which reads better at video resolution than a hairline.
    """
    visual = mesh.visual
    if not hasattr(visual, "vertex_colors"):
        # The RB5 arm ships as .dae with a texture; sample it to per-vertex
        # colour so cavity can modulate it like any other mesh.
        visual = visual.to_color()
        mesh.visual = visual
    shade = 1.0 - strength * np.power(cavity(mesh), gamma)
    colors = np.asarray(visual.vertex_colors, dtype=np.float32).copy()
    colors[:, :3] *= shade[:, None]
    mesh.visual.vertex_colors = np.clip(colors, 0, 255).astype(np.uint8)
    return mesh


def prepare(mesh: trimesh.Trimesh, mode: str, strength: float = 0.55):
    """Apply the one-off, pose-independent part of a surface mode."""
    if mode == "cavity":
        return bake_cavity(mesh.copy(), strength=strength)
    return mesh


def to_pyrender(mesh: trimesh.Trimesh, mode: str):
    """Build the pyrender mesh, always letting pyrender derive the material.

    Both modes take the same path; the difference lives in the vertex colours
    baked by :func:`prepare`. ``default`` therefore reproduces the previous
    output exactly.
    """
    return pyrender.Mesh.from_trimesh(mesh.copy(), smooth=False)
