"""Build a temporary RBY1 + XHand scene from a validated object spec."""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import mujoco

from object_config import load_object_spec


REPO = Path(__file__).resolve().parents[3]
BASE_SCENE = REPO / "src/sim/mujoco_sim/scenes/rby1_xhand.xml"


def _numbers(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _primitive_size(primitive: str, dimensions: list[float]) -> list[float]:
    if primitive == "box":
        return [value / 2.0 for value in dimensions]
    if primitive == "sphere":
        return dimensions
    # MuJoCo cylinder/capsule size is radius + half-height.
    return [dimensions[0], dimensions[1] / 2.0]


def _remove_body(worldbody: ET.Element, name: str) -> bool:
    for parent in worldbody.iter("body"):
        for child in list(parent):
            if child.tag == "body" and child.get("name") == name:
                parent.remove(child)
                return True
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") == name:
            worldbody.remove(child)
            return True
    return False


def _remove_free_object_bodies(worldbody: ET.Element) -> int:
    """Remove top-level free bodies while preserving the fixed robot/table."""
    removed = 0
    for body in list(worldbody.findall("body")):
        has_freejoint = body.find("freejoint") is not None
        has_free_joint = any(
            joint.get("type") == "free"
            for joint in body.findall("joint")
        )
        if has_freejoint or has_free_joint:
            worldbody.remove(body)
            removed += 1
    return removed


def _absolutize_assets(root: ET.Element, base_dir: Path) -> None:
    for tag in ("mesh", "texture", "hfield"):
        for element in root.findall(f".//{tag}"):
            file_value = element.get("file")
            if not file_value:
                continue
            path = Path(file_value)
            if not path.is_absolute():
                element.set("file", str((base_dir / path).resolve()))


def _add_mesh_asset(
    asset: ET.Element,
    *,
    name: str,
    path: str,
    scale: list[float],
) -> str:
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": name,
            "file": str(Path(path).resolve()),
            "scale": _numbers(scale),
        },
    )
    return name


def _add_mjcf_object(
    root: ET.Element,
    worldbody: ET.Element,
    asset: ET.Element,
    *,
    mjcf_path: str,
    spawn: dict,
    physics: dict,
) -> None:
    """Merge one standalone MJCF object's defaults/assets/body."""
    source_path = Path(mjcf_path).resolve()
    source_tree = ET.parse(source_path)
    source_root = source_tree.getroot()
    _absolutize_assets(source_root, source_path.parent)

    source_asset = source_root.find("asset")
    if source_asset is not None:
        for child in source_asset:
            asset.append(deepcopy(child))

    source_default = source_root.find("default")
    if source_default is not None:
        target_default = root.find("default")
        if target_default is None:
            target_default = ET.Element("default")
            root.insert(0, target_default)
        for child in source_default:
            target_default.append(deepcopy(child))

    source_worldbody = source_root.find("worldbody")
    source_body = (
        source_worldbody.find("body")
        if source_worldbody is not None
        else None
    )
    if source_body is None:
        raise ValueError(f"{source_path} has no worldbody/body")
    body = deepcopy(source_body)
    body.set("name", "object_root")
    body.set("pos", _numbers(spawn["position"]))
    body.set(
        "quat",
        _numbers(
            [
                spawn["quaternion_xyzw"][3],
                *spawn["quaternion_xyzw"][:3],
            ]
        ),
    )
    for element in body.iter():
        if element is body:
            continue
        name = element.get("name")
        if name:
            element.set("name", f"object_{name}")
    freejoint = body.find("freejoint")
    if freejoint is None:
        freejoint = next(
            (
                joint
                for joint in body.findall("joint")
                if joint.get("type") == "free"
            ),
            None,
        )
    if freejoint is None:
        raise ValueError(f"{source_path} object body has no free joint")
    freejoint.set("name", "object_free")
    for geom in body.iter("geom"):
        geom.set("friction", _numbers(physics["friction"]))
    worldbody.append(body)


def build_object_scene(
    object_spec: str | Path,
    output_path: str | Path,
    *,
    base_scene: str | Path = BASE_SCENE,
) -> Path:
    """Build a scene with the object described by ``object_spec``."""
    spec = load_object_spec(object_spec, check_assets=True)
    base_scene = Path(base_scene).resolve()
    output_path = Path(output_path).resolve()

    tree = ET.parse(base_scene)
    root = tree.getroot()
    _absolutize_assets(root, base_scene.parent)
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    if worldbody is None or asset is None:
        raise ValueError(f"Invalid MuJoCo scene: {base_scene}")
    _remove_free_object_bodies(worldbody)
    _remove_body(worldbody, "object_root")

    geometry = spec["geometry"]
    physics = spec["physics"]
    spawn = spec["spawn"]
    if geometry.get("mjcf"):
        _add_mjcf_object(
            root,
            worldbody,
            asset,
            mjcf_path=geometry["mjcf"],
            spawn=spawn,
            physics=physics,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="unicode")
        return output_path

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "object_root",
            "pos": _numbers(spawn["position"]),
            # MuJoCo XML quaternion order is wxyz.
            "quat": _numbers(
                [
                    spawn["quaternion_xyzw"][3],
                    *spawn["quaternion_xyzw"][:3],
                ]
            ),
        },
    )
    ET.SubElement(body, "freejoint", {"name": "object_free"})

    common = {
        "friction": _numbers(physics["friction"]),
        "condim": "4",
    }
    primitive = geometry.get("primitive")
    visual_mesh = geometry.get("visual_mesh")
    collision_mesh = geometry.get("collision_mesh")

    if visual_mesh:
        visual_name = _add_mesh_asset(
            asset,
            name="object_visual_mesh",
            path=visual_mesh,
            scale=geometry["scale"],
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": "object_visual",
                "type": "mesh",
                "mesh": visual_name,
                "rgba": _numbers(geometry["rgba"]),
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
                "density": "0",
            },
        )

    if collision_mesh:
        collision_name = _add_mesh_asset(
            asset,
            name="object_collision_mesh",
            path=collision_mesh,
            scale=geometry["scale"],
        )
        ET.SubElement(
            body,
            "geom",
            {
                **common,
                "name": "object_collision",
                "type": "mesh",
                "mesh": collision_name,
                "mass": f"{physics['mass_kg']:.9g}",
                "rgba": "0 0 0 0",
                "group": "3",
            },
        )
    elif primitive:
        ET.SubElement(
            body,
            "geom",
            {
                **common,
                "name": "object_collision",
                "type": primitive,
                "size": _numbers(
                    _primitive_size(
                        primitive, geometry["dimensions_m"]
                    )
                ),
                "mass": f"{physics['mass_kg']:.9g}",
                "rgba": (
                    "0 0 0 0"
                    if visual_mesh
                    else _numbers(geometry["rgba"])
                ),
            },
        )
    elif visual_mesh:
        # A visual-only mesh is allowed as a convenience, but it must also
        # provide collision/inertia for the free body.
        visual_geom = body.find("geom")
        assert visual_geom is not None
        visual_geom.set("contype", "1")
        visual_geom.set("conaffinity", "1")
        visual_geom.set("density", "0")
        visual_geom.set("mass", f"{physics['mass_kg']:.9g}")
        visual_geom.set("friction", _numbers(physics["friction"]))
        visual_geom.set("condim", "4")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="unicode")
    return output_path


@dataclass
class TemporaryObjectScene:
    path: Path

    def cleanup(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def temporary_object_scene(
    object_spec: str | Path,
    *,
    base_scene: str | Path = BASE_SCENE,
) -> TemporaryObjectScene:
    handle = tempfile.NamedTemporaryFile(
        prefix=".rby1_xhand_object_",
        suffix=".xml",
        dir=Path(base_scene).resolve().parent,
        delete=False,
    )
    handle.close()
    path = Path(handle.name)
    try:
        build_object_scene(object_spec, path, base_scene=base_scene)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return TemporaryObjectScene(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_spec", required=True)
    parser.add_argument("--base_scene", default=str(BASE_SCENE))
    parser.add_argument("--out", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    path = build_object_scene(
        args.object_spec,
        args.out,
        base_scene=args.base_scene,
    )
    print(f"wrote {path}")
    if args.check:
        model = mujoco.MjModel.from_xml_path(str(path))
        object_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "object_root"
        )
        if object_id < 0:
            raise RuntimeError("generated scene has no object_root")
        print(
            f"OK nq={model.nq} nu={model.nu} "
            f"object_body_id={object_id}"
        )


if __name__ == "__main__":
    main()
