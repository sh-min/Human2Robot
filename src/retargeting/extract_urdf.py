"""Extract right/left hand subtree from star1 full-body URDF and emit
standalone xhand_right.urdf / xhand_left.urdf for dex-retargeting.

Usage:
    conda activate RFM_retarget
    python extract_urdf.py
"""
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

# Source: STAR1 full-body URDF/meshes (not in this repo; edit these paths to
# point at your local copy if you want to regenerate).
SRC_URDF = "/home/bg/Dockershared/RFM/retarget/models/star1/urdf/l3_with_hand_fixedpin_xml.urdf"
SRC_MESH_DIR = "/home/bg/Dockershared/RFM/retarget/models/star1/meshes"
# Destination: this repo's vendored xhand assets.
DST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "xhand")


def collect_subtree_links(joints, root_link):
    children = defaultdict(list)
    for j in joints:
        p = j.find("parent").attrib["link"]
        c = j.find("child").attrib["link"]
        children[p].append(c)

    reachable = {root_link}
    stack = [root_link]
    while stack:
        x = stack.pop()
        for c in children[x]:
            if c not in reachable:
                reachable.add(c)
                stack.append(c)
    return reachable


def extract_one(root_link, out_urdf_name, robot_name):
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()
    links = root.findall("link")
    joints = root.findall("joint")

    reachable = collect_subtree_links(joints, root_link)
    print(f"[{robot_name}] reachable links: {len(reachable)}")

    new_robot = ET.Element("robot", attrib={"name": robot_name})
    used_meshes = set()

    for l in links:
        if l.attrib["name"] not in reachable:
            continue
        # rewrite mesh paths to ./meshes/<basename>
        for m in l.iter("mesh"):
            fn = m.attrib.get("filename", "")
            bn = os.path.basename(fn)
            used_meshes.add(bn)
            m.attrib["filename"] = f"meshes/{bn}"
        new_robot.append(l)

    for j in joints:
        c = j.find("child").attrib["link"]
        p = j.find("parent").attrib["link"]
        if c in reachable and p in reachable:
            new_robot.append(j)

    ET.indent(new_robot, space="  ")
    out_path = os.path.join(DST_DIR, out_urdf_name)
    ET.ElementTree(new_robot).write(out_path, encoding="utf-8", xml_declaration=False)
    print(f"[{robot_name}] wrote {out_path}")

    # Copy meshes
    dst_mesh_dir = os.path.join(DST_DIR, "meshes")
    os.makedirs(dst_mesh_dir, exist_ok=True)
    copied = 0
    for bn in used_meshes:
        src = os.path.join(SRC_MESH_DIR, bn)
        dst = os.path.join(dst_mesh_dir, bn)
        if not os.path.exists(src):
            print(f"  [warn] missing source mesh: {src}")
            continue
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            copied += 1
    print(f"[{robot_name}] copied {copied}/{len(used_meshes)} meshes "
          f"(total now {len(os.listdir(dst_mesh_dir))})")


if __name__ == "__main__":
    os.makedirs(DST_DIR, exist_ok=True)
    extract_one("right_hand_base_link", "xhand_right.urdf", "xhand_right")
    extract_one("left_hand_base_link", "xhand_left.urdf", "xhand_left")
    print("\nDone.")
