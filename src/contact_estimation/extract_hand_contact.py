import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HACO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'third_party', 'HACO_RELEASE'))
sys.path.insert(0, HACO_DIR)
os.chdir(HACO_DIR)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import torch
import trimesh
import pyrender
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from lib.core.config import cfg, update_config

# The released demo archive stores two MANO downsampling matrices separately
# from the HACO checkpoint. This extractor only consumes the full 778-vertex
# contact output, so provide shape-correct placeholders when the optional
# matrices are absent. The unused 336/84-vertex outputs become zero while the
# full-resolution prediction remains unchanged.
for matrix_name, rows in (('V_regressor_336.npy', 336), ('V_regressor_84.npy', 84)):
    matrix_path = os.path.join(
        HACO_DIR, 'data', 'base_data', 'human_models', 'mano', matrix_name
    )
    if not os.path.exists(matrix_path):
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        np.save(matrix_path, np.zeros((rows, 778), dtype=np.float32))

from lib.models.model import HACO
from lib.utils.human_models import mano
from lib.utils.contact_utils import get_contact_thres
from lib.utils.preprocessing import augmentation_contact
from collections import defaultdict, deque


def remove_small_contact_components(contact_mask, faces, min_size=20):
    vertex_to_faces = defaultdict(list)
    for i, f in enumerate(faces):
        for v in f:
            vertex_to_faces[v].append(i)
    visited = np.zeros(len(contact_mask), dtype=bool)
    filtered_mask = np.zeros_like(contact_mask, dtype=bool)
    for v in range(len(contact_mask)):
        if visited[v] or not contact_mask[v]:
            continue
        queue = deque([v])
        component = []
        while queue:
            curr = queue.popleft()
            if visited[curr] or not contact_mask[curr]:
                continue
            visited[curr] = True
            component.append(curr)
            for f_idx in vertex_to_faces[curr]:
                for neighbor in faces[f_idx]:
                    if not visited[neighbor] and contact_mask[neighbor]:
                        queue.append(neighbor)
        if len(component) >= min_size:
            filtered_mask[component] = True
    return filtered_mask


def get_bbox_from_joints3d(joints_3d, img_shape, focal, pad_ratio=0.15):
    cx, cy = img_shape[1] / 2.0, img_shape[0] / 2.0
    u = focal * joints_3d[:, 0] / joints_3d[:, 2] + cx
    v = focal * joints_3d[:, 1] / joints_3d[:, 2] + cy
    x_min, y_min = u.min(), v.min()
    x_max, y_max = u.max(), v.max()
    pad = pad_ratio * max(x_max - x_min, y_max - y_min)
    x_min = max(0.0, x_min - pad)
    y_min = max(0.0, y_min - pad)
    x_max = min(float(img_shape[1]), x_max + pad)
    y_max = min(float(img_shape[0]), y_max + pad)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def build_contact_mesh(verts_3d, contact_mask, faces):
    mesh = trimesh.Trimesh(vertices=verts_3d.copy(), faces=faces, process=False)
    colors = np.full((len(verts_3d), 4), [130, 130, 130, 255], dtype=np.uint8)
    colors[contact_mask] = [0, 255, 0, 255]
    mesh.visual.vertex_colors = colors
    return mesh


def render_hand(mesh, img_res=400, flip_front_back=True):
    centered = mesh.copy()
    centered.apply_translation(-mesh.vertices.mean(axis=0))
    if flip_front_back:
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
        centered.apply_transform(rot)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.3, 0.3, 0.3))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0.0, 0.0, 2.5]
    scene.add(
        pyrender.camera.IntrinsicsCamera(fx=5000, fy=5000, cx=img_res / 2, cy=img_res / 2),
        pose=camera_pose,
    )

    light_pose = np.eye(4)
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
        light_pose[:3, 3] = np.array(lp)
        scene.add(light, pose=light_pose.copy())

    # Keep the per-vertex colors assigned by build_contact_mesh: gray for the
    # full hand and green for HACO contact vertices. Supplying a single
    # material here would override those colors and hide the contact result.
    scene.add(pyrender.Mesh.from_trimesh(centered, smooth=False))

    r = pyrender.OffscreenRenderer(img_res, img_res)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return color[::-1, ::-1, :3]


def make_viz(rgb_img, side_renders, img_res=400):
    # Resize RGB to match img_res height
    h, w = rgb_img.shape[:2]
    rgb_resized = cv2.resize(rgb_img, (int(w * img_res / h), img_res))

    blank = np.full((img_res, img_res, 3), 240, dtype=np.uint8)
    panels = [rgb_resized]

    for side in ('Left', 'Right'):
        if side in side_renders:
            panel = side_renders[side].copy()
        else:
            panel = blank.copy()
        cv2.putText(panel, side, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        panels.append(panel)

    return np.hstack(panels)


def predict_contact(
    model,
    orig_img,
    bbox,
    is_right,
    use_vit_norm,
    normalize,
    device,
    eval_thres,
):
    crop_img, _, _, _, _, _ = augmentation_contact(
        orig_img.copy(), bbox, 'test', enforce_flip=(not is_right)
    )

    if use_vit_norm:
        img_tensor = crop_img.transpose(2, 0, 1) / 255.0
        img_tensor = normalize(torch.from_numpy(img_tensor).float())
    else:
        from torchvision import transforms
        img_tensor = transforms.ToTensor()(crop_img.astype(np.float32) / 255.0)

    with torch.no_grad():
        outputs = model({'input': {'image': img_tensor[None].to(device)}}, mode='test')

    contact_probability = (
        outputs['contact_out'].sigmoid()[0].detach().cpu().numpy().reshape(-1)
    )
    if contact_probability.shape != (778,):
        raise RuntimeError(
            "HACO full-resolution contact output must contain 778 MANO "
            f"vertices, got {contact_probability.shape}"
        )
    contact_mask = contact_probability > eval_thres
    contact_mask = remove_small_contact_components(
        contact_mask, faces=mano.watertight_face['right'], min_size=20
    )
    return contact_probability, contact_mask


def discover_images(input_dir, img_glob):
    """Return one deterministic source image per HaWoR sequence frame.

    Some datasets contain both ``000001.jpg`` and ``000001.png``.  Loading
    every image extension silently doubles the sequence and shifts all HaWoR
    and HACO results.  An explicit glob plus a duplicate-stem check makes that
    failure impossible.
    """
    images = sorted(Path(input_dir, 'rgb').glob(img_glob))
    images = [
        path for path in images
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]
    if not images:
        raise FileNotFoundError(
            f"no RGB images matched {img_glob!r} in {Path(input_dir, 'rgb')}"
        )
    stems = [path.stem for path in images]
    duplicates = sorted({
        stem for stem in stems if stems.count(stem) > 1
    })
    if duplicates:
        raise ValueError(
            "multiple source images share a frame stem; choose a single "
            f"extension with --img_glob (examples: {duplicates[:5]})"
        )
    return images


def main():
    parser = argparse.ArgumentParser(description='Extract hand contact vertices')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Episode directory containing rgb/ and rgb_hawor/retarget_input.npz')
    parser.add_argument(
        '--img_glob',
        default='*.jpg',
        help="Select exactly one source image per frame (default: '*.jpg')",
    )
    parser.add_argument(
        '--img_focal',
        type=float,
        default=None,
        help='Camera focal length in pixels. Default: retarget_input.npz img_focal',
    )
    parser.add_argument('--backbone', type=str, default='hamer',
                        choices=['hamer', 'vit-l-16', 'vit-b-16', 'vit-s-16',
                                 'handoccnet', 'hrnet-w48', 'hrnet-w32',
                                 'resnet-152', 'resnet-101', 'resnet-50', 'resnet-34', 'resnet-18'])
    parser.add_argument('--checkpoint', type=str, default=os.path.join(HACO_DIR, 'release_checkpoint', 'haco_neurips_hamer_checkpoint.ckpt'))
    parser.add_argument(
        '--no_viz',
        action='store_true',
        help='Skip the per-frame contact visualization PNGs',
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    rgb_dir = Path(input_dir, 'rgb')
    output_dir = os.path.join(input_dir, 'contact')
    viz_dir = os.path.join(output_dir, 'viz')
    os.makedirs(output_dir, exist_ok=True)
    if not args.no_viz:
        os.makedirs(viz_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    update_config(backbone_type=args.backbone, exp_dir='experiments_demo_image')

    # The released HACO checkpoint already contains the complete HaMeR
    # backbone. Reuse it for the constructor's initial backbone load so the
    # separate multi-gigabyte demo base_data archive is not required.
    if args.backbone == 'hamer':
        cfg.MODEL.hamer_backbone_pretrained_path = args.checkpoint

    model = HACO().to(device)
    if args.checkpoint:
        # Keep the second checkpoint copy off the GPU while loading the full
        # model state. HACO's HaMeR constructor already reads its backbone
        # checkpoint once, so retaining another CUDA state_dict can cause an
        # avoidable multi-gigabyte memory spike.
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
    model.eval()

    hawor = np.load(os.path.join(input_dir, 'rgb_hawor', 'retarget_input.npz'))
    hawor_verts = {'left': hawor['verts_left'], 'right': hawor['verts_right']}
    hawor_joints = {'left': hawor['joints_left'], 'right': hawor['joints_right']}
    hawor_valid = {'left': hawor['valid'][0], 'right': hawor['valid'][1]}
    hawor_start = int(hawor['start_idx'])
    hawor_end = int(hawor['end_idx'])
    img_focal = (
        float(args.img_focal)
        if args.img_focal is not None
        else float(hawor['img_focal'])
    )
    if not np.isfinite(img_focal) or img_focal <= 0:
        raise ValueError(f'invalid camera focal length: {img_focal}')

    images = discover_images(input_dir, args.img_glob)
    expected_frames = hawor_end - hawor_start
    if len(images) != expected_frames:
        raise ValueError(
            "source/HaWoR frame count mismatch: "
            f"{len(images)} images vs {expected_frames} HaWoR frames"
        )
    eval_thres = get_contact_thres(args.backbone)
    faces = mano.watertight_face['right']

    use_vit_norm = (args.backbone == 'hamer') or ('vit' in args.backbone)
    normalize = None
    if use_vit_norm:
        from torchvision.transforms import Normalize
        normalize = Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)

    for image_idx, frame_path in enumerate(tqdm(images)):
        frame_key = frame_path.stem
        # HaWoR indexes the sorted input sequence, not the numeric portion of
        # the filename.  Some recordings are named 000000.jpg while others
        # start at 000001.jpg, so parsing the stem shifts every prediction by
        # one frame and drops the final frame for the latter recordings.
        # ``images`` already contains exactly ``end_idx - start_idx`` frames,
        # while every HaWoR array is stored densely from zero.  ``start_idx``
        # describes the source-video slice and must not be subtracted a second
        # time here.
        frame_idx = image_idx

        if frame_idx < 0 or frame_idx >= hawor_valid['left'].shape[0]:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f'failed to read source image {frame_path}')
        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        save_dict = {
            'source_filename': np.str_(frame_path.name),
            'hawor_frame_index': np.int64(frame_idx),
            'img_focal': np.float32(img_focal),
            'contact_threshold': np.float32(eval_thres),
        }
        side_renders = {}

        for side, is_right in [('left', False), ('right', True)]:
            is_valid = bool(hawor_valid[side][frame_idx])
            save_dict[f'{side}_valid'] = np.bool_(is_valid)
            if not is_valid:
                save_dict[f'{side}_contact_mask'] = np.zeros(778, dtype=bool)
                save_dict[f'{side}_contact_probability'] = np.zeros(
                    778,
                    dtype=np.float16,
                )
                save_dict[f'{side}_contact_indices'] = np.empty(
                    0,
                    dtype=np.int64,
                )
                save_dict[f'{side}_contact_verts_3d'] = np.empty(
                    (0, 3),
                    dtype=np.float32,
                )
                continue

            bbox = get_bbox_from_joints3d(
                hawor_joints[side][frame_idx],
                orig_img.shape,
                img_focal,
            )
            contact_probability, contact_mask = predict_contact(
                model, orig_img, bbox, is_right, use_vit_norm, normalize, device, eval_thres
            )

            verts_3d = hawor_verts[side][frame_idx]
            contact_indices = np.where(contact_mask)[0]
            contact_verts_3d = verts_3d[contact_indices]

            save_dict[f'{side}_contact_mask'] = contact_mask
            save_dict[f'{side}_contact_probability'] = (
                contact_probability.astype(np.float16)
            )
            save_dict[f'{side}_contact_indices'] = contact_indices
            save_dict[f'{side}_contact_verts_3d'] = contact_verts_3d

            if not args.no_viz:
                mesh = build_contact_mesh(verts_3d, contact_mask, faces)
                side_renders[side.capitalize()] = render_hand(mesh)

        np.savez(os.path.join(output_dir, f'{frame_key}.npz'), **save_dict)

        if not args.no_viz:
            viz = make_viz(orig_img, side_renders)
            cv2.imwrite(
                os.path.join(viz_dir, f'{frame_key}.png'),
                cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
            )

    print(
        f"Done. Saved to {output_dir} "
        f"(frames={len(images)}, focal={img_focal:.2f}, glob={args.img_glob!r})"
    )


if __name__ == '__main__':
    main()
