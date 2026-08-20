from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "segment_sh_choco_mesh_pilot.py"
SPEC = importlib.util.spec_from_file_location("segment_sh_choco_mesh_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(path: Path, **metadata: object) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": MODULE.sha256_file(path),
        **metadata,
    }


def _write_bound_manifest(root: Path, image_path: Path) -> Path:
    source_path = root / "source_sh.jpg"
    source_path.write_bytes(image_path.read_bytes())
    document = {
        "schema_version": 1,
        "kind": "mesh_sota_pilot_input_bundle",
        "bundle": {
            "output_root": str(root.resolve()),
            "model_inference_performed": False,
        },
        "selection": {
            "object_label": "Choco",
            "mh_frame_index": 187,
            "sh_frame_index": 192,
            "mh_role": "primary/final",
            "sh_role": "auxiliary/evidence",
        },
        "camera_namespace": {
            "primary_view": "MH",
            "auxiliary_view": "SH",
            "stereo_code_mapping": {"camera_1": "SH", "camera_2": "MH"},
        },
        "image_geometry": {"width": 4, "height": 3},
        "outputs": {"sh_image": _record(image_path)},
        "sources": {
            "sh_image": _record(
                source_path,
                view="SH",
                pipeline_camera="camera_1",
                frame_index=192,
            )
        },
        "pixel_provenance": {},
        "invariants": {},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class SegmentSHChocoTests(unittest.TestCase):
    def test_manifest_binding_rejects_a_different_supplied_sh_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "sh_expected.jpg"
            supplied = root / "sh_other.jpg"
            expected.write_bytes(b"expected SH frame bytes")
            supplied.write_bytes(b"different SH frame bytes")
            manifest = _write_bound_manifest(root, expected)

            with self.assertRaisesRegex(
                ValueError, "does not match manifest outputs.sh_image.path"
            ):
                MODULE.load_and_bind_manifest(manifest, supplied)

    def test_manifest_binding_rejects_stale_bytes_and_sha_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sh.jpg"
            image.write_bytes(b"original bound frame")
            manifest = _write_bound_manifest(root, image)
            # Keep the same path but change the underlying input after the
            # manifest was generated.
            image.write_bytes(b"tampered frame with a different byte count")

            with self.assertRaisesRegex(
                ValueError, "byte size does not match manifest outputs.sh_image"
            ):
                MODULE.load_and_bind_manifest(manifest, image)

    def test_publication_paths_reject_collisions_and_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bundle = base / "bundle"
            bundle.mkdir()
            manifest = bundle / "manifest.png"
            image = bundle / "sh_input.png"
            checkpoint = bundle / "checkpoint.png"
            repository = bundle / "sam2"
            repository.mkdir()
            for path in (manifest, image, checkpoint):
                path.write_bytes(b"protected")
            safe_mask = bundle / "mask.png"
            safe_overlay = bundle / "overlay.png"

            attacks = (
                (safe_mask, safe_mask, "must be distinct"),
                (base / "escaped.png", safe_overlay, "inside declared bundle root"),
                (image, safe_overlay, "protected SH input image"),
                (checkpoint, safe_overlay, "protected SAM2 checkpoint"),
                (manifest, safe_overlay, "protected manifest"),
                (
                    repository / "repo_output.png",
                    safe_overlay,
                    "protected SAM2 repository",
                ),
            )
            for mask, overlay, message in attacks:
                with self.subTest(mask=mask, overlay=overlay):
                    with self.assertRaisesRegex(ValueError, message):
                        MODULE.validate_publication_paths(
                            bundle_root=bundle,
                            mask_path=mask,
                            overlay_path=overlay,
                            image_path=image,
                            checkpoint_path=checkpoint,
                            sam2_root=repository,
                            manifest_path=manifest,
                        )

            linked_output = bundle / "linked_mask.png"
            linked_output.symlink_to(image)
            with self.assertRaisesRegex(ValueError, "must not be a symbolic link"):
                MODULE.validate_publication_paths(
                    bundle_root=bundle,
                    mask_path=Path(linked_output.name),
                    overlay_path=safe_overlay,
                    image_path=image,
                    checkpoint_path=checkpoint,
                    sam2_root=repository,
                    manifest_path=manifest,
                )

            hardlinked_output = bundle / "hardlinked_mask.png"
            MODULE.os.link(image, hardlinked_output)
            with self.assertRaisesRegex(ValueError, "protected SH input image"):
                MODULE.validate_publication_paths(
                    bundle_root=bundle,
                    mask_path=hardlinked_output,
                    overlay_path=safe_overlay,
                    image_path=image,
                    checkpoint_path=checkpoint,
                    sam2_root=repository,
                    manifest_path=manifest,
                )

            resolved = MODULE.validate_publication_paths(
                bundle_root=bundle,
                mask_path=safe_mask,
                overlay_path=safe_overlay,
                image_path=image,
                checkpoint_path=checkpoint,
                sam2_root=repository,
                manifest_path=manifest,
            )
            self.assertEqual(resolved, (safe_mask.resolve(), safe_overlay.resolve()))

    def test_failed_manifest_publication_rolls_back_both_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            final = root / "bundle"
            stage.mkdir()
            final.mkdir()
            names = ("mask.png", "overlay.png", "manifest.json")
            staged = []
            published = []
            for name in names:
                staged_path = stage / name
                final_path = final / name
                staged_path.write_bytes(f"new-{name}".encode())
                final_path.write_bytes(f"old-{name}".encode())
                staged.append(staged_path)
                published.append(final_path)

            call_count = 0

            def fail_on_manifest_replace(source: Path, destination: Path) -> None:
                nonlocal call_count
                call_count += 1
                # Each existing target causes backup then publication. The
                # sixth call is the final staged-manifest publication.
                if call_count == 6:
                    raise OSError("injected manifest publication failure")
                MODULE.os.replace(source, destination)

            with self.assertRaisesRegex(RuntimeError, "was rolled back"):
                MODULE.publish_staged_transaction(
                    tuple(zip(staged, published)),
                    replace_fn=fail_on_manifest_replace,
                )

            for name, path in zip(names, published):
                self.assertEqual(path.read_bytes(), f"old-{name}".encode())

    def test_validate_box_rejects_invalid_or_out_of_bounds_coordinates(self) -> None:
        self.assertEqual(
            MODULE.validate_box([1, 2, 8, 9], width=10, height=10),
            (1, 2, 8, 9),
        )
        with self.assertRaisesRegex(ValueError, "outside image"):
            MODULE.validate_box([1, 2, 11, 9], width=10, height=10)
        with self.assertRaisesRegex(ValueError, "outside image"):
            MODULE.validate_box([3, 2, 3, 9], width=10, height=10)

    def test_candidate_selection_defaults_to_highest_score_and_returns_binary(self) -> None:
        masks = np.zeros((3, 6, 7), dtype=np.float32)
        masks[0, 1:3, 1:3] = 1
        masks[1, 2:5, 2:6] = 1
        masks[2, 0:2, 4:7] = 1
        selected, index, score = MODULE.select_mask_candidate(
            masks, np.array([0.7, 0.95, 0.8])
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(score, 0.95)
        self.assertEqual(selected.dtype, np.bool_)
        self.assertEqual(int(selected.sum()), 12)

    def test_manifest_extension_is_explicitly_inferred_and_preserves_existing_data(self) -> None:
        manifest = {
            "selection": {"sh_frame_index": 192, "object_label": "Choco"},
            "outputs": {"mh_image": {"path": "/keep/mh.jpg"}},
            "pixel_provenance": {
                "inferred_pixels_used": 0,
                "statement": "Existing RGB provenance stays intact.",
            },
            "bundle": {"model_inference_performed": False},
            "invariants": {"uses_inferred_or_inpainted_pixels": False},
        }
        original = copy.deepcopy(manifest)
        metrics = {"foreground_pixels": 123, "largest_component_fraction": 1.0}
        updated = MODULE.update_manifest_with_sh_mask(
            manifest,
            image_record={"path": "/input/sh.jpg", "frame_index": 192},
            mask_record={"path": "/output/mask.png", "frame_index": 192},
            overlay_record={"path": "/output/overlay.png", "frame_index": 192},
            checkpoint_record={"path": "/weights/sam2.pt", "sha256": "abc"},
            sam2_root=Path("/vendor/sam2"),
            config_name="sam2_hiera_l.yaml",
            prompt_box=[590, 230, 690, 390],
            candidate_scores=[0.8, 0.9, 0.7],
            selected_index=1,
            selected_score=0.9,
            selection_policy="maximum_predicted_iou_score",
            selected_metrics=metrics,
            candidate_metrics=[metrics, metrics, metrics],
        )
        self.assertEqual(updated["outputs"]["mh_image"], original["outputs"]["mh_image"])
        self.assertEqual(updated["outputs"]["sh_modal_mask"]["path"], "/output/mask.png")
        provenance = updated["pixel_provenance"]["sh_modal_mask"]
        self.assertIs(provenance["is_model_inferred"], True)
        self.assertIs(provenance["is_ground_truth"], False)
        self.assertIs(provenance["is_human_annotated"], False)
        self.assertEqual(provenance["prompt"]["box_xyxy_exclusive"], [590, 230, 690, 390])
        self.assertEqual(updated["pixel_provenance"]["inferred_pixels_used"], 0)
        self.assertIs(updated["bundle"]["model_inference_performed"], True)
        self.assertIs(updated["invariants"]["sh_modal_mask_is_ground_truth"], False)

    def test_manifest_extension_rejects_frame_mismatch(self) -> None:
        manifest = {
            "selection": {"sh_frame_index": 192},
            "outputs": {},
            "pixel_provenance": {},
            "bundle": {},
            "invariants": {},
        }
        with self.assertRaisesRegex(ValueError, "mask record frame index"):
            MODULE.update_manifest_with_sh_mask(
            manifest,
            image_record={"path": "/input/sh.jpg", "frame_index": 192},
            mask_record={"path": "/output/mask.png", "frame_index": 191},
            overlay_record={"path": "/output/overlay.png", "frame_index": 192},
            checkpoint_record={"path": "/weights/sam2.pt"},
            sam2_root=Path("/vendor/sam2"),
            config_name="sam2_hiera_l.yaml",
            prompt_box=[590, 230, 690, 390],
            candidate_scores=[0.9],
            selected_index=0,
            selected_score=0.9,
            selection_policy="maximum_predicted_iou_score",
            selected_metrics={"foreground_pixels": 1},
            candidate_metrics=[{"foreground_pixels": 1}],
            )


if __name__ == "__main__":
    unittest.main()
