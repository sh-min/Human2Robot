import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "prepare_0805_stereo_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_0805", MODULE_PATH)
prepare_0805 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_0805)


def write_gt(path: Path, *, label: str = "Choco", frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "episode": path.parent.name,
                "num_frames": frames,
                "fps": 24.0,
                "segments": [
                    {"start_frame": 0, "end_frame": frames - 1, "label": label}
                ],
            }
        ),
        encoding="utf-8",
    )


class Prepare0805DatasetTests(unittest.TestCase):
    def test_real_sources_pair_in_annotation_mh_sh_natural_order(self):
        pairs = prepare_0805.discover_episode_sources(prepare_0805.DEFAULT_SOURCE)
        self.assertEqual([pair["episode"] for pair in pairs], ["1", "2"])
        self.assertEqual(
            [(pair["MH"].name, pair["SH"].name) for pair in pairs],
            [
                ("A001_08051547_C056.mov", "Sh001_08051547_C055.mov"),
                ("A001_08051547_C057.mov", "Sh001_08051547_C056.mov"),
            ],
        )

    def test_choco_is_supported_but_old_milk_label_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            choco = root / "1" / "gt_labels.json"
            milk = root / "2" / "gt_labels.json"
            write_gt(choco, label="Choco")
            write_gt(milk, label="Milk")
            self.assertEqual(
                prepare_0805.load_and_validate_gt(choco)["segments"][0]["label"],
                "Choco",
            )
            with self.assertRaisesRegex(ValueError, "unknown label 'Milk'"):
                prepare_0805.load_and_validate_gt(milk)

    def test_calibration_namespace_is_mapped_back_to_physical_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs_root = root / "manual_pairs"
            pairs_root.mkdir()
            (pairs_root / "pairs.json").write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "camera_1": {"video": str(root / "calibration_mh.mov")},
                                "camera_2": {"video": str(root / "calibration_sh.mov")},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source": {
                            "path": str(pairs_root),
                            "image_size_wh": [1280, 720],
                        },
                        "checkerboard": {
                            "metric_scale_verified": False,
                            "length_unit": "checker_square",
                        },
                        "quality": {"status": "review", "limitations": ["test"]},
                        "camera_1": {
                            "camera_matrix": [[1030.0, 0, 638], [0, 1031.0, 363], [0, 0, 1]],
                            "distortion_k1_k2_p1_p2_k3": [0, 0, 0, 0, 0],
                        },
                        "camera_2": {
                            "camera_matrix": [[1070.0, 0, 626], [0, 1071.0, 272], [0, 0, 1]],
                            "distortion_k1_k2_p1_p2_k3": [0, 0, 0, 0, 0],
                        },
                        "stereo": {
                            "T_camera2_from_camera1": [[1, 0, 0, 1]],
                            "translation_unit": "checker_square",
                        },
                    }
                ),
                encoding="utf-8",
            )

            metadata = prepare_0805.load_calibration_metadata(calibration_path)

            self.assertEqual(metadata["calibration_camera_mapping"], {
                "camera_1": "MH",
                "camera_2": "SH",
            })
            self.assertEqual(metadata["pipeline_to_calibration_camera"], {
                "camera_1": "camera_2",
                "camera_2": "camera_1",
            })
            self.assertEqual(metadata["intrinsics_by_view"]["MH"]["fx_px"], 1030.0)
            self.assertEqual(metadata["intrinsics_by_view"]["SH"]["fx_px"], 1070.0)
            self.assertEqual(
                Path(metadata["calibration_source_videos"]["MH"]).name,
                "calibration_mh.mov",
            )
            self.assertEqual(metadata["reference_json"], str(calibration_path.resolve()))
            self.assertEqual(len(metadata["reference_sha256"]), 64)

    def test_motion_offset_accepts_clear_peak_and_fails_open_on_flat_trace(self):
        mh = [0.0, 8.0, 1.0, 15.0, 3.0, 22.0, 2.0, 11.0, 4.0, 19.0]
        sh = [90.0, 80.0] + mh + [70.0]
        clear = prepare_0805.estimate_motion_offset(
            mh,
            sh,
            max_offset=4,
            min_correlation=0.5,
            min_peak_prominence=0.04,
        )
        self.assertEqual(clear["status"], "accepted")
        self.assertEqual(clear["selected_camera1_frame_offset"], 2)

        ambiguous = prepare_0805.estimate_motion_offset(
            [1.0] * 10,
            [1.0] * 10,
            max_offset=3,
        )
        self.assertEqual(ambiguous["status"], "ambiguous_fail_open")
        self.assertEqual(ambiguous["selected_camera1_frame_offset"], 0)

    def test_prepare_layout_keeps_sh_camera1_and_mh_camera2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            mh = source / "mh" / "mh.mov"
            sh = source / "sh" / "sh.mov"
            gt = source / "annotations" / "1" / "gt_labels.json"
            mh.parent.mkdir(parents=True)
            sh.parent.mkdir(parents=True)
            mh.write_bytes(b"mh")
            sh.write_bytes(b"sh")
            write_gt(gt)
            source_pair = {
                "episode": "1",
                "order_index": 1,
                "MH": mh,
                "SH": sh,
                "gt_labels": gt,
            }
            calibration = prepare_0805.load_calibration_metadata(None)
            temporal = {
                "status": "accepted",
                "estimated_camera1_frame_offset": 2,
                "selected_camera1_frame_offset": 2,
                "reason": "test",
                "out_of_range_policy": "fail_open",
            }

            def fake_extract(source_path, destination, expected):
                self.assertEqual(expected, 3)
                destination.mkdir(parents=True)

            with mock.patch.object(
                prepare_0805, "probe_frame_count", side_effect=[3, 4]
            ), mock.patch.object(
                prepare_0805, "extract_frames", side_effect=fake_extract
            ) as extract:
                manifest = prepare_0805.prepare_episode(
                    output, source_pair, calibration, temporal
                )

            self.assertEqual(extract.call_args_list[0].args[0], sh)
            self.assertEqual(extract.call_args_list[0].args[1], output / "1/camera_1/rgb")
            self.assertEqual(extract.call_args_list[1].args[0], mh)
            self.assertEqual(extract.call_args_list[1].args[1], output / "1/camera_2/rgb")
            self.assertEqual(manifest["stereo_code_mapping"], {
                "camera_1": "SH",
                "camera_2": "MH",
            })
            self.assertEqual(manifest["label_vocabulary"], list(prepare_0805.LABELS))
            self.assertEqual(manifest["temporal_alignment"]["camera1_frame_offset"], 2)
            self.assertEqual(manifest["tail_frames_dropped"], {"MH": 0, "SH": 1})
            saved = json.loads((output / "1/stereo_manifest.json").read_text())
            self.assertEqual(saved["sources"]["SH"], str(sh.resolve()))
            self.assertEqual((output / "1/camera_1/source.mov").resolve(), sh.resolve())
            self.assertEqual((output / "1/camera_2/source.mov").resolve(), mh.resolve())


if __name__ == "__main__":
    unittest.main()
