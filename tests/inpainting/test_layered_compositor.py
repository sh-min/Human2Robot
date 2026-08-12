from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


INPAINTING_DIR = Path(__file__).resolve().parents[2] / "src" / "inpainting"
sys.path.insert(0, str(INPAINTING_DIR))

from layered_compositor import (  # noqa: E402
    STAGE_SPECS,
    FrameInputs,
    StageConfig,
    build_layer_masks,
    compose_frame,
)
from layered_compositor.video import CompatibleVideoWriter  # noqa: E402


def frame_inputs(shape: tuple[int, int] = (3, 3)) -> FrameInputs:
    height, width = shape
    image_shape = (height, width, 3)
    return FrameInputs(
        background=np.full(image_shape, 10, dtype=np.uint8),
        robot_rgb=np.full(image_shape, 100, dtype=np.uint8),
        object_rgb=np.full(image_shape, 200, dtype=np.uint8),
        robot_mask=np.zeros(shape, dtype=bool),
        robot_depth=np.zeros(shape, dtype=np.float32),
        object_mask=np.zeros(shape, dtype=bool),
        forced_object_mask=np.zeros(shape, dtype=bool),
        forced_robot_front_mask=np.zeros(shape, dtype=bool),
        split_depth=1.0,
        behind_robot_object_mask=np.zeros(shape, dtype=bool),
    )


class LayeredCompositorTest(unittest.TestCase):
    def test_stage_contract_has_stable_six_layer_order(self) -> None:
        self.assertEqual(
            [spec.key for spec in STAGE_SPECS],
            [
                "background", "robot_behind", "object", "robot_front",
                "forced_object", "forced_robot_front",
            ],
        )

    def test_each_stage_changes_only_its_owned_pixels(self) -> None:
        inputs = frame_inputs()
        inputs.robot_mask[[0, 0, 0, 1], [0, 1, 2, 0]] = True
        inputs.robot_depth[0, 0] = 2.0  # rear robot, then covered by object
        inputs.robot_depth[0, 1] = 0.0  # ordinary front robot
        inputs.robot_depth[0, 2] = 0.0  # ordinary front, then forced object
        inputs.robot_depth[1, 0] = 2.0  # reserved semantic thumb
        inputs.object_mask[0, 0] = True
        inputs.forced_object_mask[0, 2] = True
        inputs.forced_object_mask[1, 0] = True
        inputs.forced_robot_front_mask[1, 0] = True

        result = compose_frame(
            inputs,
            StageConfig(
                robot_edge_sigma=0,
                object_edge_sigma=0,
                forced_object_edge_sigma=0,
            ),
        )
        stage_1, stage_2, stage_3, stage_4, stage_5, stage_6 = result.stages
        self.assertTrue(np.all(stage_1[0, 0] == 10))
        self.assertTrue(np.all(stage_2[0, 0] == 100))
        self.assertTrue(np.all(stage_3[0, 0] == 200))
        self.assertTrue(np.all(stage_4[0, 1] == 100))
        self.assertTrue(np.all(stage_5[0, 2] == 200))
        self.assertTrue(np.all(stage_5[1, 0] == 10))
        self.assertTrue(np.all(stage_6[1, 0] == 100))

    def test_forced_robot_is_disjoint_and_has_final_authority(self) -> None:
        inputs = frame_inputs((5, 5))
        inputs.robot_mask[1:4, 1:4] = True
        inputs.robot_depth[:] = 2.0
        inputs.forced_object_mask[:] = True
        inputs.forced_robot_front_mask[2, 2] = True
        masks = build_layer_masks(
            inputs, StageConfig(forced_robot_front_dilate=1)
        )
        self.assertFalse(np.any(masks.robot_forced_front & ~inputs.robot_mask))
        self.assertFalse(np.any(masks.robot_forced_front & masks.robot_behind))
        self.assertFalse(np.any(masks.robot_forced_front & masks.robot_front))
        self.assertFalse(
            np.any(masks.robot_forced_front & masks.object_forced_front)
        )

    def test_behind_robot_object_never_covers_the_robot(self) -> None:
        inputs = frame_inputs((5, 5))
        inputs.robot_mask[1:4, 1:4] = True
        inputs.robot_depth[:] = 2.0  # every robot pixel is classified rear
        inputs.object_mask[:] = True
        inputs.forced_object_mask[:] = True
        inputs.behind_robot_object_mask[:] = True

        masks = build_layer_masks(inputs, StageConfig())
        self.assertFalse(np.any(masks.object_visible & inputs.robot_mask))
        self.assertFalse(np.any(masks.object_forced_front & inputs.robot_mask))
        # Outside the robot the static object is still restored, so background
        # inpainting damage stays covered.
        self.assertTrue(np.all(masks.object_visible[0]))

        result = compose_frame(
            inputs,
            StageConfig(
                robot_edge_sigma=0,
                object_edge_sigma=0,
                forced_object_edge_sigma=0,
            ),
        )
        self.assertTrue(np.all(result.final[2, 2] == 100))
        self.assertTrue(np.all(result.final[0, 0] == 200))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg tools are required")
    def test_h264_writer_publishes_browser_compatible_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preview.mp4"
            writer = CompatibleVideoWriter(output, 24.0, (32, 24), "h264")
            self.assertTrue(writer.isOpened())
            for value in (0, 64, 128, 255):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
            writer.release()
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,pix_fmt",
                    "-of", "csv=p=0", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.stdout.strip(), "h264,yuv420p")


if __name__ == "__main__":
    unittest.main()
