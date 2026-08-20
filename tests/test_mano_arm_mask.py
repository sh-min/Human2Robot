import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src/inpainting/build_mano_arm_mask.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("mano_arm_mask", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManoArmMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_grounded_boxes_keep_objects_and_reject_scene_wide_box(self):
        detections = [
            {"grounding_score": 0.9, "box_xyxy": [20, 30, 80, 90]},
            {"grounding_score": 0.9, "box_xyxy": [0, 0, 200, 120]},
            {"grounding_score": 0.1, "box_xyxy": [120, 30, 180, 90]},
        ]
        mask = self.module.grounded_boxes(
            detections, width=200, height=120, min_score=0.3, padding=0
        )
        self.assertTrue(mask[30:90, 20:80].all())
        self.assertFalse(mask[:, 100:].any())
        self.assertEqual(int(mask.sum()), 60 * 60)

    def test_connected_border_arm_can_bend_outside_straight_corridor(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[64:76, 64:76] = 1
        seed = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
        corridor = np.zeros_like(mask)
        corridor[60:80, :] = 1
        skin = np.zeros_like(mask)
        cv2.line(skin, (70, 70), (45, 45), 255, 10)
        cv2.line(skin, (45, 45), (0, 25), 255, 10)

        self.module.add_connected_arm_skin(
            mask, skin, seed, corridor, mano_area=200,
            max_hand_ratio=6.0, max_frame_ratio=0.20,
        )

        self.assertTrue(mask[25, 0])
        self.assertTrue(mask[45, 45])

    def test_scene_wide_skin_falls_back_to_geometry_corridor(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[64:76, 64:76] = 1
        seed = cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)
        corridor = np.zeros_like(mask)
        corridor[60:80, :] = 1
        skin = np.ones_like(mask, dtype=np.uint8) * 255

        self.module.add_connected_arm_skin(
            mask, skin, seed, corridor, mano_area=144,
            max_hand_ratio=6.0, max_frame_ratio=0.20,
        )

        self.assertTrue(mask[70, 10])
        self.assertFalse(mask[10, 10])


if __name__ == "__main__":
    unittest.main()
