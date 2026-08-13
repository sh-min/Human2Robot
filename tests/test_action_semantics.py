import tempfile
import unittest
from pathlib import Path

import yaml

from src.skill_classifier.action_semantics import (
    display_label,
    load_action_semantics,
    object_prompt_bank,
)


ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = ROOT / "src/skill_classifier/config/kitchen_action_semantics.yaml"


class ActionSemanticsTests(unittest.TestCase):
    def test_kitchen_semantics_preserve_stable_label_order(self):
        config = load_action_semantics(SEMANTICS)
        self.assertEqual(
            config["action_labels"],
            ["Cup", "Lock", "Milk", "Snack", "Sweep", "Trans"],
        )
        self.assertEqual(display_label(config, "Cup"), "Cup — 컵 걸기")
        self.assertEqual(object_prompt_bank(config)[0]["name"], "cup")
        self.assertNotIn(
            "work_surface", [item["name"] for item in object_prompt_bank(config)]
        )

    def test_ground_truth_selected_prompt_is_rejected(self):
        config = yaml.safe_load(SEMANTICS.read_text(encoding="utf-8"))
        config["conditioning_contract"]["forbid_ground_truth_selected_prompt"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ground-truth-selected"):
                load_action_semantics(path)


if __name__ == "__main__":
    unittest.main()
