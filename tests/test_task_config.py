import tempfile
import unittest
from pathlib import Path
import yaml

from src.task_config import load_task_spec


REPO = Path(__file__).parents[1]


class TaskConfigTests(unittest.TestCase):
    def test_kitchen_task_profile_is_valid_and_multi_object(self):
        spec = load_task_spec(REPO / "configs/tasks/kitchen.yaml")
        self.assertEqual(spec["task_id"], "kitchen")
        self.assertEqual(
            spec["action_labels"],
            ["Cup", "Lock", "Choco", "Snack", "Sweep", "Trans"],
        )
        self.assertIn("milk_carton", spec["object_ids"])
        self.assertIn("cup_blue", spec["object_ids"])
        self.assertEqual(
            len(spec["object_ids"]), len(set(spec["object_ids"]))
        )
        self.assertIn(
            "kitchen_dataset", spec["dataset"]["recordings_root"]
        )

    def test_task_profile_rejects_duplicate_labels(self):
        source = yaml.safe_load(
            (REPO / "configs/tasks/kitchen.yaml").read_text()
        )
        source["action_labels"] = ["Cup", "Cup"]
        source["object_specs"] = [
            str(REPO / "configs/objects/cup_blue.yaml")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_task_spec(path)


if __name__ == "__main__":
    unittest.main()
