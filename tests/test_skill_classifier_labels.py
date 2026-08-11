import unittest

from src.skill_classifier.train import resolve_action_labels


class SkillClassifierLabelTests(unittest.TestCase):
    def test_resolves_dataset_scoped_choco_vocabulary(self):
        labels = ["Cup", "Lock", "Choco", "Snack", "Sweep", "Trans"]
        recordings = [{"action_labels": labels}, {"action_labels": labels}]
        self.assertEqual(resolve_action_labels(recordings, labels), labels)

    def test_rejects_mixed_bundle_vocabularies(self):
        recordings = [
            {"action_labels": ["Cup", "Milk", "Trans"]},
            {"action_labels": ["Cup", "Choco", "Trans"]},
        ]
        with self.assertRaisesRegex(ValueError, "mixed action_labels"):
            resolve_action_labels(recordings)


if __name__ == "__main__":
    unittest.main()
