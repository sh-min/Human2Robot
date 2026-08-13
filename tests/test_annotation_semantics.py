import unittest

from src.skill_classifier.annotation_tool.annotation_tool import (
    ACTION_DESCRIPTIONS,
    LABEL_PROFILES,
    make_panel_html,
    validate_segments_for_save,
)


class AnnotationSemanticsTests(unittest.TestCase):
    def test_ui_displays_semantics_but_save_keeps_stable_id(self):
        html = make_panel_html(["HangCup", "Transition"], ACTION_DESCRIPTIONS)
        self.assertIn("Hang the cup on the cup holder", html)
        self.assertIn("Transition between actions", html)
        self.assertIn('data-skill="HangCup"', html)
        segments = validate_segments_for_save(
            10,
            [{"start_frame": 0, "end_frame": 9, "label": "HangCup"}],
        )
        self.assertEqual(segments[0]["label"], "HangCup")

    def test_choco_profile_is_visible_and_accepted(self):
        profile = LABEL_PROFILES["kitchen_choco"]
        self.assertIn("PlaceLightGreenSnackBoxInTrashBin", profile["labels"])
        self.assertNotIn("Milk", profile["labels"])
        html = make_panel_html(profile["labels"], profile["descriptions"])
        self.assertIn("Place the light green snack box in the trash bin", html)
        self.assertIn("Place the red snack box in the trash bin", html)
        segments = validate_segments_for_save(
            4,
            [{"start_frame": 0, "end_frame": 3, "label": "PlaceLightGreenSnackBoxInTrashBin"}],
            allowed_labels=profile["labels"],
        )
        self.assertEqual(segments[0]["label"], "PlaceLightGreenSnackBoxInTrashBin")


if __name__ == "__main__":
    unittest.main()
