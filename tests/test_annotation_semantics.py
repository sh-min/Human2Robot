import unittest

from src.skill_classifier.annotation_tool.annotation_tool import (
    ACTION_DESCRIPTIONS,
    LABEL_PROFILES,
    make_panel_html,
    validate_segments_for_save,
)


class AnnotationSemanticsTests(unittest.TestCase):
    def test_ui_displays_semantics_but_save_keeps_stable_id(self):
        html = make_panel_html(["Cup", "Trans"], ACTION_DESCRIPTIONS)
        self.assertIn("컵 걸기", html)
        self.assertIn("행동 사이 전환", html)
        self.assertIn('data-skill="Cup"', html)
        segments = validate_segments_for_save(
            10,
            [{"start_frame": 0, "end_frame": 9, "label": "Cup"}],
        )
        self.assertEqual(segments[0]["label"], "Cup")

    def test_choco_profile_is_visible_and_accepted(self):
        profile = LABEL_PROFILES["kitchen_choco"]
        self.assertIn("Choco", profile["labels"])
        self.assertNotIn("Milk", profile["labels"])
        html = make_panel_html(profile["labels"], profile["descriptions"])
        self.assertIn("초코 버리기", html)
        segments = validate_segments_for_save(
            4,
            [{"start_frame": 0, "end_frame": 3, "label": "Choco"}],
            allowed_labels=profile["labels"],
        )
        self.assertEqual(segments[0]["label"], "Choco")


if __name__ == "__main__":
    unittest.main()
