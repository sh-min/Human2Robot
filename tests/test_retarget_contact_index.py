"""Lightweight tests for HaCo-to-HaWoR contact-frame matching."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGETING_DIR = REPO_ROOT / "src" / "retargeting"
sys.path.insert(0, str(RETARGETING_DIR))

import retarget_from_npz as retarget  # noqa: E402


class ContactFileIndexTests(unittest.TestCase):
    def test_metadata_ignores_filename_width_and_retained_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            contact_dir = Path(tmp)
            expected = contact_dir / "odd_prefix_frame7.custom-rgb.npz"
            np.savez(expected, hawor_frame_index=np.int64(42))

            index = retarget._build_contact_file_index(contact_dir)

            self.assertEqual(
                retarget._resolve_contact_file(index, 42, 999),
                str(expected),
            )

    def test_duplicate_metadata_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            contact_dir = Path(tmp)
            np.savez(contact_dir / "frame00003.npz", hawor_frame_index=3)
            np.savez(contact_dir / "frame000003.png.npz", hawor_frame_index=3)

            with self.assertRaisesRegex(ValueError, "duplicate hawor_frame_index 3"):
                retarget._build_contact_file_index(contact_dir)

    def test_legacy_fallback_accepts_arbitrary_padding_and_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            contact_dir = Path(tmp)
            five_digit = contact_dir / "rgb_frame00007.npz"
            long_custom = contact_dir / "capture_frame00000008.rawrgb.npz"
            np.savez(five_digit, left_contact_mask=np.zeros(778, dtype=bool))
            np.savez(long_custom, left_contact_mask=np.zeros(778, dtype=bool))

            index = retarget._build_contact_file_index(contact_dir)

            self.assertEqual(
                retarget._resolve_contact_file(index, 0, 7),
                str(five_digit),
            )
            self.assertEqual(
                retarget._resolve_contact_file(index, 1, 8),
                str(long_custom),
            )

    def test_metadata_takes_precedence_over_legacy_source_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            contact_dir = Path(tmp)
            metadata = contact_dir / "unrelated_name.npz"
            legacy = contact_dir / "rgb_frame00007.npz"
            np.savez(metadata, hawor_frame_index=7)
            np.savez(legacy, left_contact_mask=np.zeros(778, dtype=bool))

            index = retarget._build_contact_file_index(contact_dir)

            self.assertEqual(
                retarget._resolve_contact_file(index, 7, 7),
                str(metadata),
            )

    def test_dense_legacy_fallback_handles_restarted_slice_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            contact_dir = Path(tmp)
            restarted = contact_dir / "rgb_frame000000.npz"
            np.savez(restarted, left_contact_mask=np.zeros(778, dtype=bool))

            index = retarget._build_contact_file_index(contact_dir)

            self.assertEqual(
                retarget._resolve_contact_file(index, 0, 120),
                str(restarted),
            )


if __name__ == "__main__":
    unittest.main()
