"""Unit tests for the causal T-Rex-style HaCo visibility adapter."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refine_dense_visibility_trex_temporal.py"
SPEC = importlib.util.spec_from_file_location("trex_haco_visibility", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trex)


class TemporalSignalTests(unittest.TestCase):
    def test_causal_statistics_do_not_use_future_frames(self):
        signal = np.zeros((12, 5), dtype=np.float32)
        changed_future = signal.copy()
        changed_future[7:] = 1.0

        original = trex.causal_window_statistics(signal, 6)
        changed = trex.causal_window_statistics(changed_future, 6)

        for original_stat, changed_stat in zip(original, changed):
            np.testing.assert_array_equal(original_stat[:7], changed_stat[:7])

    def test_sample_and_hold_uses_latest_slow_tick(self):
        signal = np.arange(30, dtype=np.float32).reshape(6, 5)
        held = trex.sample_and_hold(signal, stride=3)
        np.testing.assert_array_equal(held[:3], np.repeat(signal[:1], 3, axis=0))
        np.testing.assert_array_equal(held[3:], np.repeat(signal[3:4], 3, axis=0))

    def test_contact_state_distinguishes_onset_sustain_and_release(self):
        gate = np.asarray([[0.0], [0.7], [0.8], [0.2]], dtype=np.float32)
        states = trex.discrete_contact_state(gate)
        np.testing.assert_array_equal(states[:, 0], np.asarray([0, 1, 2, 3]))

    def test_fast_haco_residual_is_causal_and_preserves_baseline(self):
        frames = 10
        anchor = np.ones((frames, 5), dtype=np.float32)
        anchor[4:] = 0.0
        baseline = np.full((frames, 5), 0.10, dtype=np.float32)
        no_contact = np.zeros_like(anchor)
        contact = np.zeros_like(anchor)
        contact[4:, 0] = 1.0

        low = trex.build_slow_fast_signals(
            anchor,
            baseline,
            no_contact,
            temporal_window=4,
            slow_stride=4,
            contact_threshold=0.2,
            contact_weight=1.0,
            residual_strength=1.0,
        )
        high = trex.build_slow_fast_signals(
            anchor,
            baseline,
            contact,
            temporal_window=4,
            slow_stride=4,
            contact_threshold=0.2,
            contact_weight=1.0,
            residual_strength=1.0,
        )

        self.assertTrue(np.all(high["desired_ratio"] >= baseline))
        self.assertGreater(
            float(high["desired_ratio"][5, 0]),
            float(low["desired_ratio"][5, 0]),
        )
        np.testing.assert_array_equal(
            high["desired_ratio"][:4], low["desired_ratio"][:4]
        )


class SpatialExpansionTests(unittest.TestCase):
    def test_expansion_stays_on_finger_and_inside_object_support(self):
        seed = np.zeros((24, 24), dtype=bool)
        seed[10, 10] = True
        finger = np.zeros_like(seed)
        finger[5:19, 5:19] = True
        allowed = np.zeros_like(seed)
        allowed[8:16, 8:16] = True

        expanded = trex.expand_from_seed(
            seed, finger, allowed, desired_count=30
        )

        self.assertTrue(expanded[10, 10])
        self.assertEqual(int(expanded.sum()), 30)
        self.assertFalse(np.any(expanded & ~finger))
        self.assertFalse(np.any((expanded & ~seed) & ~allowed))


if __name__ == "__main__":
    unittest.main()
