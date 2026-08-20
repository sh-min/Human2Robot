import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src/inpainting/rb5_arm_ik.py"
)


def _load_arm_ik():
    spec = importlib.util.spec_from_file_location("rb5_arm_ik_constraints", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RobotTrajectoryConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pinocchio  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("pinocchio is unavailable") from exc
        cls.arm_ik = _load_arm_ik()

    def test_rate_limiter_enforces_position_and_velocity(self):
        trajectory = np.array([[0.0, 0.0], [2.0, -2.0], [-2.0, 2.0]])
        limited = self.arm_ik.rate_limit_trajectory(
            trajectory,
            velocity_limit=np.array([1.0, 2.0]),
            fps=10.0,
            velocity_scale=0.5,
            lower=np.array([-0.2, -0.3]),
            upper=np.array([0.2, 0.3]),
        )
        self.assertTrue(np.all(limited >= np.array([-0.2, -0.3])))
        self.assertTrue(np.all(limited <= np.array([0.2, 0.3])))
        self.assertTrue(
            np.all(
                np.abs(np.diff(limited, axis=0))
                <= np.array([0.05, 0.1]) + 1e-12
            )
        )

    def test_rb5_output_constraint_validator_rejects_speeding(self):
        model, _, _ = self.arm_ik.load_model()
        q = np.tile(np.zeros(model.nq), (2, 1))
        q[1, 0] = 0.2
        with self.assertRaisesRegex(ValueError, "velocity"):
            self.arm_ik.assert_trajectory_limits(q, model, fps=30.0)

    def test_collision_projection_preserves_a_clear_rb5_trajectory(self):
        model, data, _ = self.arm_ik.load_model()
        q = np.tile(np.zeros(model.nq), (3, 1))
        projected, adjusted, pairs = (
            self.arm_ik.project_collision_free_trajectory(
                model, data, self.arm_ik.RB5_URDF, q
            )
        )
        np.testing.assert_array_equal(projected, q)
        self.assertEqual(adjusted, 0)
        self.assertGreater(pairs, 0)


if __name__ == "__main__":
    unittest.main()
