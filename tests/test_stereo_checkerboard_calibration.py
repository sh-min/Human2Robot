import unittest

import numpy as np

from calibration.calibrate_stereo_checkerboard import (
    _calibrated_epipolar_rms,
    _invert_transform,
    _matrix4,
    _object_points,
    _sampson_rms,
)


class StereoCheckerboardCalibrationTest(unittest.TestCase):
    def test_object_point_scale_and_layout(self):
        points = _object_points((3, 2), 0.025)
        self.assertEqual(points.shape, (6, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points[-1], [0.05, 0.025, 0.0])

    def test_transform_inverse(self):
        angle = np.deg2rad(31.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transform = _matrix4(rotation, np.array([0.3, -0.2, 1.1]))
        np.testing.assert_allclose(
            _invert_transform(transform) @ transform,
            np.eye(4),
            atol=1e-12,
        )

    def test_sampson_error_is_zero_for_exact_correspondence(self):
        fundamental = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        points1 = np.array([[10.0, 20.0], [50.0, 70.0]])
        points2 = np.array([[15.0, 20.0], [62.0, 70.0]])
        self.assertAlmostEqual(
            _sampson_rms(fundamental, points1, points2),
            0.0,
            places=12,
        )

    def test_calibrated_epipolar_error_uses_pinhole_pixels(self):
        fundamental = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        camera_matrix = np.array(
            [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros(5)
        points1 = np.array([[610.0, 300.0], [700.0, 410.0]])
        points2 = np.array([[630.0, 300.0], [730.0, 410.0]])
        self.assertAlmostEqual(
            _calibrated_epipolar_rms(
                fundamental,
                points1,
                points2,
                camera_matrix,
                distortion,
                camera_matrix,
                distortion,
            ),
            0.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
