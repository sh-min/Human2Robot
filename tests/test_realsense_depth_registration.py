"""Lightweight synthetic tests for calibrated RealSense depth registration."""

from __future__ import annotations

import unittest

import numpy as np

from calibration.register_realsense_depth import (
    Intrinsics,
    apply_brown_conrady,
    compose_rigid_transforms,
    invert_brown_conrady,
    parse_rs2_depth_to_color_transform,
    project_normalized_to_pixels,
    register_depth_frame,
)


def _intrinsics(width: int, height: int, *, fx: float = 1.0) -> Intrinsics:
    return Intrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fx,
        ppx=0.0,
        ppy=0.0,
        model="Brown Conrady",
        coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


class RealSenseDepthRegistrationTest(unittest.TestCase):
    def test_forward_brown_projection_matches_closed_form(self):
        intrinsics = Intrinsics(
            width=640,
            height=480,
            fx=100.0,
            fy=120.0,
            ppx=10.0,
            ppy=20.0,
            model="Inverse Brown Conrady",
            coeffs=(0.1, 0.01, 0.001, -0.002, 0.0005),
        )
        point = np.array([[0.2, -0.1]], dtype=np.float64)
        radius_squared = 0.2**2 + (-0.1) ** 2
        radial = (
            1.0
            + 0.1 * radius_squared
            + 0.01 * radius_squared**2
            + 0.0005 * radius_squared**3
        )
        expected_x = (
            0.2 * radial
            + 2.0 * 0.001 * 0.2 * -0.1
            - 0.002 * (radius_squared + 2.0 * 0.2**2)
        )
        expected_y = (
            -0.1 * radial
            + 0.001 * (radius_squared + 2.0 * (-0.1) ** 2)
            + 2.0 * -0.002 * 0.2 * -0.1
        )
        pixels = project_normalized_to_pixels(
            point,
            intrinsics,
            projection_model_override="opencv_brown_forward",
        )
        np.testing.assert_allclose(
            pixels,
            [[expected_x * 100.0 + 10.0, expected_y * 120.0 + 20.0]],
            atol=1.0e-12,
        )

    def test_inverse_brown_newton_round_trip(self):
        points = np.array(
            [[-0.7, -0.4], [-0.2, 0.3], [0.0, 0.0], [0.65, 0.45]],
            dtype=np.float64,
        )
        coefficients = np.array([-0.054, 0.062, 0.0002, 0.0013, -0.020])
        distorted = apply_brown_conrady(points, coefficients)
        recovered = invert_brown_conrady(distorted, coefficients)
        np.testing.assert_allclose(recovered, points, atol=1.0e-11)

    def test_rs2_rotation_is_parsed_column_major(self):
        # A valid 90-degree Z rotation with serialized columns.
        raw = (
            "rotation=0,1,0,-1,0,0,0,0,1;"
            "translation=0.1,-0.2,0.3"
        )
        rotation, translation = parse_rs2_depth_to_color_transform(raw)
        np.testing.assert_allclose(
            rotation,
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(translation, [0.1, -0.2, 0.3])

    def test_identity_registration_preserves_metric_depth(self):
        raw = np.array([[1000, 0, 2000], [500, 750, 0]], dtype=np.uint16)
        intrinsics = _intrinsics(width=3, height=2)
        registered, valid = register_depth_frame(
            raw,
            depth_units_m=0.001,
            source_intrinsics=intrinsics,
            target_intrinsics=intrinsics,
            rotation_target_from_depth=np.eye(3),
            translation_target_from_depth=np.zeros(3),
            min_depth_m=0.1,
            max_depth_m=3.0,
        )
        np.testing.assert_array_equal(valid, raw > 0)
        np.testing.assert_allclose(registered[valid], raw[valid] * 0.001)
        self.assertTrue(np.isnan(registered[~valid]).all())

    def test_z_buffer_keeps_nearest_projected_surface(self):
        # With target fx=0.1, both source columns round to target pixel x=0.
        raw = np.array([[1000, 2000]], dtype=np.uint16)
        source = _intrinsics(width=2, height=1, fx=1.0)
        target = _intrinsics(width=1, height=1, fx=0.1)
        registered, valid = register_depth_frame(
            raw,
            depth_units_m=0.001,
            source_intrinsics=source,
            target_intrinsics=target,
            rotation_target_from_depth=np.eye(3),
            translation_target_from_depth=np.zeros(3),
            min_depth_m=0.1,
            max_depth_m=3.0,
        )
        self.assertTrue(valid[0, 0])
        self.assertAlmostEqual(float(registered[0, 0]), 1.0, places=6)

    def test_factory_and_refined_composition_matches_direct_reference(self):
        refined_rotation = np.array(
            [
                [-0.88085240, 0.21175580, -0.42338933],
                [-0.28495992, 0.47700365, 0.83142369],
                [0.37801705, 0.85301055, -0.35982790],
            ]
        )
        refined_translation = np.array([0.38073620, -0.87296286, 1.09063763])
        factory_rotation, factory_translation = (
            parse_rs2_depth_to_color_transform(
                "rotation=0.999993,0.000462,-0.003680,-0.000466,0.999999,"
                "-0.001124,0.003679,0.001125,0.999993;"
                "translation=-0.059094,0.000057,0.000528"
            )
        )
        rotation, translation = compose_rigid_transforms(
            refined_rotation,
            refined_translation,
            factory_rotation,
            factory_translation,
        )
        np.testing.assert_allclose(
            rotation,
            [
                [-0.87919033, 0.21264196, -0.42638880],
                [-0.28779719, 0.47620144, 0.83090613],
                [0.37973266, 0.85323799, -0.35747502],
            ],
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            translation,
            [0.43257781, -0.85565726, 1.06815772],
            atol=1.0e-8,
        )


if __name__ == "__main__":
    unittest.main()
