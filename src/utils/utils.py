"""Pipeline-wide utility functions."""

import numpy as np


def rotmat_to_axis_angle(R):
    """Convert 3x3 rotation matrix to axis-angle (3,) via Rodrigues."""
    R = np.asarray(R, dtype=np.float64)
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float32)
