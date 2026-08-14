"""Data contracts shared by the layered compositor stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StageConfig:
    """Independent tuning knobs for the six-stage compositor."""

    robot_edge_sigma: float = 1.2
    robot_edge_mode: str = "clamped"
    object_edge_sigma: float = 0.8
    forced_object_edge_sigma: float = 0.45
    forced_robot_front_dilate: int = 0


@dataclass(frozen=True)
class FrameInputs:
    """All image and geometry inputs for one frame.

    Arrays use OpenCV BGR order.  Masks are converted to boolean by the stage
    module, so callers may provide boolean or 0/1 arrays.
    """

    background: np.ndarray
    robot_rgb: np.ndarray
    object_rgb: np.ndarray
    robot_mask: np.ndarray
    robot_depth: np.ndarray
    object_mask: np.ndarray
    forced_object_mask: np.ndarray
    forced_robot_front_mask: np.ndarray
    # Scalar plane, or an (H, W) per-pixel split surface when the caller has a
    # contact-derived depth map. Both broadcast against robot_depth.
    split_depth: float | np.ndarray
    behind_robot_object_mask: np.ndarray | None = None
    shadow_alpha: np.ndarray | None = None


@dataclass(frozen=True)
class LayerMasks:
    """Mutually ordered masks consumed by stages 2 through 6."""

    robot_visible: np.ndarray
    robot_behind: np.ndarray
    object_visible: np.ndarray
    robot_front: np.ndarray
    object_forced_front: np.ndarray
    robot_forced_front: np.ndarray


@dataclass(frozen=True)
class CompositeFrame:
    """Cumulative images and isolated masks from one compositor pass."""

    stages: tuple[np.ndarray, ...]
    masks: LayerMasks

    @property
    def final(self) -> np.ndarray:
        return self.stages[-1]
