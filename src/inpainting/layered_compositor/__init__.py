"""Composable six-stage interaction-object compositor.

The public API intentionally exposes every stage as a small function.  A stage
can therefore be tuned, tested, or replaced without running video I/O or
changing the other occlusion rules.
"""

from .models import CompositeFrame, FrameInputs, LayerMasks, StageConfig
from .stages import (
    STAGE_SPECS,
    build_layer_masks,
    compose_frame,
    stage_1_background,
    stage_2_robot_behind,
    stage_3_object,
    stage_4_robot_front,
    stage_5_forced_object,
    stage_6_forced_robot_front,
)

__all__ = [
    "CompositeFrame",
    "FrameInputs",
    "LayerMasks",
    "STAGE_SPECS",
    "StageConfig",
    "build_layer_masks",
    "compose_frame",
    "stage_1_background",
    "stage_2_robot_behind",
    "stage_3_object",
    "stage_4_robot_front",
    "stage_5_forced_object",
    "stage_6_forced_robot_front",
]
