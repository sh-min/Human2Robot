"""Pure, independently replaceable stages for 2.5D robot/object compositing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import CompositeFrame, FrameInputs, LayerMasks, StageConfig


@dataclass(frozen=True)
class StageSpec:
    """Human-readable contract for one cumulative compositing stage."""

    key: str
    label: str
    meaning: str
    improvement_target: str


STAGE_SPECS = (
    StageSpec(
        "background",
        "STEP 1  BACKGROUND",
        "Human-free scene plate used as the base of every later stage.",
        "Improve the human mask, temporal inpainting, and contact-region cleanup.",
    ),
    StageSpec(
        "robot_behind",
        "STEP 2  + ROBOT BEHIND",
        "Robot pixels deeper than the hand/object split plane.",
        "Improve depth alignment, split depth, and temporal depth smoothing.",
    ),
    StageSpec(
        "object",
        "STEP 3  + OBJECT",
        "Recovered source-RGB interaction object placed over the rear robot.",
        "Improve modal/amodal masks, object texture completion, and edge feathering.",
    ),
    StageSpec(
        "robot_front",
        "STEP 4  + ROBOT FRONT",
        "Ordinary robot pixels in front of the split plane, excluding forced parts.",
        "Improve robot raster edges and per-finger depth classification.",
    ),
    StageSpec(
        "forced_object",
        "STEP 5  + OBJECT FORCED FRONT",
        "Rigid-object regions that must cover the four curled fingers.",
        "Improve grasp-specific masks; keep the forced region conservative.",
    ),
    StageSpec(
        "forced_robot_front",
        "STEP 6  + THUMB FRONT  (FINAL)",
        "Semantic robot part, currently the rendered thumb, drawn last.",
        "Improve semantic-link rendering, depth agreement, and small seam dilation.",
    ),
)


def _alpha(mask: np.ndarray, sigma: float, mode: str = "gaussian",
           support: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(mask, dtype=np.float32)
    if sigma > 0 and mode == "gaussian":
        value = cv2.GaussianBlur(value, (0, 0), sigma)
    elif sigma > 0 and mode == "clamped":
        value = cv2.GaussianBlur(value, (0, 0), sigma)
        if support is None:
            raise ValueError("clamped alpha requires a support mask")
        value[~np.asarray(support, dtype=bool)] = 0.0
    elif sigma > 0 and mode == "inside":
        distance = cv2.distanceTransform(
            np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3
        )
        value = np.clip(distance / max(1.0, 1.08 * sigma), 0.0, 1.0)
    elif mode not in {"gaussian", "clamped", "inside"}:
        raise ValueError(f"unknown alpha mode: {mode}")
    return np.clip(value, 0.0, 1.0)[..., None]


def blend(accumulator: np.ndarray, content: np.ndarray, mask: np.ndarray,
          sigma: float, mode: str = "gaussian",
          support: np.ndarray | None = None) -> np.ndarray:
    """Blend one content layer over an accumulated float image."""

    alpha = _alpha(mask, sigma, mode, support)
    return (alpha * np.asarray(content, dtype=np.float32)
            + (1.0 - alpha) * np.asarray(accumulator, dtype=np.float32))


def build_layer_masks(inputs: FrameInputs, config: StageConfig) -> LayerMasks:
    """Classify pixels once and enforce the cross-stage occlusion invariants."""

    visible = np.asarray(inputs.robot_mask, dtype=bool)
    depth = np.asarray(inputs.robot_depth, dtype=np.float32)
    forced_robot = visible & np.asarray(
        inputs.forced_robot_front_mask, dtype=bool
    )
    if config.forced_robot_front_dilate > 0 and forced_robot.any():
        radius = config.forced_robot_front_dilate
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        forced_robot = cv2.dilate(
            forced_robot.astype(np.uint8), kernel, iterations=1
        ).astype(bool) & visible

    # A forced robot part is reserved for stage 6.  Removing it from both
    # ordinary robot masks makes each isolated stage semantically disjoint.
    behind = visible & (depth >= inputs.split_depth) & ~forced_robot
    front = visible & (depth < inputs.split_depth) & ~forced_robot

    # A static scene object is only an interaction object while it is grasped.
    # Before that it rests on the table behind the hand, so no object layer may
    # draw it over the robot; the single depth split cannot express that,
    # because it is calibrated for the grasped object instead.
    if inputs.behind_robot_object_mask is None:
        behind_robot_object = np.zeros_like(visible)
    else:
        behind_robot_object = (
            np.asarray(inputs.behind_robot_object_mask, dtype=bool) & visible
        )

    # The stage-6 semantic part has final authority.  Carving it out here also
    # prevents a feathered stage-5 object edge from tinting the thumb.
    forced_object = (np.asarray(inputs.forced_object_mask, dtype=bool)
                     & ~forced_robot & ~behind_robot_object)
    return LayerMasks(
        robot_visible=visible,
        robot_behind=behind,
        object_visible=(np.asarray(inputs.object_mask, dtype=bool)
                        & ~behind_robot_object),
        robot_front=front,
        object_forced_front=forced_object,
        robot_forced_front=forced_robot,
    )


def stage_1_background(inputs: FrameInputs) -> np.ndarray:
    """Start from a human-free scene plate, darkened by the robot's shadow.

    The shadow belongs to this stage because it is cast *on the plate*: every
    later layer draws opaque content over it, so a robot or object pixel is
    never tinted by its own shadow.
    """

    plate = np.asarray(inputs.background, dtype=np.float32).copy()
    if inputs.shadow_alpha is not None:
        alpha = np.asarray(inputs.shadow_alpha, dtype=np.float32)[..., None]
        plate *= 1.0 - np.clip(alpha, 0.0, 1.0)
    return plate


def stage_2_robot_behind(accumulator: np.ndarray, inputs: FrameInputs,
                         masks: LayerMasks, config: StageConfig) -> np.ndarray:
    """Add robot geometry that belongs behind the interaction object."""

    return blend(accumulator, inputs.robot_rgb, masks.robot_behind,
                 config.robot_edge_sigma, config.robot_edge_mode,
                 masks.robot_visible)


def stage_3_object(accumulator: np.ndarray, inputs: FrameInputs,
                   masks: LayerMasks, config: StageConfig) -> np.ndarray:
    """Restore the interaction object from its completed RGB source."""

    return blend(accumulator, inputs.object_rgb, masks.object_visible,
                 config.object_edge_sigma)


def stage_4_robot_front(accumulator: np.ndarray, inputs: FrameInputs,
                        masks: LayerMasks, config: StageConfig) -> np.ndarray:
    """Add normally depth-classified front robot geometry."""

    return blend(accumulator, inputs.robot_rgb, masks.robot_front,
                 config.robot_edge_sigma, config.robot_edge_mode,
                 masks.robot_visible)


def stage_5_forced_object(accumulator: np.ndarray, inputs: FrameInputs,
                          masks: LayerMasks, config: StageConfig) -> np.ndarray:
    """Put selected rigid-object interiors in front of curled fingers."""

    return blend(accumulator, inputs.object_rgb, masks.object_forced_front,
                 config.forced_object_edge_sigma)


def stage_6_forced_robot_front(accumulator: np.ndarray, inputs: FrameInputs,
                               masks: LayerMasks,
                               config: StageConfig) -> np.ndarray:
    """Draw a semantic robot part last; currently used for the thumb."""

    return blend(accumulator, inputs.robot_rgb, masks.robot_forced_front,
                 config.robot_edge_sigma, config.robot_edge_mode,
                 masks.robot_visible)


def compose_frame(inputs: FrameInputs,
                  config: StageConfig | None = None) -> CompositeFrame:
    """Run all six stages and retain every cumulative intermediate image."""

    config = config or StageConfig()
    masks = build_layer_masks(inputs, config)
    stage_1 = stage_1_background(inputs)
    stage_2 = stage_2_robot_behind(stage_1, inputs, masks, config)
    stage_3 = stage_3_object(stage_2, inputs, masks, config)
    stage_4 = stage_4_robot_front(stage_3, inputs, masks, config)
    stage_5 = stage_5_forced_object(stage_4, inputs, masks, config)
    stage_6 = stage_6_forced_robot_front(stage_5, inputs, masks, config)
    return CompositeFrame(
        stages=(stage_1, stage_2, stage_3, stage_4, stage_5, stage_6),
        masks=masks,
    )
