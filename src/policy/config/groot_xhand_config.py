"""GR00T N1 modality config for RBY1 + bimanual XHand.

Registers a NEW_EMBODIMENT config that maps the 38-D state/action vector
(produced by pkl_to_lerobot) to GR00T's modality system.

State/action vector layout (matches pkl_to_lerobot.schema):
    0:12   right_hand_joint   (12 finger DOFs)
   12:15   right_wrist_pos    (xyz camera-frame)
   15:19   right_wrist_quat   (xyzw quaternion)
   19:31   left_hand_joint
   31:34   left_wrist_pos
   34:38   left_wrist_quat

Usage:
    # Finetuning GR00T N1 on the converted dataset:
    python gr00t_finetune.py \\
        --dataset-path /path/to/lerobot_dataset \\
        --modality-config-path src/policy/config/groot_xhand_config.py \\
        --embodiment-tag NEW_EMBODIMENT \\
        --num-gpus 1

    # Inference:
    python standalone_inference_script.py \\
        --model-path /path/to/checkpoint \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path src/policy/config/groot_xhand_config.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

xhand_bimanual_config = {
    # Video: single head-mounted camera, current frame only.
    # The key must match the "video" entry in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["observation.images.head_cam"],
    ),
    # State: current proprioceptive reading.
    # Keys must match the "state" entries in meta/modality.json.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_hand_joint",   # 12D finger joints
            "right_wrist_pos",    # 3D position
            "right_wrist_quat",   # 4D quaternion
            "left_hand_joint",    # 12D finger joints
            "left_wrist_pos",     # 3D position
            "left_wrist_quat",    # 4D quaternion
        ],
    ),
    # Action: 16-step prediction horizon.
    # One ActionConfig per modality key, in the same order.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "right_hand_joint",
            "right_wrist_pos",
            "right_wrist_quat",
            "left_hand_joint",
            "left_wrist_pos",
            "left_wrist_quat",
        ],
        action_configs=[
            # Right hand finger joints: relative (delta) for better generalization.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Right wrist position: relative.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Right wrist quaternion: absolute (quaternion delta is ill-defined).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Left hand finger joints: relative.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Left wrist position: relative.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Left wrist quaternion: absolute.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(xhand_bimanual_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
