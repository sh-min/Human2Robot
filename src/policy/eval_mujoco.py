"""Evaluate a trained policy in the MuJoCo RBY1+XHand environment.

Supports two policy backends:
  1. LeRobot (Diffusion Policy, ACT, etc.) -- loaded via PreTrainedPolicy
  2. GR00T N1.7 -- loaded via Gr00tPolicy

The evaluation loop renders episodes, computes metrics (episode length,
total reward), and optionally saves rollout videos.

Usage:
    # LeRobot Diffusion Policy:
    MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \\
        --backend lerobot \\
        --checkpoint /path/to/lerobot_checkpoint \\
        --n_episodes 10 \\
        --save_video

    # GR00T N1.7:
    MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \\
        --backend groot \\
        --checkpoint /path/to/groot_checkpoint \\
        --modality_config src/policy/config/groot_xhand_config.py \\
        --n_episodes 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent


def _make_env(
    image_size: int = 224,
    max_steps: int = 300,
    active_hands: tuple[str, ...] = ("left",),
    object_spec: str | None = None,
    randomize_object: bool = True,
):
    """Instantiate the MuJoCo sim environment."""
    from sim.mujoco_sim.env import EnvConfig, RBY1XHandEnv

    cfg = EnvConfig(
        image_size=image_size,
        max_episode_steps=max_steps,
        active_hands=active_hands,
        object_spec=object_spec,
        randomize_object=randomize_object,
    )
    return RBY1XHandEnv(cfg)


def _load_lerobot_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a LeRobot pretrained policy."""
    from lerobot.policies.factory import (
        get_policy_class,
        make_pre_post_processors,
    )
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    config.pretrained_path = Path(checkpoint_path)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        checkpoint_path,
        config=config,
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint_path,
    )
    return policy, preprocessor, postprocessor


def _load_groot_policy(
    checkpoint_path: str,
    modality_config: str | None = None,
    device: str = "cuda",
):
    """Load a GR00T N1.7 policy."""
    if modality_config:
        import importlib.util

        spec = importlib.util.spec_from_file_location("modality_cfg", modality_config)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    from gr00t.policy import Gr00tPolicy
    from gr00t.data.embodiment_tags import EmbodimentTag

    policy = Gr00tPolicy(
        model_path=checkpoint_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device=device,
    )
    return policy


def _obs_to_policy_input(
    obs: dict,
    backend: str,
    instruction: str | None = None,
) -> dict:
    """Convert MuJoCo env observation to the format expected by the policy."""
    import torch

    state = obs["observation.state"]
    image = obs["observation.images.head_cam"]

    if backend == "lerobot":
        return {
            "observation.state": torch.from_numpy(state).unsqueeze(0).float(),
            "observation.images.head_cam": (
                torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ),
        }
    else:
        instruction = instruction or "manipulate object"
        # GR00T N1.7 expects explicit modality dictionaries and B,T leading
        # dimensions, rather than the flattened keys used by older releases.
        return {
            "video": {
                "observation.images.head_cam": image[None, None, ...],
            },
            "state": {
                "right_hand_joint": state[None, None, 0:12],
                "right_wrist_pos": state[None, None, 12:15],
                "right_wrist_quat": state[None, None, 15:19],
                "left_hand_joint": state[None, None, 19:31],
                "left_wrist_pos": state[None, None, 31:34],
                "left_wrist_quat": state[None, None, 34:38],
            },
            "language": {
                "annotation.human.task_description": [[instruction]],
            },
        }


def _policy_output_to_action(output, backend: str) -> np.ndarray:
    """Extract the next action from the policy's output."""
    if backend == "lerobot":
        import torch

        if isinstance(output, dict) and "action" in output:
            action = output["action"]
        else:
            action = output
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.ndim == 3:
            action = action[0, 0]
        elif action.ndim == 2:
            action = action[0]
        return action.astype(np.float32)
    else:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, dict):
            raise TypeError(
                f"GR00T N1.7 output must be a modality dict, got {type(output)}"
            )
        keys = (
            "right_hand_joint",
            "right_wrist_pos",
            "right_wrist_quat",
            "left_hand_joint",
            "left_wrist_pos",
            "left_wrist_quat",
        )
        missing = [key for key in keys if key not in output]
        if missing:
            raise KeyError(f"GR00T output is missing modalities: {missing}")
        chunks = []
        for key in keys:
            value = np.asarray(output[key])
            if value.ndim == 3:
                value = value[0, 0]
            elif value.ndim == 2:
                value = value[0]
            chunks.append(value.reshape(-1))
        return np.concatenate(chunks).astype(np.float32)


def _save_rollout_video(frames: list[np.ndarray], path: Path, fps: int = 30):
    """Write a list of RGB frames to an mp4."""
    import subprocess
    import tempfile
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="eval_mujoco_"))
    try:
        for i, frame in enumerate(frames):
            import imageio.v2 as imageio

            imageio.imwrite(str(tmp / f"frame_{i:05d}.png"), frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(tmp / "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def evaluate(
    backend: str,
    checkpoint: str,
    *,
    modality_config: str | None = None,
    n_episodes: int = 10,
    max_steps: int = 300,
    image_size: int = 224,
    device: str = "cuda",
    save_video: bool = False,
    output_dir: str = "output/eval",
    active_hands: tuple[str, ...] | None = None,
    object_spec: str | None = None,
    randomize_object: bool = True,
    instruction: str | None = None,
):
    """Run evaluation loop."""
    if object_spec and (instruction is None or active_hands is None):
        from object_config import load_object_spec

        loaded_spec = load_object_spec(object_spec)
        if instruction is None:
            instruction = loaded_spec["task"]["instruction"]
        if active_hands is None:
            active_hands = tuple(loaded_spec["control"]["active_hands"])
    if active_hands is None:
        active_hands = ("left",)

    if backend == "lerobot":
        policy, preprocessor, postprocessor = _load_lerobot_policy(
            checkpoint, device=device
        )
    elif backend == "groot":
        policy = _load_groot_policy(checkpoint, modality_config=modality_config, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    env = _make_env(
        image_size=image_size,
        max_steps=max_steps,
        active_hands=active_hands,
        object_spec=object_spec,
        randomize_object=randomize_object,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_lengths = []
    episode_rewards = []
    episode_successes = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        frames = []
        total_reward = 0.0
        step = 0
        done = False

        t0 = time.time()
        while not done:
            policy_input = _obs_to_policy_input(
                obs, backend, instruction=instruction
            )
            if backend == "lerobot":
                policy_input = preprocessor(policy_input)
                raw_output = policy.select_action(policy_input)
                raw_output = postprocessor(raw_output)
            else:
                raw_output = policy.get_action(policy_input)
            action = _policy_output_to_action(raw_output, backend)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            if save_video:
                frames.append(env.render())

        elapsed = time.time() - t0
        episode_lengths.append(step)
        episode_rewards.append(total_reward)
        episode_successes.append(bool(info.get("success", False)))
        print(
            f"Episode {ep:3d}: steps={step:4d}  reward={total_reward:.2f}  "
            f"success={episode_successes[-1]}  "
            f"time={elapsed:.1f}s  ({step/elapsed:.1f} fps)"
        )

        if save_video and frames:
            _save_rollout_video(frames, out_dir / f"episode_{ep:03d}.mp4")

    env.close()

    avg_len = np.mean(episode_lengths)
    avg_rew = np.mean(episode_rewards)
    success_rate = np.mean(episode_successes)
    print(f"\n{'='*50}")
    print(f"Results over {n_episodes} episodes:")
    print(f"  avg length: {avg_len:.1f}")
    print(f"  avg reward: {avg_rew:.3f}")
    print(f"  success rate: {success_rate:.1%}")
    print(f"{'='*50}")

    import json

    metrics = {
        "n_episodes": n_episodes,
        "avg_length": float(avg_len),
        "avg_reward": float(avg_rew),
        "success_rate": float(success_rate),
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "object_spec": object_spec,
        "instruction": instruction,
    }
    metrics_path = out_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained policy in MuJoCo.")
    ap.add_argument(
        "--backend", required=True, choices=["lerobot", "groot"],
        help="Policy backend: 'lerobot' for Diffusion/ACT, 'groot' for GR00T N1.7",
    )
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument(
        "--modality_config", default=None,
        help="Path to GR00T modality config .py (required for groot backend)",
    )
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--output_dir", default="output/eval")
    ap.add_argument(
        "--active_hands",
        default="spec",
        choices=["spec", "left", "right", "both"],
        help="Controlled hands; 'spec' reads control.active_hands.",
    )
    ap.add_argument(
        "--object_spec",
        default=None,
        help="Object YAML. Enables object randomization and success metrics.",
    )
    ap.add_argument(
        "--fixed_object_pose",
        action="store_true",
        help="Use the object spec's nominal pose instead of randomization.",
    )
    ap.add_argument(
        "--instruction",
        default=None,
        help="Language instruction passed to the GR00T backend.",
    )
    args = ap.parse_args()
    active_hands = None
    if args.active_hands == "both":
        active_hands = ("right", "left")
    elif args.active_hands != "spec":
        active_hands = (args.active_hands,)

    evaluate(
        backend=args.backend,
        checkpoint=args.checkpoint,
        modality_config=args.modality_config,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        image_size=args.image_size,
        device=args.device,
        save_video=args.save_video,
        output_dir=args.output_dir,
        active_hands=active_hands,
        object_spec=args.object_spec,
        randomize_object=not args.fixed_object_pose,
        instruction=args.instruction,
    )


if __name__ == "__main__":
    main()
