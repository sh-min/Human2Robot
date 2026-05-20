"""Evaluate a trained policy in the MuJoCo RBY1+XHand environment.

Supports two policy backends:
  1. LeRobot (Diffusion Policy, ACT, etc.) -- loaded via PreTrainedPolicy
  2. GR00T N1 -- loaded via Gr00tPolicy

The evaluation loop renders episodes, computes metrics (episode length,
total reward), and optionally saves rollout videos.

Usage:
    # LeRobot Diffusion Policy:
    MUJOCO_GL=egl PYTHONPATH=$PWD/src python -m policy.eval_mujoco \\
        --backend lerobot \\
        --checkpoint /path/to/lerobot_checkpoint \\
        --n_episodes 10 \\
        --save_video

    # GR00T N1:
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


def _make_env(image_size: int = 224, max_steps: int = 300):
    """Instantiate the MuJoCo sim environment."""
    from sim.mujoco_sim.env import EnvConfig, RBY1XHandEnv

    cfg = EnvConfig(image_size=image_size, max_episode_steps=max_steps)
    return RBY1XHandEnv(cfg)


def _load_lerobot_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a LeRobot pretrained policy."""
    from lerobot.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    policy = make_policy(config)
    policy.eval()
    return policy


def _load_groot_policy(
    checkpoint_path: str,
    modality_config: str | None = None,
    device: str = "cuda",
):
    """Load a GR00T N1 policy."""
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


def _obs_to_policy_input(obs: dict, backend: str) -> dict:
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
        return {
            "video.observation.images.head_cam": np.expand_dims(image, axis=0),
            "state": np.expand_dims(state, axis=0),
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
        if isinstance(output, dict) and "action" in output:
            action = np.asarray(output["action"])
        else:
            action = np.asarray(output)
        if action.ndim == 3:
            action = action[0, 0]
        elif action.ndim == 2:
            action = action[0]
        return action.astype(np.float32)


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
):
    """Run evaluation loop."""
    if backend == "lerobot":
        policy = _load_lerobot_policy(checkpoint, device=device)
    elif backend == "groot":
        policy = _load_groot_policy(checkpoint, modality_config=modality_config, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    env = _make_env(image_size=image_size, max_steps=max_steps)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_lengths = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        frames = []
        total_reward = 0.0
        step = 0
        done = False

        t0 = time.time()
        while not done:
            policy_input = _obs_to_policy_input(obs, backend)
            raw_output = (
                policy.select_action(policy_input)
                if backend == "lerobot"
                else policy.get_action(policy_input)
            )
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
        print(
            f"Episode {ep:3d}: steps={step:4d}  reward={total_reward:.2f}  "
            f"time={elapsed:.1f}s  ({step/elapsed:.1f} fps)"
        )

        if save_video and frames:
            _save_rollout_video(frames, out_dir / f"episode_{ep:03d}.mp4")

    env.close()

    avg_len = np.mean(episode_lengths)
    avg_rew = np.mean(episode_rewards)
    print(f"\n{'='*50}")
    print(f"Results over {n_episodes} episodes:")
    print(f"  avg length: {avg_len:.1f}")
    print(f"  avg reward: {avg_rew:.3f}")
    print(f"{'='*50}")

    import json

    metrics = {
        "n_episodes": n_episodes,
        "avg_length": float(avg_len),
        "avg_reward": float(avg_rew),
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
    }
    metrics_path = out_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained policy in MuJoCo.")
    ap.add_argument(
        "--backend", required=True, choices=["lerobot", "groot"],
        help="Policy backend: 'lerobot' for Diffusion/ACT, 'groot' for GR00T N1",
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
    args = ap.parse_args()

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
    )


if __name__ == "__main__":
    main()
