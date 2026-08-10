"""
view_sim.py
Interactive viewer for Gymnasium MuJoCo environments.
Runs random actions or loads model weights for visual evaluation.
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from agent import ActorCritic
from env_utils import make_render_env, colorize_geoms, normalize_obs


def run_random(env_id: str, episodes: int):
    env = make_render_env(env_id, render_mode="human")
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(1 / 60)
        print(f"[random policy] episode {ep + 1}: return={total_reward:.2f}")
    env.close()


def run_checkpoint(env_id: str, checkpoint_path: str, episodes: int, deterministic: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    env = make_render_env(ckpt.get("env_id", env_id), render_mode="human")
    colorize_geoms(env)
    agent = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden_size"]).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    if ckpt.get("obs_rms_mean") is None:
        print("[warn] checkpoint has no saved observation normalization stats. Running with raw observations.")

    act_low = torch.as_tensor(env.action_space.low, device=device)
    act_high = torch.as_tensor(env.action_space.high, device=device)

    for ep in range(episodes):
        raw_obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs = normalize_obs(raw_obs, ckpt)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    # Deterministic policy pass skips action distribution noise
                    action = agent.actor_mean(obs_t)
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_t)
            action = torch.clamp(action, act_low, act_high).squeeze(0).cpu().numpy()
            raw_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            time.sleep(1 / 60)
        print(f"[checkpoint] episode {ep + 1}: return={total_reward:.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="InvertedDoublePendulum-v5")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint file. Omit to run random policy.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from policy distribution instead of using policy mean.")
    args = parser.parse_args()

    if args.checkpoint is None:
        print("[info] no checkpoint given, running random actions as a render smoke test.")
        run_random(args.env_id, args.episodes)
    else:
        run_checkpoint(args.env_id, args.checkpoint, args.episodes, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()