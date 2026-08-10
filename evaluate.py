"""
evaluate.py
Batch evaluation harness for saved model checkpoints.
Runs deterministic or stochastic evaluation rollouts and logs return statistics or CSVs.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch

from agent import ActorCritic
from env_utils import make_render_env, colorize_geoms, normalize_obs


def evaluate(
    checkpoint_path: str,
    episodes: int,
    render: bool,
    deterministic: bool,
    csv_out: str | None,
    max_steps: int = 1000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    env_id = ckpt.get("env_id", "InvertedDoublePendulum-v5")

    if ckpt.get("obs_rms_mean") is None:
        print("[warn] checkpoint has no saved observation normalization stats. Running with raw observations.")

    if render:
        env = make_render_env(env_id, render_mode="human")
    else:
        env = gym.make(env_id)

    agent = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden_size"]).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    act_low = torch.as_tensor(env.action_space.low, device=device)
    act_high = torch.as_tensor(env.action_space.high, device=device)

    returns, lengths = [], []

    try:
        for ep in range(episodes):
            raw_obs, _ = env.reset(seed=1000 + ep)

            if render:
                colorize_geoms(env)

            done = False
            total_reward, steps = 0.0, 0
            while not done:
                obs = normalize_obs(raw_obs, ckpt)
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    if deterministic:
                        action = agent.actor_mean(obs_t)
                    else:
                        action, _, _, _ = agent.get_action_and_value(obs_t)
                action = torch.clamp(action, act_low, act_high).squeeze(0).cpu().numpy()
                raw_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                # truncate manually if max_steps cap reached
                done = terminated or truncated or (steps >= max_steps)

                if render:
                    time.sleep(1 / 60)

            returns.append(total_reward)
            lengths.append(steps)
            print(f"episode {ep + 1:3d}/{episodes} | return={total_reward:8.2f} | length={steps}")
    finally:
        env.close()

    returns_arr, lengths_arr = np.array(returns), np.array(lengths)
    print("\nEvaluation summary")
    print(f"episodes:         {episodes}")
    print(f"mean return:      {returns_arr.mean():.2f} +/- {returns_arr.std():.2f}")
    print(f"min / max return: {returns_arr.min():.2f} / {returns_arr.max():.2f}")
    print(f"mean length:      {lengths_arr.mean():.1f}")
    print(f"trained steps:    {ckpt.get('global_step', 'unknown')}")

    if csv_out:
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        with open(csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "return", "length"])
            for i, (r, l) in enumerate(zip(returns, lengths)):
                writer.writerow([i + 1, r, l])
        print(f"[info] per-episode results written -> {csv_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode before truncating.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from distribution rather than taking policy mean.")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional output path for per-episode CSV exports.")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
        csv_out=args.csv_out,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()