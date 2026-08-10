"""
live_view.py

Persistent MuJoCo viewer monitoring a checkpoint file to render episodes.
Reloads weights on disk modification to decouple rendering from trainer execution cycles.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from agent import ActorCritic
from env_utils import make_render_env, colorize_geoms, load_checkpoint, build_agent, normalize_obs


def wait_for_checkpoint(path: str, poll_seconds: float = 1.0) -> None:
    if os.path.exists(path):
        return
    print(f"[live_view] waiting for checkpoint to appear at {path} ...")
    while not os.path.exists(path):
        time.sleep(poll_seconds)
    print("[live_view] checkpoint found, starting.")


def run(checkpoint_path: str, deterministic: bool, episodes: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wait_for_checkpoint(checkpoint_path)

    ckpt = load_checkpoint(checkpoint_path, device)
    env_id = ckpt.get("env_id", "InvertedDoublePendulum-v5")
    env = make_render_env(env_id, render_mode="human")
    colorize_geoms(env)

    agent = build_agent(ckpt, device)
    last_mtime = os.path.getmtime(checkpoint_path)
    last_global_step = ckpt.get("global_step", 0)

    act_low = torch.as_tensor(env.action_space.low, device=device)
    act_high = torch.as_tensor(env.action_space.high, device=device)

    ep = 0
    print(f"[live_view] watching {checkpoint_path} (target episodes: {'continuous' if episodes < 0 else episodes})")
    
    try:
        while episodes < 0 or ep < episodes:
            # refresh network weights if trainer overwrote target file
            try:
                mtime = os.path.getmtime(checkpoint_path)
            except FileNotFoundError:
                mtime = last_mtime
                
            if mtime != last_mtime:
                try:
                    ckpt = load_checkpoint(checkpoint_path, device)
                    agent.load_state_dict(ckpt["model_state_dict"])
                    agent.eval()
                    last_mtime = mtime
                    if ckpt.get("global_step", 0) != last_global_step:
                        last_global_step = ckpt.get("global_step", 0)
                        print(f"[live_view] reloaded checkpoint (global_step={last_global_step})")
                except (RuntimeError, EOFError, OSError):
                    # ignore partial reads from concurrent file writes
                    pass

            raw_obs, _ = env.reset()
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
                done = terminated or truncated
                
                # pace rendering to real time
                time.sleep(1 / 60)

            ep += 1
            print(f"[live_view] episode {ep} | return={total_reward:8.2f} | length={steps} "
                  f"| global_step={last_global_step}")
                  
    except KeyboardInterrupt:
        print("\n[live_view] stopped by user.")
    except Exception as e:
        print(f"[live_view] viewer closed ({e.__class__.__name__}: {e})")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint updated in place by background trainer.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from distribution rather than taking policy mean.")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to render (default: 20). Pass -1 for continuous execution.")
    args = parser.parse_args()
    
    run(args.checkpoint, deterministic=not args.stochastic, episodes=args.episodes)


if __name__ == "__main__":
    main()