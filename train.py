"""
train.py
Training loop for PPO on Gymnasium MuJoCo environments.
Tracks running observation statistics and syncs state to JSON for GUI polling and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agent import ActorCritic


@dataclass
class PPOConfig:
    env_id: str = "InvertedDoublePendulum-v5"
    total_steps: int = 2_000_000_000
    num_envs: int = 8
    num_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 20
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    hidden_size: int = 128
    obs_clip: float = 10.0
    seed: int = 1
    checkpoint_every_updates: int = 10
    run_name: str = "ppo_double_pendulum"


class RunningMeanStd:
    """
    Online Welford variance tracking across parallel environments.
    Prevents value loss destabilization when episode lengths stretch and returns scale up.
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: np.ndarray, clip: float) -> np.ndarray:
        normed = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normed, -clip, clip)


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def get_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            # catch missing architecture kernels on new GPU arches early
            _ = torch.zeros(1, device="cuda") + 1
            return torch.device("cuda")
        except RuntimeError as e:
            print(
                "[warn] CUDA device detected but a test kernel failed to run "
                f"({e}). Falling back to CPU."
            )
            return torch.device("cpu")
    return torch.device("cpu")


def main(cfg: PPOConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    print(f"[info] using device: {device}")

    run_dir = os.path.join("runs", cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    writer = SummaryWriter(run_dir)
    writer.add_text("config", str(cfg))

    status_path = os.path.join(run_dir, "status.json")
    stats_path = os.path.join(run_dir, "stats.json")
    stats_history: list[dict] = []

    def write_status(state: str, **extra):
        payload = {
            "state": state,
            "run_name": cfg.run_name,
            "env_id": cfg.env_id,
            "device": str(device),
            "num_envs": cfg.num_envs,
            "total_steps": cfg.total_steps,
            "checkpoint_path": os.path.join("models", f"{cfg.run_name}.pth"),
            "start_time": start_time_wall,
            "timestamp": time.time(),
        }
        payload.update(extra)
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, status_path)

    def write_stats():
        tmp = stats_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(stats_history, f)
        os.replace(tmp, stats_path)

    start_time_wall = time.time()
    write_status("running")

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed + i) for i in range(cfg.num_envs)]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    act_low = torch.as_tensor(envs.single_action_space.low, device=device)
    act_high = torch.as_tensor(envs.single_action_space.high, device=device)

    agent = ActorCritic(obs_dim, act_dim, cfg.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_steps // batch_size

    # rollout buffers
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((cfg.num_steps, cfg.num_envs, act_dim), device=device)
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

    raw_next_obs, _ = envs.reset(seed=cfg.seed)
    obs_rms.update(raw_next_obs)
    next_obs = torch.as_tensor(
        obs_rms.normalize(raw_next_obs, cfg.obs_clip), dtype=torch.float32, device=device
    )
    next_done = torch.zeros(cfg.num_envs, device=device)

    recent_returns: deque = deque(maxlen=50)
    recent_lengths: deque = deque(maxlen=50)

    # handle process termination cleanly without dropping state or leaving state flags hanging
    stop_requested = {"flag": False}

    def _handle_stop(signum, frame):
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    global_step = 0
    start_time = time.time()
    interrupted = False

    for update in range(1, num_updates + 1):
        if stop_requested["flag"]:
            interrupted = True
            print(f"\n[info] stop requested — exiting after update {update - 1}.")
            break

        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # rollout collection
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value

            clipped_action = torch.clamp(action, act_low, act_high)
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            raw_next_obs, reward, terminated, truncated, infos = envs.step(
                clipped_action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            obs_rms.update(raw_next_obs)
            next_obs_np = obs_rms.normalize(raw_next_obs, cfg.obs_clip)

            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            if "episode" in infos:
                ep_returns = infos["episode"]["r"]
                ep_lengths = infos["episode"]["l"]
                mask = infos["_episode"] if "_episode" in infos else None
                for i in range(cfg.num_envs):
                    if mask is None or mask[i]:
                        writer.add_scalar("charts/episodic_return", ep_returns[i], global_step)
                        writer.add_scalar("charts/episodic_length", ep_lengths[i], global_step)
                        recent_returns.append(float(ep_returns[i]))
                        recent_lengths.append(float(ep_lengths[i]))

        # GAE target estimation
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards_buf, device=device)
            last_gae_lam = torch.zeros(cfg.num_envs, device=device)
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - dones_buf[t + 1]
                    next_val = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * next_val * next_non_terminal - values_buf[t]
                last_gae_lam = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1, act_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # policy and value function update
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]

                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                log_ratio = new_logprob - b_logprobs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # telemetry and checkpoint export
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        avg_return_str = f"{np.mean(recent_returns):.1f}" if recent_returns else "n/a"
        avg_length_str = f"{np.mean(recent_lengths):.0f}" if recent_lengths else "n/a"
        print(f"[update {update}/{num_updates}] step={global_step} sps={sps} "
              f"avg_return(last{len(recent_returns)})={avg_return_str} "
              f"avg_length={avg_length_str} "
              f"policy_loss={pg_loss.item():.4f} value_loss={v_loss.item():.4f}")

        stats_history.append({
            "update": update,
            "num_updates": num_updates,
            "global_step": global_step,
            "sps": sps,
            "elapsed": time.time() - start_time,
            "avg_return": float(np.mean(recent_returns)) if recent_returns else None,
            "avg_length": float(np.mean(recent_lengths)) if recent_lengths else None,
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        write_stats()
        write_status("running", update=update, num_updates=num_updates, global_step=global_step,
                      sps=sps, avg_return=stats_history[-1]["avg_return"])

        if update % cfg.checkpoint_every_updates == 0 or update == num_updates:
            ckpt_path = os.path.join("models", f"{cfg.run_name}.pth")
            torch.save({
                "model_state_dict": agent.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_size": cfg.hidden_size,
                "env_id": cfg.env_id,
                "global_step": global_step,
                "obs_rms_mean": obs_rms.mean,
                "obs_rms_var": obs_rms.var,
                "obs_rms_count": obs_rms.count,
                "obs_clip": cfg.obs_clip,
            }, ckpt_path)
            print(f"[info] checkpoint saved -> {ckpt_path}")

    if interrupted:
        ckpt_path = os.path.join("models", f"{cfg.run_name}.pth")
        torch.save({
            "model_state_dict": agent.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_size": cfg.hidden_size,
            "env_id": cfg.env_id,
            "global_step": global_step,
            "obs_rms_mean": obs_rms.mean,
            "obs_rms_var": obs_rms.var,
            "obs_rms_count": obs_rms.count,
            "obs_clip": cfg.obs_clip,
        }, ckpt_path)
        print(f"[info] final checkpoint saved on stop -> {ckpt_path}")

    write_status("stopped" if interrupted else "finished", global_step=global_step,
                 update=update, num_updates=num_updates)

    envs.close()
    writer.close()


def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="InvertedDoublePendulum-v5")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-name", type=str, default="ppo_double_pendulum")
    args = p.parse_args()
    return PPOConfig(
        env_id=args.env_id,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main(parse_args())