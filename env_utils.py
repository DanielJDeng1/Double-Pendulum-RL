"""
env_utils.py
Environment initialization and checkpoint helpers shared across evaluation scripts.
Overrides default MuJoCo camera parameters to keep the full mechanism in frame.
"""

from __future__ import annotations

import glob
import os

import gymnasium as gym
import numpy as np
import torch

from agent import ActorCritic

CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 6.5,
    "lookat": np.array([0.0, 0.0, 1.0]),
    "elevation": -15.0,
    "azimuth": 90.0,
}

COLORS = [
    [0.2, 0.2, 0.2, 1.0],
    [0.1, 0.5, 0.9, 1.0],
    [0.9, 0.2, 0.2, 1.0],
    [0.2, 0.8, 0.2, 1.0],
    [0.9, 0.8, 0.1, 1.0],
    [0.8, 0.2, 0.8, 1.0],
]


def make_render_env(env_id: str, render_mode: str = "human") -> gym.Env:
    return gym.make(env_id, render_mode=render_mode, default_camera_config=CAMERA_CONFIG)


def colorize_geoms(env: gym.Env) -> None:
    """
    Applies contrasting RGBA colors to environment geometries for playback visibility.
    """
    try:
        model = env.unwrapped.model
        for i in range(model.ngeom):
            model.geom_rgba[i] = COLORS[i % len(COLORS)]
    except AttributeError:
        pass


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def build_agent(ckpt: dict, device: torch.device) -> ActorCritic:
    agent = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden_size"]).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


def normalize_obs(obs: np.ndarray, ckpt: dict) -> np.ndarray:
    """
    Normalizes observation vector using saved RunningMeanStd parameters from training.
    """
    mean = ckpt.get("obs_rms_mean")
    var = ckpt.get("obs_rms_var")
    clip = ckpt.get("obs_clip", 10.0)
    if mean is None or var is None:
        return obs
    normed = (obs - mean) / np.sqrt(var + 1e-8)
    return np.clip(normed, -clip, clip)


def latest_checkpoint(models_dir: str = "models", run_name: str | None = None) -> str | None:
    pattern = os.path.join(models_dir, f"{run_name}.pth" if run_name else "*.pth")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def list_checkpoints(models_dir: str = "models") -> list[str]:
    return sorted(glob.glob(os.path.join(models_dir, "*.pth")), key=os.path.getmtime, reverse=True)