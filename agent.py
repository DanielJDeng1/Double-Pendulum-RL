"""
agent.py

Actor-critic network architecture for continuous action space PPO.
Uses isolated networks for policy and value heads to prevent gradient interference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Applies orthogonal initialization and constant bias scaling."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Decoupled actor and critic MLPs for continuous control.

    Args:
        obs_dim: Dimensionality of flattened observation space.
        act_dim: Dimensionality of action space.
        hidden_size: Hidden layer width.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 128):
        super().__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, act_dim), std=0.01),
        )

        # Learnable state-independent log std initialized to 0.0 for initial std=1.0 exploration
        self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        Samples new actions during rollouts or evaluates log probability and entropy 
        of existing action tensors during PPO policy gradient updates.
        """
        mean = self.actor_mean(obs)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value