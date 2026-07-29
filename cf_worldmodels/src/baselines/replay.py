"""
Infinite Replay baseline: accumulates ALL transitions from ALL past tasks.
This is the upper bound — minimum forgetting expected.
"""
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from torch import Tensor

from src.models.rssm import BaseWorldModel, RSSM
from src.utils.buffer import ReplayBuffer


class InfiniteReplayWorldModel(BaseWorldModel, nn.Module):
    """
    Infinite replay baseline.
    Accumulates all data from all tasks and samples uniformly.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int,
                 beta_kl: float = 1.0):
        super().__init__()
        self.rssm = RSSM(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            beta_kl=beta_kl,
        )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Accumulated replay buffer (no size limit)
        self._all_episodes: List[List[Dict]] = []
        self._task_counts: Dict[int, int] = {}

    def add_task_data(self, task_id: int, n_samples: int = 0,
                      episodes: List[List[Dict]] = None) -> None:
        """Add data from a task to the accumulated buffer."""
        if episodes is not None:
            self._all_episodes.extend(episodes)
        self._task_counts[task_id] = self._task_counts.get(task_id, 0) + (
            n_samples if n_samples > 0 else len(episodes or [])
        )

    def buffer_size(self) -> int:
        """Total number of samples across all tasks."""
        return sum(self._task_counts.values())

    def encode(self, obs: Tensor) -> Tensor:
        return self.rssm.encode(obs)

    def transition(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        return self.rssm.transition(h, z, a)

    def predict_stoch(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        return self.rssm.predict_stoch(h)

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        return self.rssm.decode(h, z)

    def compute_loss(self, batch: dict) -> Tuple[Tensor, dict]:
        return self.rssm.compute_loss(batch)

    def get_uncertainty(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        return self.rssm.get_uncertainty(h, z, a)
