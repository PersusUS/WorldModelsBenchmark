"""
Fine-tuning baseline: sequential training with no forgetting protection.
This is the lower bound — maximum forgetting expected.
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor

from src.models.rssm import BaseWorldModel, RSSM


class FineTuningWorldModel(BaseWorldModel, nn.Module):
    """
    Fine-tuning baseline. Identical to RSSM but with explicit task switching.
    Optimizer resets between tasks but weights do NOT reset.
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
        # Expose key attributes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def encode(self, obs: Tensor) -> Tensor:
        """obs: (B, 3, 64, 64) -> z: (B, latent_dim)"""
        return self.rssm.encode(obs)

    def transition(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """h: (B, H), z: (B, L), a: (B, A) -> h_next: (B, H)"""
        return self.rssm.transition(h, z, a)

    def predict_stoch(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """h: (B, H) -> mu, log_sigma each (B, latent_dim)"""
        return self.rssm.predict_stoch(h)

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """-> obs_recon: (B, C, H, W)"""
        return self.rssm.decode(h, z)

    def compute_loss(self, batch: dict) -> Tuple[Tensor, dict]:
        """Returns (total_loss, component_dict)"""
        return self.rssm.compute_loss(batch)

    def get_uncertainty(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """Returns zeros — no uncertainty estimation."""
        return self.rssm.get_uncertainty(h, z, a)
