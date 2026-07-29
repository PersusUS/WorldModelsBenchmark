"""
RSSM baseline world model based on DreamerV1.
Reference: https://arxiv.org/abs/1912.01603
GRUCell: https://pytorch.org/docs/2.1/generated/torch.nn.GRUCell.html
"""
from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.vae import ConvVAE, reconstruction_loss


class BaseWorldModel(ABC):
    """Abstract base for all world model implementations."""

    @abstractmethod
    def encode(self, obs: Tensor) -> Tensor:
        """obs: (B, C, H, W) -> z: (B, latent_dim)"""

    @abstractmethod
    def transition(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """h: (B, H), z: (B, L), a: (B, A) -> h_next: (B, H)"""

    @abstractmethod
    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """-> obs_recon: (B, C, H, W)"""

    @abstractmethod
    def compute_loss(self, batch: dict) -> Tuple[Tensor, dict]:
        """Returns (total_loss, component_dict)"""

    @abstractmethod
    def get_uncertainty(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """Returns u_t: (B,). Baseline returns zeros. UG-MTM returns MC variance."""


class RSSM(BaseWorldModel, nn.Module):
    """
    Recurrent State-Space Model (RSSM) baseline.
    h_t = GRUCell([z_{t-1}, a_{t-1}], h_{t-1})
    z_t ~ N(mu(h_t), sigma(h_t))
    """

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int,
                 beta_kl: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.beta_kl = beta_kl

        # VAE for observation encoding/decoding
        self.vae = ConvVAE(latent_dim=latent_dim, beta=beta_kl)

        # Deterministic transition: GRUCell
        # Input: [z_{t-1}, a_{t-1}], Hidden: h_{t-1}
        self.gru = nn.GRUCell(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
        )

        # Stochastic state prediction from h_t -> mu, log_sigma for z_t
        self.stoch_fc = nn.Linear(hidden_dim, latent_dim * 2)

    def encode(self, obs: Tensor) -> Tensor:
        """obs: (B, 3, 64, 64) -> z: (B, latent_dim)"""
        return self.vae.encode(obs)

    def transition(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """
        h: (B, hidden_dim), z: (B, latent_dim), a: (B, action_dim)
        returns: h_next: (B, hidden_dim)
        """
        x = torch.cat([z, a], dim=-1)  # (B, latent_dim + action_dim)
        h_next = self.gru(x, h)        # (B, hidden_dim)
        return h_next

    def predict_stoch(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """h: (B, hidden_dim) -> mu: (B, latent_dim), log_sigma: (B, latent_dim)"""
        params = self.stoch_fc(h)
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, log_sigma

    def sample_stoch(self, h: Tensor) -> Tensor:
        """Sample z from stochastic state given h."""
        mu, log_sigma = self.predict_stoch(h)
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """Decode observation from h and z. Uses VAE decoder on z."""
        # (B, latent_dim) -> (B, 3, 64, 64)
        return self.vae.decode(z)

    def compute_loss(self, batch: dict) -> Tuple[Tensor, dict]:
        """
        batch: dict with:
            obs: (B, T, 3, 64, 64)
            actions: (B, T, action_dim)
        Returns (total_loss, {"reconstruction": float, "kl": float, "nll": float})
        """
        device = next(self.parameters()).device
        obs = batch["obs"].to(device)       # (B, T, 3, 64, 64)
        actions = batch["actions"].to(device)  # (B, T, action_dim)
        B, T = obs.shape[0], obs.shape[1]

        # Initialize hidden state
        h = torch.zeros(B, self.hidden_dim, device=device)  # (B, H)

        total_recon = 0.0
        total_kl = 0.0

        for t in range(T):
            obs_t = obs[:, t]  # (B, 3, 64, 64)

            # Encode observation to get posterior z
            z_post = self.vae.encode(obs_t)  # (B, latent_dim)

            # Get prior prediction from h
            mu_prior, log_sigma_prior = self.predict_stoch(h)

            # Posterior params (from VAE encoder)
            mu_post, log_sigma_post = self.vae._encode_params(obs_t)

            # KL between posterior and prior
            kl = 0.5 * torch.sum(
                log_sigma_prior - log_sigma_post
                + (log_sigma_post.exp() + (mu_post - mu_prior).pow(2))
                / (log_sigma_prior.exp() + 1e-8)
                - 1,
                dim=-1,
            )  # (B,)
            total_kl = total_kl + kl.mean()

            # Reconstruction
            recon = self.vae.decode(z_post)  # (B, 3, 64, 64)
            recon_loss = reconstruction_loss(recon, obs_t)
            total_recon = total_recon + recon_loss

            # Transition to next hidden state
            if t < T - 1:
                h = self.transition(h, z_post, actions[:, t])

        total_recon = total_recon / T
        total_kl = total_kl / T
        nll = total_recon + self.beta_kl * total_kl
        total_loss = nll

        return total_loss, {
            "reconstruction": total_recon.item(),
            "kl": total_kl.item(),
            "nll": nll.item(),
        }

    def get_uncertainty(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """Baseline returns zeros — no uncertainty estimation."""
        # h: (B, hidden_dim)
        return torch.zeros(h.shape[0], device=h.device)
