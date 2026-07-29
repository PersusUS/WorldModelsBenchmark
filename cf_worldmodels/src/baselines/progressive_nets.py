"""
Progressive Networks baseline.
Reference: Rusu et al. 2016 https://arxiv.org/abs/1606.04671

Column architecture:
    Each column is a GRUCell.
    Lateral connections: h_k = GRU_k([z, a, V_k @ h_{k-1}])
    where V_k is a learnable projection from previous column's hidden state.

add_column():
    1. Freeze all existing columns (requires_grad_(False))
    2. Add new GRUCell column with fresh weights
    3. Add lateral projection from last frozen column
"""
from typing import Tuple, List
import torch
import torch.nn as nn
from torch import Tensor

from src.models.rssm import BaseWorldModel
from src.models.vae import ConvVAE, reconstruction_loss


class ProgressiveNetWorldModel(BaseWorldModel, nn.Module):
    """Progressive Networks world model with one GRU column per task."""

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int,
                 beta_kl: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.beta_kl = beta_kl

        # Shared VAE
        self.vae = ConvVAE(latent_dim=latent_dim, beta=beta_kl)

        # Columns: list of GRUCells
        input_dim = latent_dim + action_dim
        self.columns = nn.ModuleList([
            nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim)
        ])

        # Lateral connections: projections from previous column
        self.laterals = nn.ModuleList()

        # Stochastic state predictors per column
        self.stoch_fcs = nn.ModuleList([
            nn.Linear(hidden_dim, latent_dim * 2)
        ])

        self._active_column = 0

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    def add_column(self) -> None:
        """Freeze current columns, add new column with lateral connections."""
        # Freeze all existing columns
        for col in self.columns:
            col.requires_grad_(False)
        for fc in self.stoch_fcs:
            fc.requires_grad_(False)
        for lat in self.laterals:
            lat.requires_grad_(False)

        # Add lateral projection from last column
        self.laterals.append(
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Add new column with input augmented by lateral
        input_dim = self.latent_dim + self.action_dim + self.hidden_dim
        self.columns.append(
            nn.GRUCell(input_size=input_dim, hidden_size=self.hidden_dim)
        )
        self.stoch_fcs.append(
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
        )

        self._active_column = len(self.columns) - 1

        # Move new modules to same device
        device = next(self.columns[0].parameters()).device
        self.columns[-1].to(device)
        self.laterals[-1].to(device)
        self.stoch_fcs[-1].to(device)

    def encode(self, obs: Tensor) -> Tensor:
        """obs: (B, 3, 64, 64) -> z: (B, latent_dim)"""
        return self.vae.encode(obs)

    def transition(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """
        h: (B, hidden_dim), z: (B, latent_dim), a: (B, action_dim)
        returns: h_next: (B, hidden_dim)
        Uses the active (latest) column with lateral input from previous.
        """
        col_idx = self._active_column
        base_input = torch.cat([z, a], dim=-1)  # (B, latent_dim + action_dim)

        if col_idx == 0:
            return self.columns[0](base_input, h)

        # Run the frozen columns bottom-up. Only column 0 consumes the raw
        # input; every later column expects its predecessor's lateral appended,
        # so the chain has to be replayed rather than jumping straight to
        # column col_idx - 1.
        with torch.no_grad():
            h_prev = self.columns[0](base_input, h)
            for k in range(1, col_idx):
                lateral_k = self.laterals[k - 1](h_prev)
                h_prev = self.columns[k](
                    torch.cat([base_input, lateral_k], dim=-1), h
                )

        # Lateral connection into the active (trainable) column
        lateral = self.laterals[col_idx - 1](h_prev)
        aug_input = torch.cat([base_input, lateral], dim=-1)
        return self.columns[col_idx](aug_input, h)

    def predict_stoch(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """h: (B, hidden_dim) -> mu, log_sigma each (B, latent_dim)"""
        params = self.stoch_fcs[self._active_column](h)
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, log_sigma

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """-> obs_recon: (B, C, H, W)"""
        return self.vae.decode(z)

    def compute_loss(self, batch: dict) -> Tuple[Tensor, dict]:
        """Compute loss using RSSM-style training on active column."""
        device = next(self.parameters()).device
        obs = batch["obs"].to(device)       # (B, T, 3, 64, 64)
        actions = batch["actions"].to(device)  # (B, T, action_dim)
        B, T = obs.shape[0], obs.shape[1]

        h = torch.zeros(B, self.hidden_dim, device=device)
        total_recon = 0.0
        total_kl = 0.0

        for t in range(T):
            obs_t = obs[:, t]
            z_post = self.vae.encode(obs_t)
            mu_prior, log_sigma_prior = self.predict_stoch(h)
            mu_post, log_sigma_post = self.vae._encode_params(obs_t)

            kl = 0.5 * torch.sum(
                log_sigma_prior - log_sigma_post
                + (log_sigma_post.exp() + (mu_post - mu_prior).pow(2))
                / (log_sigma_prior.exp() + 1e-8)
                - 1,
                dim=-1,
            )
            total_kl = total_kl + kl.mean()

            recon = self.vae.decode(z_post)
            recon_loss = reconstruction_loss(recon, obs_t)
            total_recon = total_recon + recon_loss

            if t < T - 1:
                h = self.transition(h, z_post, actions[:, t])

        total_recon = total_recon / T
        total_kl = total_kl / T
        nll = total_recon + self.beta_kl * total_kl

        return nll, {
            "reconstruction": total_recon.item(),
            "kl": total_kl.item(),
            "nll": nll.item(),
        }

    def get_uncertainty(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """Returns zeros — no uncertainty estimation."""
        return torch.zeros(h.shape[0], device=h.device)
