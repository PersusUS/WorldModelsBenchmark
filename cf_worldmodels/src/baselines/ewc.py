"""
EWC (Elastic Weight Consolidation) baseline.
Reference: Kirkpatrick et al. 2017 https://arxiv.org/abs/1612.00796

After task i, compute diagonal Fisher:
    F_i = E_{x~D_i} [(d/d_theta log P(x|theta))^2]
Store theta*_i (current weights)

Modified loss:
    L = L_task + lambda * sum_i sum_j F_{i,j} * (theta_j - theta*_{i,j})^2
"""
from typing import Tuple, List, Dict
import copy
import torch
import torch.nn as nn
from torch import Tensor

from src.models.rssm import BaseWorldModel, RSSM


class EWCWorldModel(BaseWorldModel, nn.Module):
    """EWC world model with Fisher-based regularization."""

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int,
                 ewc_lambda: float = 1000.0, beta_kl: float = 1.0):
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
        self.ewc_lambda = ewc_lambda

        # Store Fisher diagonals and optimal params for each consolidated task
        self._fisher_diags: List[Dict[str, Tensor]] = []
        self._optimal_params: List[Dict[str, Tensor]] = []

    def consolidate(self, dataset: dict, device: torch.device) -> None:
        """
        Compute and store Fisher diagonal and theta* for current task.

        dataset: dict with 'obs' (N, latent_dim), 'actions' (N, action_dim)
        and 'next_obs' (N, latent_dim), as produced by
        `src.benchmark.protocol.build_latent_eval_dataset`.

        The Fisher diagonal is the expected squared gradient of the task
        log-likelihood, so it has to be estimated on real task data: computed
        on noise it encodes no information about which parameters matter.

        The expectation is over *per-sample* gradients — E[g^2], not (E[g])^2.
        Squaring a mini-batch gradient measures only the part of the gradient
        that survives averaging, and the two differ by exactly the gradient
        variance, which is where most of the Fisher signal lives. Hence the
        per-sample loop below: one backward pass per transition, which at the
        Fisher subset sizes used here (tens of transitions) costs nothing.
        """
        if "next_obs" not in dataset:
            raise KeyError(
                "dataset must contain 'next_obs': the Fisher diagonal is "
                "defined over the task log-likelihood log P(z'|z, a). Use "
                "protocol.build_latent_eval_dataset()."
            )

        self.rssm.eval()

        # Store current optimal parameters
        optimal = {}
        for name, param in self.rssm.named_parameters():
            optimal[name] = param.data.clone()
        self._optimal_params.append(optimal)

        # Compute diagonal Fisher Information Matrix
        fisher = {}
        for name, param in self.rssm.named_parameters():
            fisher[name] = torch.zeros_like(param.data)

        obs = dataset["obs"].to(device)
        actions = dataset["actions"].to(device)
        next_obs = dataset["next_obs"].to(device)
        N = obs.shape[0]

        # One transition at a time: the Fisher is an average of squared
        # per-sample gradients, and mini-batching would average the gradients
        # before squaring them.
        for i in range(N):
            obs_b = obs[i:i + 1]
            actions_b = actions[i:i + 1]
            next_b = next_obs[i:i + 1]

            h = torch.zeros(1, self.hidden_dim, device=device)
            h_next = self.rssm.transition(h, obs_b, actions_b)
            mu, log_sigma = self.rssm.predict_stoch(h_next)

            # Log-likelihood of the observed next latent state under the model
            sigma = torch.exp(0.5 * log_sigma)
            log_prob = -0.5 * torch.sum(
                torch.log(2 * torch.tensor(torch.pi, device=device))
                + log_sigma
                + (next_b - mu).pow(2) / (sigma.pow(2) + 1e-8),
                dim=-1,
            )
            loss = -log_prob.squeeze(0)

            self.rssm.zero_grad()
            loss.backward()

            for name, param in self.rssm.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) / N

        self._fisher_diags.append(fisher)
        self.rssm.eval()

    def ewc_loss(self) -> Tensor:
        """Compute EWC regularization term."""
        device = next(self.rssm.parameters()).device
        penalty = torch.tensor(0.0, device=device)

        for fisher, optimal in zip(self._fisher_diags, self._optimal_params):
            for name, param in self.rssm.named_parameters():
                if name in fisher:
                    penalty = penalty + torch.sum(
                        fisher[name] * (param - optimal[name]).pow(2)
                    )

        return self.ewc_lambda * penalty

    def encode(self, obs: Tensor) -> Tensor:
        return self.rssm.encode(obs)

    def transition(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        return self.rssm.transition(h, z, a)

    def predict_stoch(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        return self.rssm.predict_stoch(h)

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        return self.rssm.decode(h, z)

    def compute_loss(self, batch: dict) -> Tuple[Tensor, dict]:
        """Returns task loss + EWC penalty."""
        total_loss, comps = self.rssm.compute_loss(batch)
        ewc_pen = self.ewc_loss()
        total_loss = total_loss + ewc_pen
        comps["ewc_penalty"] = ewc_pen.item()
        return total_loss, comps

    def get_uncertainty(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        return self.rssm.get_uncertainty(h, z, a)
