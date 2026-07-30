"""
UG-MTM: Uncertainty-Gated Mixture of Transition Models.
Replaces f_det in RSSM with K=4 GRU experts gated by epistemic uncertainty.

References:
    MC Dropout: Gal & Ghahramani 2016 https://arxiv.org/abs/1506.02142
    MoE: Shazeer et al. 2017 https://arxiv.org/abs/1701.06538
    Autograd: https://pytorch.org/docs/2.1/notes/autograd.html
"""
import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.rssm import BaseWorldModel
from src.models.vae import ConvVAE, reconstruction_loss


class ExpertPool(nn.Module):
    """K independent GRUCells with no shared parameters."""

    def __init__(self, K: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.K = K
        self.experts = nn.ModuleList()
        # Each expert is initialized from its own generator so the pool is
        # diverse but reproducible. A local generator is essential here: the
        # experiment runner seeds the global RNG before building the model,
        # and reseeding it during construction would silently discard that
        # seed and make runs unreproducible.
        stdv = 1.0 / math.sqrt(hidden_dim) if hidden_dim > 0 else 0.0
        for k in range(K):
            cell = nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim)
            generator = torch.Generator().manual_seed(k * 1337 + 42)
            with torch.no_grad():
                for param in cell.parameters():
                    param.uniform_(-stdv, stdv, generator=generator)
            self.experts.append(cell)


def compute_uncertainty(model: nn.Module, inputs: Tensor, T: int = 10) -> Tensor:
    """
    MC Dropout epistemic uncertainty.
    CRITICAL: model.train() must be called to keep dropout active.

    Returns variance across T forward passes: (B,)
    """
    if T <= 1:
        return torch.zeros(inputs.shape[0], device=inputs.device)

    was_training = model.training
    model.train()  # Keep dropout active

    outputs = []
    for _ in range(T):
        out = model(inputs)  # (B, D)
        outputs.append(out)

    model.train(was_training)  # Restore original mode

    stacked = torch.stack(outputs, dim=0)  # (T, B, D)
    # Variance across T passes, mean across output dimensions
    variance = stacked.var(dim=0).mean(dim=-1)  # (B,)
    return variance


class ThresholdNet(nn.Module):
    """
    Learns tau from recent uncertainty history.
    MLP: Linear(window, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
    Sigmoid ensures tau in (0, 1).
    """

    def __init__(self, window: int = 50):
        super().__init__()
        self.window = window
        self.mlp = nn.Sequential(
            nn.Linear(window, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # Uncertainty history, as a circular buffer. Both the contents and the
        # write position are registered buffers: a plain int attribute would be
        # left out of state_dict, so reloading a checkpoint would restore the
        # history but reset the cursor to 0 and resume overwriting from the
        # wrong end of the window.
        self.register_buffer("_history", torch.zeros(window))
        self.register_buffer("_ptr", torch.zeros((), dtype=torch.long))

    def update_history(self, u_mean: float) -> None:
        """Add mean uncertainty to the circular buffer."""
        self._history[int(self._ptr) % self.window] = u_mean
        self._ptr += 1

    def forward(self) -> Tensor:
        """Returns learned threshold tau: scalar tensor."""
        return self.mlp(self._history.clone().unsqueeze(0)).squeeze()


def compute_gates(u_t: Tensor, tau: Tensor, K: int,
                  K_active: int, lambda_: float = 10.0) -> Tensor:
    """
    Returns gates: (B, K) summing to 1.

    Existing experts (k < K_active):
        g_k = softmax(w_k * sigmoid(-lambda * (u_t - tau)))
    Active expert (k = K_active):
        g_K = softmax(w_K * sigmoid(lambda * (u_t - tau)))

    When u_t < tau: existing experts dominate (low uncertainty = known domain)
    When u_t >= tau: active expert dominates (high uncertainty = new domain)
    """
    B = u_t.shape[0]
    device = u_t.device

    # Compute gating signals
    # u_t: (B,), tau: scalar
    diff = u_t - tau  # (B,)

    # Use log-space for sharper gating
    logits = torch.full((B, K), -1e6, device=device)  # unused experts get -inf

    for k in range(K):
        if k < K_active:
            # Existing experts: active when u_t < tau (diff < 0)
            logits[:, k] = -lambda_ * diff  # positive when diff < 0
        elif k == K_active:
            # Active (new) expert: active when u_t >= tau (diff >= 0)
            logits[:, k] = lambda_ * diff   # positive when diff > 0

    # Normalize to sum to 1
    gates = F.softmax(logits, dim=-1)  # (B, K)
    return gates


def register_gradient_scaling_hooks(
    experts: nn.ModuleList,
    gate_weights: List[float],
    handles: list,
) -> None:
    """
    Register backward hooks that scale gradients by gate weight.

    For each expert k:
        grad_scaled = grad * gate_weights[k]

    Active expert (high gate) receives nearly full gradients.
    Frozen experts (low gate) receive near-zero gradients.
    Shared dynamics (moderate gate on multiple experts) get partial gradients
    enabling forward transfer.
    """
    for k, expert in enumerate(experts):
        scale = gate_weights[k]
        for param in expert.parameters():
            if param.requires_grad:
                handle = param.register_hook(lambda grad, s=scale: grad * s)
                handles.append(handle)


class UncertaintyHead(nn.Module):
    """Small network with dropout for MC Dropout uncertainty estimation."""

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, input_dim) -> (B, output_dim)
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


class UG_MTM(BaseWorldModel, nn.Module):
    """
    Full UG-MTM model.
    Replaces f_det in RSSM with K-expert mixture gated by uncertainty.
    """

    def __init__(self, cfg):
        super().__init__()
        self.latent_dim = cfg.model.latent_dim
        self.hidden_dim = cfg.model.hidden_dim
        self.action_dim = cfg.model.action_dim
        self.K = cfg.model.num_experts
        self.T = cfg.model.mc_dropout_T
        # MC-dropout passes used at evaluation time. Defaults to the training
        # value when the config does not set it.
        self.T_eval = cfg.model.get("mc_dropout_T_eval", cfg.model.mc_dropout_T)
        self.gate_lambda = cfg.model.gate_lambda
        self.dropout_rate = cfg.model.dropout_rate
        self.beta_kl = cfg.model.beta_kl

        # VAE
        self.vae = ConvVAE(latent_dim=self.latent_dim, beta=self.beta_kl)

        # Expert pool: K independent GRUCells
        input_dim = self.latent_dim + self.action_dim
        self.expert_pool = ExpertPool(
            K=self.K, input_dim=input_dim, hidden_dim=self.hidden_dim
        )

        # Uncertainty head with dropout for MC Dropout
        self.uncertainty_head = UncertaintyHead(
            input_dim=input_dim,
            output_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )

        # ThresholdNet
        self.threshold_net = ThresholdNet(window=cfg.model.threshold_window)

        # Stochastic state prediction
        self.stoch_fc = nn.Linear(self.hidden_dim, self.latent_dim * 2)

        # Active expert index (starts at 0)
        self.K_active = 0

        # Stores backward hook handles for gradient scaling cleanup
        self._grad_hooks = []

    def encode(self, obs: Tensor) -> Tensor:
        """obs: (B, 3, 64, 64) -> z: (B, latent_dim)"""
        return self.vae.encode(obs)

    def get_uncertainty(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """
        Returns MC Dropout variance, NOT zeros.
        h: (B, hidden_dim), z: (B, latent_dim), a: (B, action_dim)
        returns: (B,)
        """
        x = torch.cat([z, a], dim=-1)  # (B, latent_dim + action_dim)
        u = compute_uncertainty(self.uncertainty_head, x, T=self.T)
        return u

    def transition(self, h: Tensor, z: Tensor, a: Tensor) -> Tensor:
        """
        Full forward pass:
        1. Compute uncertainty u_t via MC Dropout (T passes)
        2. Update ThresholdNet with u_t
        3. Compute gates
        4. Apply gradient masking (training only)
        5. Run all K GRUs, weighted sum

        In eval mode: deterministic gating (no MC Dropout noise).
        h: (B, hidden_dim), z: (B, latent_dim), a: (B, action_dim)
        returns: h_next: (B, hidden_dim)
        """
        x = torch.cat([z, a], dim=-1)  # (B, latent_dim + action_dim)

        if self.training:
            # 1. Compute uncertainty via MC Dropout
            u_t = compute_uncertainty(self.uncertainty_head, x, T=self.T)  # (B,)

            # 2. Update ThresholdNet history
            self.threshold_net.update_history(u_t.mean().item())
            tau = self.threshold_net()  # scalar

            # 3. Compute gates
            gates = compute_gates(
                u_t, tau, self.K, self.K_active, self.gate_lambda
            )  # (B, K)

            # 4. Apply gradient scaling hooks
            gate_weights = gates.mean(dim=0).tolist()
            for handle in self._grad_hooks:
                handle.remove()
            self._grad_hooks.clear()
            register_gradient_scaling_hooks(
                self.expert_pool.experts,
                gate_weights,
                self._grad_hooks,
            )
        else:
            # Eval mode: the gate must still be driven by the model's actual
            # epistemic uncertainty on this input. Forcing u_t = 0 here would
            # make the routing depend on tau alone, so every input would be
            # scored as in-distribution and the newly activated expert would be
            # disconnected at evaluation time regardless of what it learned.
            #
            # T_eval >= T because a variance estimated from few passes is noisy,
            # and evaluation should not add avoidable variance to the metrics.
            u_t = compute_uncertainty(self.uncertainty_head, x, T=self.T_eval)
            tau = self.threshold_net()
            gates = compute_gates(
                u_t, tau, self.K, self.K_active, self.gate_lambda
            )
            # Note: the uncertainty history is deliberately NOT updated here.
            # Evaluation must not mutate model state.

        # 5. Run all K GRUs, weighted sum
        h_next = torch.zeros_like(h)  # (B, hidden_dim)
        for k in range(self.K):
            h_k = self.expert_pool.experts[k](x, h)  # (B, hidden_dim)
            h_next = h_next + gates[:, k].unsqueeze(-1) * h_k

        return h_next

    def predict_stoch(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """h: (B, hidden_dim) -> mu, log_sigma each (B, latent_dim)"""
        params = self.stoch_fc(h)
        mu, log_sigma = params.chunk(2, dim=-1)
        return mu, log_sigma

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """-> obs_recon: (B, C, H, W)"""
        return self.vae.decode(z)

    def compute_loss(self, batch: dict) -> Tuple[Tensor, dict]:
        """
        L = L_reconstruction + L_KL + L_uncertainty
        Returns (total_loss, {"reconstruction": f, "kl": f, "uncertainty": f})
        """
        device = next(self.parameters()).device
        obs = batch["obs"].to(device)       # (B, T, 3, 64, 64) or (B, 3, 64, 64)
        actions = batch["actions"].to(device)

        # Handle single-step batch
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
            actions = actions.unsqueeze(1)

        B, T = obs.shape[0], obs.shape[1]

        h = torch.zeros(B, self.hidden_dim, device=device)
        total_recon = 0.0
        total_kl = 0.0
        total_uncertainty = 0.0

        for t in range(T):
            obs_t = obs[:, t]  # (B, 3, 64, 64)

            z_post = self.vae.encode(obs_t)
            mu_prior, log_sigma_prior = self.predict_stoch(h)
            mu_post, log_sigma_post = self.vae._encode_params(obs_t)

            # KL
            kl = 0.5 * torch.sum(
                log_sigma_prior - log_sigma_post
                + (log_sigma_post.exp() + (mu_post - mu_prior).pow(2))
                / (log_sigma_prior.exp() + 1e-8)
                - 1,
                dim=-1,
            )
            total_kl = total_kl + kl.mean()

            # Reconstruction
            recon = self.vae.decode(z_post)
            recon_loss = reconstruction_loss(recon, obs_t)
            total_recon = total_recon + recon_loss

            # Uncertainty calibration loss
            if t < T - 1:
                a_t = actions[:, t]
                u_t = self.get_uncertainty(h, z_post, a_t)
                # Uncertainty calibration: encourage uncertainty to predict actual error
                h_next = self.transition(h, z_post, a_t)

                # Compute next-step prediction error as calibration target
                with torch.no_grad():
                    obs_next = obs[:, t + 1]
                    z_next = self.vae.encode(obs_next)
                    mu_pred, _ = self.predict_stoch(h_next)
                    pred_error = (z_next - mu_pred).pow(2).mean(dim=-1)  # (B,)

                uncertainty_loss = F.mse_loss(u_t, pred_error)
                total_uncertainty = total_uncertainty + uncertainty_loss

                h = h_next

        total_recon = total_recon / T
        total_kl = total_kl / T
        total_uncertainty = total_uncertainty / max(T - 1, 1)

        total_loss = total_recon + self.beta_kl * total_kl + total_uncertainty

        return total_loss, {
            "reconstruction": total_recon.item() if torch.is_tensor(total_recon) else total_recon,
            "kl": total_kl.item() if torch.is_tensor(total_kl) else total_kl,
            "uncertainty": total_uncertainty.item() if torch.is_tensor(total_uncertainty) else total_uncertainty,
            "nll": (total_recon + self.beta_kl * total_kl).item() if torch.is_tensor(total_recon) else 0.0,
        }
