"""
Distance metrics between environment pairs.
"""
import numpy as np
import torch
from torch import Tensor

from src.benchmark.metrics import diag_gaussian_kl


def compute_d_param(config_A: dict, config_B: dict) -> float:
    """
    Parametric distance (Family 3 / Gymnasium variable physics only).
    d = ||phi_A - phi_B||_2 / ||phi_A||_2
    phi = [gravity, mass_scale, friction_scale]
    """
    phi_A = np.array([
        config_A.get("gravity", 9.8),
        config_A.get("mass_scale", 1.0),
        config_A.get("friction_scale", 1.0),
    ], dtype=np.float64)
    phi_B = np.array([
        config_B.get("gravity", 9.8),
        config_B.get("mass_scale", 1.0),
        config_B.get("friction_scale", 1.0),
    ], dtype=np.float64)

    norm_A = np.linalg.norm(phi_A)
    if norm_A < 1e-12:
        return 0.0
    return float(np.linalg.norm(phi_A - phi_B) / norm_A)


@torch.no_grad()
def compute_d_trans(model_A, model_B, shared_dataset: dict,
                    device: torch.device) -> float:
    """
    Transition distance (universal, all families).
    d = E_{(s,a)~pi_rand} [D_KL(P_A(s'|s,a) || P_B(s'|s,a))]
    Approximated using shared rollouts and the two trained models.

    shared_dataset: dict with 'obs' (N, latent_dim) and 'actions' (N, action_dim)

    The divergence is the exact diagonal-Gaussian KL under the log-variance
    convention — see `src.benchmark.metrics.diag_gaussian_kl`.
    """
    obs = shared_dataset["obs"].to(device)       # (N, latent_dim)
    actions = shared_dataset["actions"].to(device)  # (N, action_dim)
    N = obs.shape[0]

    model_A.eval()
    model_B.eval()

    # Initialize hidden states
    h_A = torch.zeros(N, model_A.hidden_dim, device=device)
    h_B = torch.zeros(N, model_B.hidden_dim, device=device)

    # Get transition predictions from both models
    h_A_next = model_A.transition(h_A, obs, actions)
    h_B_next = model_B.transition(h_B, obs, actions)

    # Get stochastic state distributions
    mu_A, log_var_A = model_A.predict_stoch(h_A_next)
    mu_B, log_var_B = model_B.predict_stoch(h_B_next)

    # KL(P_A || P_B) for diagonal Gaussians, log-variance convention
    kl = diag_gaussian_kl(mu_A, log_var_A, mu_B, log_var_B)  # (N,)

    return float(kl.mean().item())
