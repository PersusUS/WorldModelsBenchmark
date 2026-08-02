"""
Distance metrics between environment pairs.
"""
import numpy as np
import torch
from torch import Tensor

from src.benchmark.metrics import diag_gaussian_kl


# The physics vector phi of Eq. 8, with the value each entry takes when a task
# does not declare it. One list, because a key added to the environment
# wrappers has to enter the distance and its label together or the two drift:
# a perturbation the distance scores and the paper's task table does not print
# is worse than either alone.
PHYSICS_PARAMS = {"gravity": 9.8, "mass_scale": 1.0, "friction_scale": 1.0}

# The fields that identify *which* environment a task spec names, as opposed to
# how it is perturbed. Eq. 8 is defined only when these agree.
TASK_IDENTITY = ("env_id", "domain_name", "task_name")


def compute_d_param(config_A: dict, config_B: dict) -> float:
    """
    Parametric distance (Family 3 / Gymnasium variable physics only).
    d = ||phi_A - phi_B||_2 / ||phi_A||_2
    phi = [gravity, mass_scale, friction_scale]
    """
    phi_A = np.array([config_A.get(k, v) for k, v in PHYSICS_PARAMS.items()],
                     dtype=np.float64)
    phi_B = np.array([config_B.get(k, v) for k, v in PHYSICS_PARAMS.items()],
                     dtype=np.float64)

    norm_A = np.linalg.norm(phi_A)
    if norm_A < 1e-12:
        return 0.0
    return float(np.linalg.norm(phi_A - phi_B) / norm_A)


def d_param_for_pair(tasks: dict):
    """
    Eq. 8 for a stored `tasks` block, or None where it would misdescribe the
    pair.

    `compute_d_param` differences a physics vector, so it says something only
    about two tasks that differ in physics *alone*. Where the body or the
    layout changes it falls back to defaults on both sides and returns the
    distance between two identical vectors --- zero for the dm_control pairs
    that swap cheetah for walker, which is the largest construction change in
    that family reported as no change at all. That gap is why Eq. 9 exists,
    and callers should show nothing rather than show the zero.

    `tasks`: {"task_A": spec, "task_B": spec} as every metrics.json stores it.
    """
    task_A, task_B = tasks["task_A"], tasks["task_B"]
    if any(task_A.get(k) != task_B.get(k) for k in TASK_IDENTITY):
        return None
    if not (task_A.get("params") and task_B.get("params")):
        return None
    return compute_d_param(task_A["params"], task_B["params"])


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
