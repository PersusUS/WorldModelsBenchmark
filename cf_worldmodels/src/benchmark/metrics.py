"""
Benchmark metrics for measuring catastrophic forgetting in world models.
PF: Prediction Fidelity
RD: Rollout Divergence
PIS: Policy Impact Score
WMF: World Model Forgetting (aggregate)
FT: Forward Transfer
"""
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence


def diag_gaussian_kl(mu_p: Tensor, log_var_p: Tensor,
                     mu_q: Tensor, log_var_q: Tensor) -> Tensor:
    """
    D_KL( N(mu_p, diag(exp(log_var_p))) || N(mu_q, diag(exp(log_var_q))) ),
    summed over the last (feature) dimension.

    The second output of `predict_stoch` is **log-variance** throughout this
    codebase: `sample_stoch`, `compute_nll` and every `compute_loss` build the
    standard deviation as `exp(0.5 * log_sigma)`. Both arguments here follow
    that same convention, and the divergence is delegated to
    `torch.distributions.kl_divergence` rather than written out by hand.

    Returns: (...,) tensor with the batch dimensions of the inputs.
    """
    p = Normal(mu_p, torch.exp(0.5 * log_var_p))
    q = Normal(mu_q, torch.exp(0.5 * log_var_q))
    return kl_divergence(p, q).sum(dim=-1)


@torch.no_grad()
def compute_nll(model, dataset: dict, device: torch.device) -> float:
    """
    Compute negative log-likelihood of the dataset under the model:

        NLL = -E_{(z, a, z') ~ D} [ log P_theta(z' | z, a) ]

    dataset: dict with 'obs' (N, latent_dim), 'actions' (N, action_dim) and
    'next_obs' (N, latent_dim). Build it with
    `src.benchmark.protocol.build_latent_eval_dataset`.

    The target is z', the NEXT latent state. Scoring z against the model's
    prediction of z' would measure how far the transition moves the state
    rather than how well it predicts it.
    """
    if "next_obs" not in dataset:
        raise KeyError(
            "dataset must contain 'next_obs': NLL is defined over "
            "(z, a, z') triples. Use protocol.build_latent_eval_dataset()."
        )

    model.eval()
    obs = dataset["obs"].to(device)            # (N, latent_dim)   z
    actions = dataset["actions"].to(device)    # (N, action_dim)   a
    target = dataset["next_obs"].to(device)    # (N, latent_dim)   z'
    N = obs.shape[0]

    h = torch.zeros(N, model.hidden_dim, device=device)
    h_next = model.transition(h, obs, actions)

    # Predict stochastic state and compute NLL
    mu, log_sigma = model.predict_stoch(h_next)
    # NLL under a diagonal Gaussian, with log_sigma parameterizing log-variance:
    #   0.5 * sum(log(2*pi) + log_sigma + (z' - mu)^2 / sigma^2)
    sigma = torch.exp(0.5 * log_sigma)
    nll = 0.5 * torch.sum(
        torch.log(2 * torch.tensor(torch.pi, device=device))
        + log_sigma
        + (target - mu).pow(2) / (sigma.pow(2) + 1e-8),
        dim=-1,
    )  # (N,)
    return float(nll.mean().item())


def compute_pf(model_i, model_k, dataset_i: dict,
               device: torch.device) -> float:
    """
    Prediction Fidelity: PF(i,k) = NLL(M_k, D_i) - NLL(M_i, D_i)
    Positive PF = model_k forgot task i after training on tasks i+1..k.
    """
    nll_k = compute_nll(model_k, dataset_i, device)
    nll_i = compute_nll(model_i, dataset_i, device)
    return nll_k - nll_i


@torch.no_grad()
def compute_rd(model_i, model_k, dataset_i: dict,
               horizon: int = 15, n_samples: int = 50) -> float:
    """
    Rollout Divergence: RD = (1/N) sum D_KL(q(tau|M_i) || q(tau|M_k))
    tau = trajectory of `horizon` steps sampled from dataset_i start states.
    Approximated via Monte Carlo.

    The per-step divergence is the exact diagonal-Gaussian KL under the
    log-variance convention — see `diag_gaussian_kl`.
    """
    device = next(model_i.parameters()).device
    model_i.eval()
    model_k.eval()

    obs = dataset_i["obs"].to(device)
    actions = dataset_i["actions"].to(device)
    N = min(n_samples, obs.shape[0])

    # Sample start indices
    indices = torch.randperm(obs.shape[0])[:N]
    z_start = obs[indices]  # (N, latent_dim)

    total_kl = 0.0

    h_i = torch.zeros(N, model_i.hidden_dim, device=device)
    h_k = torch.zeros(N, model_k.hidden_dim, device=device)
    z_i = z_start
    z_k = z_start

    for step in range(horizon):
        # Drive the rollout with actions resampled from the dataset's own
        # action distribution. Gaussian noise would be off-manifold for
        # bounded continuous action spaces and meaningless for the one-hot
        # discrete actions used by the MiniGrid family.
        a = actions[torch.randint(0, actions.shape[0], (N,), device=actions.device)]

        # Transition
        h_i = model_i.transition(h_i, z_i, a)
        h_k = model_k.transition(h_k, z_k, a)

        # Get distributions
        mu_i, log_var_i = model_i.predict_stoch(h_i)
        mu_k, log_var_k = model_k.predict_stoch(h_k)

        # KL(P_i || P_k)
        kl = diag_gaussian_kl(mu_i, log_var_i, mu_k, log_var_k)  # (N,)
        total_kl = total_kl + kl.mean().item()

        # Sample next z from model_i for continued rollout
        z_i = mu_i  # use mean for deterministic rollout
        z_k = mu_k

    return total_kl / horizon


def compute_wmf(pf_list: List[float], rd_list: List[float],
                pis_list: List[float],
                alpha: float = 0.4, beta: float = 0.4,
                gamma: float = 0.2) -> float:
    """
    World Model Forgetting (aggregate).
    WMF = (1/(k-1)) sum [alpha*PF + beta*RD + gamma*PIS]
    Raises ValueError if alpha + beta + gamma != 1.0
    """
    if abs(alpha + beta + gamma - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {alpha + beta + gamma:.6f}"
        )
    k = len(pf_list)
    if k == 0:
        return 0.0

    total = 0.0
    for pf, rd, pis in zip(pf_list, rd_list, pis_list):
        total += alpha * pf + beta * rd + gamma * pis

    return total / k


def compute_task_A_fit_gain(nll_after_A: float, nll_random: float) -> float:
    """
    How much training on task A improved the fit to task A: nll_random - nll_A.

    This used to be called "forward transfer" (F20). It is not: no task-B data
    enters it, so every method sharing an architecture gets the same number by
    construction — measured, the delta between fine-tuning and replay was
    exactly 0.000 on 10/10 seeds.

    The quantity is still worth keeping. It answers F17's question — was there
    anything to forget? — against a random-init reference on the very dataset
    PF is computed over. Forward transfer is `compute_forward_transfer`.
    """
    return nll_random - nll_after_A


def compute_forward_transfer(error_from_scratch: float,
                             error_after_pretraining: float) -> float:
    """
    Forward transfer: how much pretraining on task A helped learn task B.

    FT = error(B | trained from scratch) - error(B | pretrained on A)

    Positive means the pretrained model reached a lower error on task B under
    the same budget and the same data, i.e. knowledge of A transferred.

    Both errors are held-out **pixel** reconstruction on task B, not latent
    NLL. The two models were trained independently, so their latent spaces are
    unrelated bases and a latent NLL would score one of them in coordinates
    that belong to the other. Pixels are the one scale both share.
    """
    return error_from_scratch - error_after_pretraining
