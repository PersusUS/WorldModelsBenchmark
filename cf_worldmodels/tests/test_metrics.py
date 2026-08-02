"""Tests for src/benchmark/metrics.py (PF, RD, WMF, FT)."""
import copy

import pytest
import torch

from src.benchmark.metrics import (
    compute_task_A_fit_gain,
    compute_forward_transfer,
    compute_nll,
    compute_pf,
    compute_rd,
    compute_wmf,
    diag_gaussian_kl,
)
from src.models.rssm import RSSM

from conftest import LATENT_DIM, HIDDEN_DIM, ACTION_DIM


@pytest.fixture
def model():
    return RSSM(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                action_dim=ACTION_DIM)


# --- WMF -------------------------------------------------------------------

def test_wmf_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_wmf([1.0], [1.0], [1.0], alpha=0.5, beta=0.5, gamma=0.5)


def test_wmf_accepts_default_weights():
    assert compute_wmf([1.0], [1.0], [1.0]) == pytest.approx(1.0)


def test_wmf_empty_lists_return_zero():
    assert compute_wmf([], [], []) == 0.0


def test_wmf_is_weighted_mean_over_tasks():
    # alpha*PF + beta*RD + gamma*PIS, averaged over the 2 entries
    wmf = compute_wmf([1.0, 3.0], [0.0, 0.0], [0.0, 0.0],
                      alpha=0.4, beta=0.4, gamma=0.2)
    assert wmf == pytest.approx(0.4 * 2.0)


def test_wmf_gamma_weights_pis():
    """The term is still in Eq. 6, and Eq. 6 is what compute_wmf reproduces.
    What is withdrawn (D18) is PIS as a reported metric, not this argument."""
    wmf = compute_wmf([0.0], [0.0], [5.0], alpha=0.4, beta=0.4, gamma=0.2)
    assert wmf == pytest.approx(1.0)


# --- PF / NLL --------------------------------------------------------------

def test_nll_is_finite_scalar(model, latent_dataset, device):
    nll = compute_nll(model, latent_dataset, device)
    assert isinstance(nll, float)
    assert torch.isfinite(torch.tensor(nll))


def test_nll_requires_the_next_state_target(model, latent_dataset, device):
    """NLL is -log P(z'|z, a); without z' there is nothing to score against."""
    del latent_dataset["next_obs"]
    with pytest.raises(KeyError, match="next_obs"):
        compute_nll(model, latent_dataset, device)


def test_nll_is_scored_against_next_obs_not_obs(model, latent_dataset, device):
    """Changing only the target must change the NLL."""
    baseline = compute_nll(model, latent_dataset, device)
    latent_dataset["next_obs"] = latent_dataset["next_obs"] + 3.0
    assert compute_nll(model, latent_dataset, device) != pytest.approx(baseline)


def test_pf_is_zero_for_identical_models(model, latent_dataset, device):
    """No forgetting can be measured between a model and a copy of itself."""
    twin = copy.deepcopy(model)
    assert compute_pf(model, twin, latent_dataset, device) == pytest.approx(0.0, abs=1e-5)


def test_pf_is_positive_when_second_model_is_worse(model, latent_dataset, device):
    """PF(i, k) = NLL(M_k, D_i) - NLL(M_i, D_i); a degraded M_k must score higher."""
    degraded = copy.deepcopy(model)
    with torch.no_grad():
        for param in degraded.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    pf = compute_pf(model, degraded, latent_dataset, device)
    assert pf > 0.0


def test_pf_is_antisymmetric(model, latent_dataset, device):
    other = copy.deepcopy(model)
    with torch.no_grad():
        for param in other.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    forward = compute_pf(model, other, latent_dataset, device)
    backward = compute_pf(other, model, latent_dataset, device)
    assert forward == pytest.approx(-backward, abs=1e-4)


# --- diagonal Gaussian KL --------------------------------------------------

def test_kl_matches_torch_reference():
    """Regression for F13: the hand-written formula was neither the log-std
    nor the log-variance KL. It must now match torch exactly."""
    torch.manual_seed(0)
    mu_p, log_var_p = torch.randn(4, 6), torch.randn(4, 6)
    mu_q, log_var_q = torch.randn(4, 6), torch.randn(4, 6)

    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(mu_p, torch.exp(0.5 * log_var_p)),
        torch.distributions.Normal(mu_q, torch.exp(0.5 * log_var_q)),
    ).sum(dim=-1)

    kl = diag_gaussian_kl(mu_p, log_var_p, mu_q, log_var_q)
    assert torch.allclose(kl, reference, atol=1e-6)


def test_kl_uses_the_log_variance_convention():
    """The second output of `predict_stoch` is log-variance everywhere else in
    the codebase (`exp(0.5 * log_sigma)` is the std). Interpreting it as a
    log-deviation would give a different, wrong number."""
    mu = torch.zeros(1, 1)
    log_var_p = torch.zeros(1, 1)         # variance 1
    log_var_q = torch.full((1, 1), 2.0)   # variance e^2, std e

    # KL(N(0,1) || N(0, e^2)) = log(e) + (1 - e^2) / (2 e^2)
    expected = 1.0 + (1.0 - torch.e ** 2) / (2 * torch.e ** 2)
    assert diag_gaussian_kl(mu, log_var_p, mu, log_var_q).item() == pytest.approx(
        expected, abs=1e-6
    )


def test_kl_is_zero_for_identical_distributions():
    mu, log_var = torch.randn(3, 5), torch.randn(3, 5)
    kl = diag_gaussian_kl(mu, log_var, mu.clone(), log_var.clone())
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_is_non_negative_and_asymmetric():
    torch.manual_seed(1)
    mu_p, log_var_p = torch.randn(8, 4), torch.randn(8, 4)
    mu_q, log_var_q = torch.randn(8, 4), torch.randn(8, 4)

    forward = diag_gaussian_kl(mu_p, log_var_p, mu_q, log_var_q)
    backward = diag_gaussian_kl(mu_q, log_var_q, mu_p, log_var_p)
    assert torch.all(forward >= 0.0)
    assert torch.all(backward >= 0.0)
    assert not torch.allclose(forward, backward)


# --- RD --------------------------------------------------------------------

def test_rd_is_zero_for_identical_models(model, latent_dataset):
    twin = copy.deepcopy(model)
    rd = compute_rd(model, twin, latent_dataset, horizon=3, n_samples=8)
    assert rd == pytest.approx(0.0, abs=1e-5)


def test_rd_is_non_negative_and_finite(model, latent_dataset):
    other = copy.deepcopy(model)
    with torch.no_grad():
        for param in other.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    rd = compute_rd(model, other, latent_dataset, horizon=3, n_samples=8)
    assert rd >= 0.0
    assert torch.isfinite(torch.tensor(rd))


def test_rd_rolls_out_with_actions_from_the_dataset(model, latent_dataset):
    """Rollout actions must be drawn from the task's own action distribution;
    Gaussian noise is off-manifold for bounded and for one-hot action spaces."""
    latent_dataset["actions"] = torch.full((16, ACTION_DIM), 7.0)
    captured = []
    original = model.transition

    def spy(h, z, a):
        captured.append(a.clone())
        return original(h, z, a)

    model.transition = spy
    compute_rd(model, copy.deepcopy(model), latent_dataset,
               horizon=2, n_samples=4)
    model.transition = original

    assert captured, "transition was never called"
    assert all(torch.all(a == 7.0) for a in captured)


def test_rd_respects_n_samples_cap(model, latent_dataset):
    """n_samples larger than the dataset must not raise."""
    rd = compute_rd(model, copy.deepcopy(model), latent_dataset,
                    horizon=2, n_samples=10_000)
    assert torch.isfinite(torch.tensor(rd))


# --- task-A fit gain (what used to be called FT) ---------------------------

def test_task_A_fit_gain_positive_when_training_on_A_helped():
    """nll_random - nll_after_A; positive means task A was learned."""
    assert compute_task_A_fit_gain(nll_after_A=1.0,
                                   nll_random=5.0) == pytest.approx(4.0)


def test_task_A_fit_gain_negative_when_training_on_A_hurt():
    assert compute_task_A_fit_gain(nll_after_A=5.0,
                                   nll_random=1.0) == pytest.approx(-4.0)


# --- forward transfer ------------------------------------------------------

def test_forward_transfer_positive_when_pretraining_helps():
    """Lower error after pretraining than from scratch means A transferred."""
    assert compute_forward_transfer(error_from_scratch=10.0,
                                    error_after_pretraining=4.0)         == pytest.approx(6.0)


def test_forward_transfer_negative_when_pretraining_hurts():
    assert compute_forward_transfer(error_from_scratch=4.0,
                                    error_after_pretraining=10.0)         == pytest.approx(-6.0)


def test_forward_transfer_needs_task_B_data_to_differ_between_methods():
    """F20: the old FT saw no task-B data, so every method scored the same.

    The from-scratch arm is shared, so any difference between two methods can
    only come from their own task-B error - which is what makes this one
    method-sensitive and the old one method-blind.
    """
    from_scratch = 10.0
    method_a = compute_forward_transfer(from_scratch, 4.0)
    method_b = compute_forward_transfer(from_scratch, 7.0)
    assert method_a != method_b
