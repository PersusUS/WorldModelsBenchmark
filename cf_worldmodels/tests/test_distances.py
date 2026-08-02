"""Tests for src/benchmark/distances.py (d_param, d_trans)."""
import copy

import pytest
import torch

from src.benchmark.distances import (
    PHYSICS_PARAMS,
    compute_d_param,
    compute_d_trans,
    d_param_for_pair,
)
from src.models.rssm import RSSM

from conftest import LATENT_DIM, HIDDEN_DIM, ACTION_DIM


# --- d_param ---------------------------------------------------------------

def test_d_param_is_zero_for_identical_configs():
    cfg = {"gravity": 9.8, "mass_scale": 1.0, "friction_scale": 1.0}
    assert compute_d_param(cfg, dict(cfg)) == pytest.approx(0.0)


def test_d_param_known_value():
    """d = ||phi_A - phi_B|| / ||phi_A|| with phi = [g, mass, friction]."""
    a = {"gravity": 3.0, "mass_scale": 4.0, "friction_scale": 0.0}
    b = {"gravity": 0.0, "mass_scale": 4.0, "friction_scale": 0.0}
    # ||a - b|| = 3, ||a|| = 5
    assert compute_d_param(a, b) == pytest.approx(0.6)


def test_d_param_fills_defaults_for_missing_keys():
    """An empty config must fall back to the reference physics values."""
    default = {"gravity": 9.8, "mass_scale": 1.0, "friction_scale": 1.0}
    assert compute_d_param({}, default) == pytest.approx(0.0)


def test_d_param_is_not_symmetric():
    """Normalizing by ||phi_A|| makes the measure directional, by design."""
    a = {"gravity": 9.8, "mass_scale": 1.0, "friction_scale": 1.0}
    b = {"gravity": 4.0, "mass_scale": 3.0, "friction_scale": 0.5}
    assert compute_d_param(a, b) != pytest.approx(compute_d_param(b, a))


def test_d_param_grows_with_parameter_separation():
    ref = {"gravity": 9.8, "mass_scale": 1.0, "friction_scale": 1.0}
    near = {"gravity": 7.0, "mass_scale": 1.0, "friction_scale": 1.0}
    far = {"gravity": 4.0, "mass_scale": 1.0, "friction_scale": 1.0}
    assert compute_d_param(ref, near) < compute_d_param(ref, far)


def test_d_param_guards_against_zero_norm_reference():
    zero = {"gravity": 0.0, "mass_scale": 0.0, "friction_scale": 0.0}
    other = {"gravity": 1.0, "mass_scale": 1.0, "friction_scale": 1.0}
    assert compute_d_param(zero, other) == 0.0


def test_d_param_matches_benchmark_config_ordering():
    """The three Gymnasium levels must be monotonically increasing in d_param."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/benchmark/gymnasium.yaml")
    values = []
    for level in ["distance_min", "distance_med", "distance_max"]:
        seq = cfg.benchmark.sequences[level]
        values.append(
            compute_d_param(dict(seq.task_A.params), dict(seq.task_B.params))
        )
    assert values == sorted(values)
    assert values[0] > 0.0


# --- d_param_for_pair: where Eq. 8 says nothing -----------------------------

def test_d_param_for_pair_is_none_when_the_environment_itself_changes():
    """cheetah -> walker changes everything and leaves the physics vector at
    its default on both sides, so Eq. 8 would score the largest construction
    change in that family as a zero. That gap is why Eq. 9 exists."""
    swap = {"task_A": {"domain_name": "cheetah", "task_name": "run",
                       "params": {}},
            "task_B": {"domain_name": "walker", "task_name": "run",
                       "params": {}}}
    assert d_param_for_pair(swap) is None
    # And the trap it exists to avoid: the raw equation happily returns 0.
    assert compute_d_param({}, {}) == pytest.approx(0.0)


def test_d_param_for_pair_scores_a_pure_physics_perturbation():
    physics = {"task_A": {"env_id": "HalfCheetah-v4",
                          "params": {"gravity": 9.8, "mass_scale": 1.0,
                                     "friction_scale": 1.0}},
               "task_B": {"env_id": "HalfCheetah-v4",
                          "params": {"gravity": 4.0, "mass_scale": 1.0,
                                     "friction_scale": 1.0}}}
    assert d_param_for_pair(physics) == pytest.approx(0.5858, abs=1e-4)


def test_physics_params_is_the_vector_the_equation_differences():
    """The key list is shared with the paper's task-table labels; if the two
    ever disagree, a perturbation gets scored and not printed."""
    assert list(PHYSICS_PARAMS) == ["gravity", "mass_scale", "friction_scale"]


# --- d_trans ---------------------------------------------------------------

@pytest.fixture
def model():
    return RSSM(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                action_dim=ACTION_DIM)


def test_d_trans_is_zero_for_identical_models(model, latent_dataset, device):
    twin = copy.deepcopy(model)
    d = compute_d_trans(model, twin, latent_dataset, device)
    assert d == pytest.approx(0.0, abs=1e-5)


def test_d_trans_is_positive_for_different_models(model, latent_dataset, device):
    other = copy.deepcopy(model)
    with torch.no_grad():
        for param in other.parameters():
            param.add_(torch.randn_like(param) * 0.2)

    d = compute_d_trans(model, other, latent_dataset, device)
    assert d > 0.0
    assert torch.isfinite(torch.tensor(d))


def test_d_trans_matches_torch_kl_reference(model, latent_dataset, device):
    """Regression for F13: d_trans (Eq. 9) must be the exact diagonal-Gaussian
    KL under the log-variance convention, not a hand-written approximation."""
    other = copy.deepcopy(model)
    with torch.no_grad():
        for param in other.parameters():
            param.add_(torch.randn_like(param) * 0.2)

    obs = latent_dataset["obs"].to(device)
    actions = latent_dataset["actions"].to(device)
    n = obs.shape[0]

    with torch.no_grad():
        mu_A, log_var_A = model.predict_stoch(
            model.transition(torch.zeros(n, model.hidden_dim, device=device),
                             obs, actions)
        )
        mu_B, log_var_B = other.predict_stoch(
            other.transition(torch.zeros(n, other.hidden_dim, device=device),
                             obs, actions)
        )
        reference = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_A, torch.exp(0.5 * log_var_A)),
            torch.distributions.Normal(mu_B, torch.exp(0.5 * log_var_B)),
        ).sum(dim=-1).mean().item()

    assert compute_d_trans(model, other, latent_dataset, device) == pytest.approx(
        reference, rel=1e-6
    )


def test_d_trans_leaves_models_in_eval_mode(model, latent_dataset, device):
    model.train()
    other = copy.deepcopy(model)
    compute_d_trans(model, other, latent_dataset, device)
    assert not model.training
    assert not other.training
