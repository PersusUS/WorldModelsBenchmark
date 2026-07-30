"""Tests for the continual-learning baselines in src/baselines/."""
import copy

import pytest
import torch

from src.baselines.ewc import EWCWorldModel
from src.baselines.finetuning import FineTuningWorldModel
from src.baselines.progressive_nets import ProgressiveNetWorldModel
from src.baselines.replay import InfiniteReplayWorldModel
from src.models.rssm import BaseWorldModel

from conftest import LATENT_DIM, HIDDEN_DIM, ACTION_DIM

BASELINES = [
    FineTuningWorldModel,
    InfiniteReplayWorldModel,
    EWCWorldModel,
    ProgressiveNetWorldModel,
]


def build(cls):
    return cls(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)


# --- shared contract -------------------------------------------------------

@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.__name__)
def test_implements_base_world_model(cls):
    assert isinstance(build(cls), BaseWorldModel)


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.__name__)
def test_exposes_dimensions_used_by_metrics(cls):
    """compute_nll/compute_rd read hidden_dim and action_dim off the model."""
    model = build(cls)
    assert model.hidden_dim == HIDDEN_DIM
    assert model.action_dim == ACTION_DIM
    assert model.latent_dim == LATENT_DIM


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.__name__)
def test_transition_and_predict_stoch_shapes(cls):
    model = build(cls)
    h = torch.zeros(4, HIDDEN_DIM)
    z = torch.randn(4, LATENT_DIM)
    a = torch.randn(4, ACTION_DIM)

    h_next = model.transition(h, z, a)
    assert h_next.shape == (4, HIDDEN_DIM)

    mu, log_sigma = model.predict_stoch(h_next)
    assert mu.shape == (4, LATENT_DIM)
    assert log_sigma.shape == (4, LATENT_DIM)


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.__name__)
def test_encode_and_decode_shapes(cls, obs_batch):
    model = build(cls)
    z = model.encode(obs_batch)
    assert z.shape == (2, LATENT_DIM)
    assert model.decode(torch.zeros(2, HIDDEN_DIM), z).shape == (2, 3, 64, 64)


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.__name__)
def test_compute_loss_is_finite_and_differentiable(cls, seq_batch):
    model = build(cls)
    loss, comps = model.compute_loss(seq_batch)
    assert torch.isfinite(loss)
    assert "reconstruction" in comps and "kl" in comps

    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
    )


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.__name__)
def test_get_uncertainty_shape(cls):
    model = build(cls)
    u = model.get_uncertainty(
        torch.zeros(5, HIDDEN_DIM), torch.randn(5, LATENT_DIM),
        torch.randn(5, ACTION_DIM),
    )
    assert u.shape == (5,)


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.__name__)
def test_baselines_report_zero_uncertainty(cls):
    """Only UG-MTM estimates uncertainty; baselines must return exact zeros."""
    model = build(cls)
    u = model.get_uncertainty(
        torch.zeros(5, HIDDEN_DIM), torch.randn(5, LATENT_DIM),
        torch.randn(5, ACTION_DIM),
    )
    assert torch.all(u == 0)


# --- fine-tuning -----------------------------------------------------------

def test_finetuning_updates_all_weights(seq_batch):
    """The lower bound must have nothing frozen."""
    model = build(FineTuningWorldModel)
    before = model.rssm.gru.weight_ih.detach().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss, _ = model.compute_loss(seq_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.allclose(before, model.rssm.gru.weight_ih)


# --- EWC -------------------------------------------------------------------

def test_ewc_penalty_is_zero_before_consolidation():
    model = build(EWCWorldModel)
    assert model.ewc_loss().item() == pytest.approx(0.0)


def test_ewc_consolidate_stores_fisher_and_optimal_params(latent_dataset, device):
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)

    assert len(model._fisher_diags) == 1
    assert len(model._optimal_params) == 1
    names = {n for n, _ in model.rssm.named_parameters()}
    assert set(model._fisher_diags[0]) == names
    assert set(model._optimal_params[0]) == names


def test_ewc_consolidate_requires_the_next_state_target(latent_dataset, device):
    """The Fisher diagonal is defined over log P(z'|z, a)."""
    model = build(EWCWorldModel)
    del latent_dataset["next_obs"]
    with pytest.raises(KeyError, match="next_obs"):
        model.consolidate(latent_dataset, device)


def test_ewc_fisher_is_non_negative(latent_dataset, device):
    """Fisher is a mean of squared gradients and can never go negative."""
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    for tensor in model._fisher_diags[0].values():
        assert torch.all(tensor >= 0)


def test_ewc_fisher_is_non_trivial_for_transition_params(latent_dataset, device):
    """A Fisher matrix of all zeros would make EWC a no-op."""
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    fisher = model._fisher_diags[0]
    assert fisher["gru.weight_ih"].sum().item() > 0
    assert fisher["stoch_fc.weight"].sum().item() > 0


def _transition_nll(model, obs, actions, next_obs):
    """-log P(z' | z, a), per sample, exactly as EWC's Fisher defines it."""
    h = torch.zeros(obs.shape[0], model.hidden_dim)
    h_next = model.rssm.transition(h, obs, actions)
    mu, log_sigma = model.rssm.predict_stoch(h_next)
    log_prob = -0.5 * torch.sum(
        torch.log(2 * torch.tensor(torch.pi))
        + log_sigma
        + (next_obs - mu).pow(2) / (log_sigma.exp() + 1e-8),
        dim=-1,
    )
    return -log_prob


def _squared_grads(model, loss):
    model.rssm.zero_grad()
    loss.backward()
    # The VAE parameters get no gradient from a transition-only likelihood.
    return {
        name: (torch.zeros_like(param) if param.grad is None
               else param.grad.detach().pow(2))
        for name, param in model.rssm.named_parameters()
    }


def test_ewc_fisher_is_the_mean_of_squared_per_sample_gradients(
        latent_dataset, device):
    """F = E[g^2] over single transitions, computed here from scratch."""
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    fisher = model._fisher_diags[0]

    n = latent_dataset["obs"].shape[0]
    expected = {
        name: torch.zeros_like(param)
        for name, param in model.rssm.named_parameters()
    }
    for i in range(n):
        nll = _transition_nll(
            model,
            latent_dataset["obs"][i:i + 1],
            latent_dataset["actions"][i:i + 1],
            latent_dataset["next_obs"][i:i + 1],
        ).squeeze(0)
        for name, squared in _squared_grads(model, nll).items():
            expected[name] += squared / n

    for name, value in expected.items():
        assert torch.allclose(fisher[name], value, atol=1e-10)


def test_ewc_fisher_keeps_the_gradient_variance(latent_dataset, device):
    """Regression for I3: the Fisher was (E[g])^2, which throws away Var[g].

    Squaring a mini-batch gradient keeps only the component that survives
    averaging. E[g^2] = (E[g])^2 + Var[g], and on real data the variance term
    dominates — so the buggy estimator underweighted every parameter, letting
    EWC's penalty collapse towards a no-op.
    """
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    fisher = model._fisher_diags[0]

    batch_nll = _transition_nll(
        model,
        latent_dataset["obs"],
        latent_dataset["actions"],
        latent_dataset["next_obs"],
    ).mean()
    batch_squared = _squared_grads(model, batch_nll)

    for name, value in fisher.items():
        # Jensen: the per-sample Fisher can never fall below (E[g])^2.
        assert torch.all(value >= batch_squared[name] - 1e-10)

    # And on the transition parameters it must be far above it, not equal.
    for name in ("gru.weight_ih", "stoch_fc.weight"):
        assert fisher[name].sum() > 5.0 * batch_squared[name].sum()


def test_ewc_fisher_is_zero_on_the_encoder(latent_dataset, device):
    """EWC protects M and nothing else, and the Fisher says so out loud.

    The Fisher is defined over log P(z'|z, a), which the VAE parameters do not
    enter, so their Fisher is exactly zero and the penalty never constrains
    them. This is why EWC's pixel-space reconstruction of task A degrades just
    like fine-tuning's (F18/P9): the encoder is outside its scope by
    construction, not by accident.
    """
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    fisher = model._fisher_diags[0]

    encoder = [name for name in fisher if name.startswith("vae.")]
    assert encoder, "the RSSM is expected to own the VAE"
    for name in encoder:
        assert torch.all(fisher[name] == 0)


def test_ewc_fisher_is_zero_on_the_recurrent_weights(latent_dataset, device):
    """Known limitation, locked in so it cannot change silently.

    The Fisher set is single transitions started from h = 0, so `gru.weight_hh`
    contributes nothing to the likelihood and receives no Fisher mass: the
    recurrent pathway is unprotected. compute_nll scores from h = 0 too, so PF
    cannot see it either — but RD rolls out 15 steps and does.
    """
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    assert torch.all(model._fisher_diags[0]["gru.weight_hh"] == 0)
    assert model._fisher_diags[0]["gru.weight_ih"].sum() > 0


def test_ewc_penalty_stays_zero_at_the_consolidated_optimum(latent_dataset, device):
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    assert model.ewc_loss().item() == pytest.approx(0.0, abs=1e-8)


def test_ewc_penalty_grows_as_weights_drift(latent_dataset, device):
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)

    with torch.no_grad():
        model.rssm.gru.weight_ih.add_(0.1)
    small = model.ewc_loss().item()

    with torch.no_grad():
        model.rssm.gru.weight_ih.add_(0.4)
    large = model.ewc_loss().item()

    assert 0.0 < small < large


def test_ewc_penalty_scales_with_lambda(latent_dataset, device):
    model = EWCWorldModel(LATENT_DIM, HIDDEN_DIM, ACTION_DIM, ewc_lambda=1.0)
    model.consolidate(latent_dataset, device)
    with torch.no_grad():
        model.rssm.gru.weight_ih.add_(0.1)

    base = model.ewc_loss().item()
    model.ewc_lambda = 10.0
    assert model.ewc_loss().item() == pytest.approx(10.0 * base, rel=1e-5)


def test_ewc_accumulates_one_penalty_term_per_task(latent_dataset, device):
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    model.consolidate(latent_dataset, device)
    assert len(model._fisher_diags) == 2


def test_ewc_compute_loss_includes_penalty(seq_batch, latent_dataset, device):
    model = build(EWCWorldModel)
    _, comps_before = model.compute_loss(seq_batch)
    assert comps_before["ewc_penalty"] == pytest.approx(0.0)

    model.consolidate(latent_dataset, device)
    with torch.no_grad():
        model.rssm.gru.weight_ih.add_(0.2)

    _, comps_after = model.compute_loss(seq_batch)
    assert comps_after["ewc_penalty"] > 0.0


def test_ewc_penalty_is_differentiable(latent_dataset, device):
    model = build(EWCWorldModel)
    model.consolidate(latent_dataset, device)
    with torch.no_grad():
        model.rssm.gru.weight_ih.add_(0.2)

    model.zero_grad()
    model.ewc_loss().backward()
    assert model.rssm.gru.weight_ih.grad is not None
    assert model.rssm.gru.weight_ih.grad.abs().sum().item() > 0


# --- Progressive Networks --------------------------------------------------

def test_progressive_starts_with_one_column():
    model = build(ProgressiveNetWorldModel)
    assert model.num_columns == 1
    assert model._active_column == 0


def test_progressive_add_column_grows_the_network():
    model = build(ProgressiveNetWorldModel)
    model.add_column()

    assert model.num_columns == 2
    assert model._active_column == 1
    assert len(model.laterals) == 1
    assert len(model.stoch_fcs) == 2


def test_progressive_add_column_freezes_previous_columns():
    """Frozen columns are what make Progressive Nets forgetting-free."""
    model = build(ProgressiveNetWorldModel)
    model.add_column()

    assert not any(p.requires_grad for p in model.columns[0].parameters())
    assert not any(p.requires_grad for p in model.stoch_fcs[0].parameters())
    assert all(p.requires_grad for p in model.columns[1].parameters())


def test_progressive_new_column_accepts_lateral_input():
    model = build(ProgressiveNetWorldModel)
    model.add_column()
    assert model.columns[1].weight_ih.shape[1] == (
        LATENT_DIM + ACTION_DIM + HIDDEN_DIM
    )


def test_progressive_transition_works_after_adding_a_column():
    model = build(ProgressiveNetWorldModel)
    model.add_column()

    h = torch.zeros(3, HIDDEN_DIM)
    z = torch.randn(3, LATENT_DIM)
    a = torch.randn(3, ACTION_DIM)
    assert model.transition(h, z, a).shape == (3, HIDDEN_DIM)


def test_progressive_predict_stoch_follows_active_column():
    model = build(ProgressiveNetWorldModel)
    h = torch.randn(3, HIDDEN_DIM)
    first_mu, _ = model.predict_stoch(h)

    model.add_column()
    second_mu, _ = model.predict_stoch(h)

    assert not torch.allclose(first_mu, second_mu)


def test_progressive_training_leaves_old_column_unchanged(seq_batch):
    model = build(ProgressiveNetWorldModel)
    model.add_column()
    frozen_before = model.columns[0].weight_ih.detach().clone()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-2)
    loss, _ = model.compute_loss(seq_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.allclose(frozen_before, model.columns[0].weight_ih)


def test_progressive_stacking_three_columns():
    model = build(ProgressiveNetWorldModel)
    model.add_column()
    model.add_column()

    assert model.num_columns == 3
    assert len(model.laterals) == 2
    h = torch.zeros(2, HIDDEN_DIM)
    z = torch.randn(2, LATENT_DIM)
    a = torch.randn(2, ACTION_DIM)
    assert model.transition(h, z, a).shape == (2, HIDDEN_DIM)


# --- Infinite Replay -------------------------------------------------------

def test_replay_buffer_starts_empty():
    model = build(InfiniteReplayWorldModel)
    assert model.buffer_size() == 0
    assert model._all_episodes == []


def test_replay_accumulates_episodes_across_tasks(episode):
    model = build(InfiniteReplayWorldModel)
    model.add_task_data(task_id=0, episodes=[episode, episode])
    model.add_task_data(task_id=1, episodes=[episode])

    assert len(model._all_episodes) == 3
    assert model.buffer_size() == 3


def test_replay_never_evicts_earlier_task_data(episode):
    """'Infinite' replay is the upper bound: task-A data must persist."""
    model = build(InfiniteReplayWorldModel)
    model.add_task_data(task_id=0, episodes=[episode])
    first = model._all_episodes[0]

    for task_id in range(1, 6):
        model.add_task_data(task_id=task_id, episodes=[episode] * 10)

    assert model._all_episodes[0] is first


def test_replay_add_task_data_accepts_explicit_sample_count():
    model = build(InfiniteReplayWorldModel)
    model.add_task_data(task_id=0, n_samples=250)
    assert model.buffer_size() == 250
