"""Tests for src/benchmark/protocol.py (evaluation dataset construction)."""
import copy

import numpy as np
import pytest
import torch

from src.benchmark.metrics import compute_nll, compute_pf
from src.benchmark.protocol import build_latent_eval_dataset
from src.models.rssm import RSSM
from src.utils.buffer import ReplayBuffer

from conftest import LATENT_DIM, HIDDEN_DIM, ACTION_DIM


@pytest.fixture
def model():
    return RSSM(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                action_dim=ACTION_DIM)


def make_buffer(n_episodes=3, length=6):
    buf = ReplayBuffer(max_episodes=n_episodes, seq_len=2)
    for _ in range(n_episodes):
        buf.add_episode([
            {
                "obs": np.random.rand(64, 64, 3).astype(np.float32),
                "action": np.random.rand(ACTION_DIM).astype(np.float32),
                "reward": 0.0,
                "done": i == length - 1,
            }
            for i in range(length)
        ])
    return buf


def test_returns_latent_transition_triples(model, device):
    ds = build_latent_eval_dataset(model, make_buffer(), device,
                                   n_transitions=7, seed=0)
    assert set(ds) == {"obs", "actions", "next_obs"}
    assert ds["obs"].shape == (7, LATENT_DIM)
    assert ds["next_obs"].shape == (7, LATENT_DIM)
    assert ds["actions"].shape == (7, ACTION_DIM)


def test_is_consumable_by_the_metrics(model, device):
    """The whole point of the helper is to feed compute_nll / compute_pf."""
    ds = build_latent_eval_dataset(model, make_buffer(), device,
                                   n_transitions=10, seed=0)
    nll = compute_nll(model, ds, device)
    assert np.isfinite(nll)
    assert compute_pf(model, copy.deepcopy(model), ds, device) == pytest.approx(
        0.0, abs=1e-5
    )


def test_actions_are_float32_tensors(model, device):
    ds = build_latent_eval_dataset(model, make_buffer(), device,
                                   n_transitions=5, seed=0)
    assert ds["actions"].dtype == torch.float32


def test_caps_at_the_available_number_of_transitions(model, device):
    """3 episodes x 6 steps yields 15 usable (t, t+1) pairs, not 18."""
    ds = build_latent_eval_dataset(model, make_buffer(3, 6), device,
                                   n_transitions=10_000, seed=0)
    assert ds["obs"].shape[0] == 15


def test_is_reproducible_for_a_fixed_seed(model, device):
    buf = make_buffer()
    a = build_latent_eval_dataset(model, buf, device, n_transitions=8, seed=3)
    b = build_latent_eval_dataset(model, buf, device, n_transitions=8, seed=3)
    assert torch.equal(a["obs"], b["obs"])
    assert torch.equal(a["next_obs"], b["next_obs"])
    assert torch.equal(a["actions"], b["actions"])


def test_different_seeds_select_different_transitions(model, device):
    buf = make_buffer()
    a = build_latent_eval_dataset(model, buf, device, n_transitions=8, seed=1)
    b = build_latent_eval_dataset(model, buf, device, n_transitions=8, seed=2)
    assert not torch.equal(a["obs"], b["obs"])


def test_encoding_is_deterministic_regardless_of_model_mode(model, device):
    """Encoding must use the posterior mean, not a sample, or the same buffer
    would yield a different D_i on every call."""
    buf = make_buffer()
    model.eval()
    from_eval = build_latent_eval_dataset(model, buf, device,
                                          n_transitions=8, seed=0)
    model.train()
    from_train = build_latent_eval_dataset(model, buf, device,
                                           n_transitions=8, seed=0)
    assert torch.equal(from_eval["obs"], from_train["obs"])


def test_restores_the_models_training_mode(model, device):
    buf = make_buffer()
    model.train()
    build_latent_eval_dataset(model, buf, device, n_transitions=4, seed=0)
    assert model.training

    model.eval()
    build_latent_eval_dataset(model, buf, device, n_transitions=4, seed=0)
    assert not model.training


def test_next_obs_is_the_following_frame_not_the_same_one(model, device):
    ds = build_latent_eval_dataset(model, make_buffer(), device,
                                   n_transitions=10, seed=0)
    assert not torch.allclose(ds["obs"], ds["next_obs"])


def test_empty_buffer_raises_a_clear_error(model, device):
    empty = ReplayBuffer(max_episodes=2, seq_len=2)
    with pytest.raises(ValueError, match="no episode"):
        build_latent_eval_dataset(model, empty, device)


def test_single_step_episodes_yield_no_transitions(model, device):
    buf = ReplayBuffer(max_episodes=2, seq_len=1)
    buf.add_episode([{
        "obs": np.random.rand(64, 64, 3).astype(np.float32),
        "action": np.random.rand(ACTION_DIM).astype(np.float32),
        "reward": 0.0,
        "done": True,
    }])
    with pytest.raises(ValueError):
        build_latent_eval_dataset(model, buf, device)
