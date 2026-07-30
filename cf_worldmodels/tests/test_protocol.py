"""Tests for src/benchmark/protocol.py (evaluation dataset construction)."""
import copy

import numpy as np
import pytest
import torch

from src.benchmark.metrics import compute_nll, compute_pf
from src.benchmark.protocol import (
    build_latent_eval_dataset,
    evaluate_reconstruction,
)
from src.models.rssm import RSSM
from src.models.vae import reconstruction_loss
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


class TestEvaluateReconstruction:
    """
    Pixel-space task-A quality (F17). This is the benchmark's only quality
    signal that is comparable across training budgets, so the properties it
    depends on are worth pinning down.
    """

    def test_returns_a_finite_positive_error(self, model, device):
        value = evaluate_reconstruction(model, make_buffer(), device,
                                        n_frames=10, seed=0)
        assert np.isfinite(value)
        assert value > 0.0

    def test_is_on_the_same_scale_as_the_training_loss(self, model, device):
        """
        It must be comparable with the "reconstruction" component of
        compute_loss, or reporting the two side by side is misleading.
        """
        buf = make_buffer(n_episodes=1, length=4)
        frames = np.stack([s["obs"] for s in buf.episodes[0]])
        x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2)

        model.eval()
        with torch.no_grad():
            z = model.encode(x)
            h = torch.zeros(x.shape[0], model.hidden_dim, device=device)
            expected = float(reconstruction_loss(model.decode(h, z), x).item())

        value = evaluate_reconstruction(model, buf, device, n_frames=4, seed=0)
        assert value == pytest.approx(expected, rel=1e-5)

    def test_chunking_does_not_change_the_result(self, model, device):
        """Frames are encoded in chunks to bound memory; the mean must not
        depend on the chunk size."""
        buf = make_buffer(n_episodes=3, length=6)
        one_shot = evaluate_reconstruction(model, buf, device, n_frames=18,
                                           seed=0, chunk_size=64)
        chunked = evaluate_reconstruction(model, buf, device, n_frames=18,
                                          seed=0, chunk_size=5)
        assert chunked == pytest.approx(one_shot, rel=1e-5)

    def test_is_deterministic_for_a_fixed_seed(self, model, device):
        buf = make_buffer()
        model.train()
        a = evaluate_reconstruction(model, buf, device, n_frames=8, seed=1)
        b = evaluate_reconstruction(model, buf, device, n_frames=8, seed=1)
        assert a == b

    def test_restores_the_models_training_mode(self, model, device):
        buf = make_buffer()
        model.train()
        evaluate_reconstruction(model, buf, device, n_frames=4, seed=0)
        assert model.training

        model.eval()
        evaluate_reconstruction(model, buf, device, n_frames=4, seed=0)
        assert not model.training

    def test_caps_at_the_available_number_of_frames(self, model, device):
        """Asking for more frames than exist must not raise."""
        value = evaluate_reconstruction(model, make_buffer(2, 3), device,
                                        n_frames=10_000, seed=0)
        assert np.isfinite(value)

    def test_empty_buffer_raises_a_clear_error(self, model, device):
        empty = ReplayBuffer(max_episodes=2, seq_len=2)
        with pytest.raises(ValueError, match="no frames"):
            evaluate_reconstruction(model, empty, device)

    def test_training_lowers_it(self, model, device):
        """
        The signal has to move in the right direction, or it cannot serve as
        evidence that task A was learned.
        """
        buf = make_buffer(n_episodes=2, length=4)
        before = evaluate_reconstruction(model, buf, device, n_frames=8, seed=0)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        frames = np.stack([s["obs"] for ep in buf.episodes for s in ep])
        x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2)
        model.train()
        for _ in range(30):
            loss, _ = model.vae.compute_loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()

        after = evaluate_reconstruction(model, buf, device, n_frames=8, seed=0)
        assert after < before


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
