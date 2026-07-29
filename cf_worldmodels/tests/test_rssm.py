"""Tests for src/models/rssm.py (BaseWorldModel contract + RSSM baseline)."""
import pytest
import torch

from src.models.rssm import RSSM, BaseWorldModel

from conftest import LATENT_DIM, HIDDEN_DIM, ACTION_DIM


@pytest.fixture
def rssm():
    return RSSM(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                action_dim=ACTION_DIM)


def test_implements_base_world_model(rssm):
    assert isinstance(rssm, BaseWorldModel)


def test_base_world_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseWorldModel()


def test_transition_shape(rssm):
    h = torch.zeros(4, HIDDEN_DIM)
    z = torch.randn(4, LATENT_DIM)
    a = torch.randn(4, ACTION_DIM)
    assert rssm.transition(h, z, a).shape == (4, HIDDEN_DIM)


def test_transition_depends_on_hidden_state(rssm):
    """A GRU cell must carry state; identical inputs with different h differ."""
    z = torch.randn(4, LATENT_DIM)
    a = torch.randn(4, ACTION_DIM)
    h1 = rssm.transition(torch.zeros(4, HIDDEN_DIM), z, a)
    h2 = rssm.transition(torch.ones(4, HIDDEN_DIM), z, a)
    assert not torch.allclose(h1, h2)


def test_predict_stoch_shapes(rssm):
    mu, log_sigma = rssm.predict_stoch(torch.randn(4, HIDDEN_DIM))
    assert mu.shape == (4, LATENT_DIM)
    assert log_sigma.shape == (4, LATENT_DIM)


def test_sample_stoch_uses_mean_in_eval(rssm):
    rssm.eval()
    h = torch.randn(4, HIDDEN_DIM)
    mu, _ = rssm.predict_stoch(h)
    assert torch.allclose(rssm.sample_stoch(h), mu)


def test_sample_stoch_is_stochastic_in_train(rssm):
    rssm.train()
    h = torch.randn(4, HIDDEN_DIM)
    assert not torch.allclose(rssm.sample_stoch(h), rssm.sample_stoch(h))


def test_encode_decode_roundtrip_shapes(rssm, obs_batch):
    z = rssm.encode(obs_batch)
    assert z.shape == (2, LATENT_DIM)
    h = torch.zeros(2, HIDDEN_DIM)
    assert rssm.decode(h, z).shape == (2, 3, 64, 64)


def test_compute_loss_returns_finite_components(rssm, seq_batch):
    loss, comps = rssm.compute_loss(seq_batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert set(comps) == {"reconstruction", "kl", "nll"}
    for value in comps.values():
        assert isinstance(value, float)


def test_compute_loss_backward_reaches_gru_and_vae(rssm, seq_batch):
    loss, _ = rssm.compute_loss(seq_batch)
    loss.backward()
    assert rssm.gru.weight_ih.grad is not None
    assert rssm.stoch_fc.weight.grad is not None
    assert rssm.vae.enc_fc.weight.grad is not None


def test_get_uncertainty_is_zero_for_baseline(rssm):
    h = torch.randn(5, HIDDEN_DIM)
    z = torch.randn(5, LATENT_DIM)
    a = torch.randn(5, ACTION_DIM)
    u = rssm.get_uncertainty(h, z, a)
    assert u.shape == (5,)
    assert torch.all(u == 0)
