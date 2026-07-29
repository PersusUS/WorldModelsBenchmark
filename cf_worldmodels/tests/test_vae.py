"""Tests for src/models/vae.py (ConvVAE)."""
import pytest
import torch
import torch.nn.functional as F

from src.models.vae import ConvVAE, reconstruction_loss

from conftest import LATENT_DIM


def test_encode_params_shapes(obs_batch):
    vae = ConvVAE(latent_dim=LATENT_DIM)
    mu, log_sigma = vae._encode_params(obs_batch)
    assert mu.shape == (2, LATENT_DIM)
    assert log_sigma.shape == (2, LATENT_DIM)


def test_encode_returns_latent(obs_batch):
    vae = ConvVAE(latent_dim=LATENT_DIM)
    z = vae.encode(obs_batch)
    assert z.shape == (2, LATENT_DIM)


def test_decode_returns_64x64(obs_batch):
    """The decoder over-produces 80x80 and center-crops; verify the crop."""
    vae = ConvVAE(latent_dim=LATENT_DIM)
    z = torch.randn(2, LATENT_DIM)
    recon = vae.decode(z)
    assert recon.shape == (2, 3, 64, 64)


def test_decode_output_in_unit_interval():
    """Sigmoid output must match the [0, 1] range of normalized observations."""
    vae = ConvVAE(latent_dim=LATENT_DIM)
    recon = vae.decode(torch.randn(4, LATENT_DIM) * 10)
    assert recon.min() >= 0.0
    assert recon.max() <= 1.0


def test_encode_deterministic_in_eval_mode(obs_batch):
    vae = ConvVAE(latent_dim=LATENT_DIM)
    vae.eval()
    z1 = vae.encode(obs_batch)
    z2 = vae.encode(obs_batch)
    assert torch.allclose(z1, z2)
    mu, _ = vae._encode_params(obs_batch)
    assert torch.allclose(z1, mu)


def test_encode_stochastic_in_train_mode(obs_batch):
    vae = ConvVAE(latent_dim=LATENT_DIM)
    vae.train()
    z1 = vae.encode(obs_batch)
    z2 = vae.encode(obs_batch)
    assert not torch.allclose(z1, z2)


def test_compute_loss_components(obs_batch):
    vae = ConvVAE(latent_dim=LATENT_DIM)
    loss, comps = vae.compute_loss(obs_batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert set(comps) == {"reconstruction", "kl"}
    assert comps["reconstruction"] >= 0.0


def test_compute_loss_backward_reaches_encoder_and_decoder(obs_batch):
    vae = ConvVAE(latent_dim=LATENT_DIM)
    loss, _ = vae.compute_loss(obs_batch)
    loss.backward()
    assert vae.enc_fc.weight.grad is not None
    assert vae.dec_fc.weight.grad is not None
    assert torch.isfinite(vae.enc_fc.weight.grad).all()


def test_reconstruction_loss_sums_over_pixels_and_averages_over_batch():
    """Averaging over pixels instead would divide this term by 12288 while the
    KL stays summed over latent dims, which collapses the posterior."""
    recon = torch.zeros(4, 3, 64, 64)
    target = torch.ones(4, 3, 64, 64)
    assert reconstruction_loss(recon, target).item() == pytest.approx(3 * 64 * 64)


def test_reconstruction_loss_is_not_the_pixelwise_mean():
    recon = torch.zeros(4, 3, 64, 64)
    target = torch.rand(4, 3, 64, 64)
    assert reconstruction_loss(recon, target).item() > 100 * F.mse_loss(
        recon, target
    ).item()


def test_reconstruction_term_dominates_kl_at_initialisation(obs_batch):
    """If the KL outweighs reconstruction from the start, the optimizer's
    cheapest move is to ignore the input."""
    vae = ConvVAE(latent_dim=LATENT_DIM)
    _, comps = vae.compute_loss(obs_batch)
    assert comps["reconstruction"] > comps["kl"]


@pytest.mark.slow
def test_training_does_not_collapse_the_posterior():
    """Regression test for the posterior collapse that made every latent
    identical: the encoder must map different inputs to different codes."""
    torch.manual_seed(0)
    vae = ConvVAE(latent_dim=16)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # Four fixed, clearly distinct images.
    data = torch.stack([
        torch.full((3, 64, 64), 0.1),
        torch.full((3, 64, 64), 0.9),
        torch.cat([torch.full((3, 32, 64), 0.2),
                   torch.full((3, 32, 64), 0.8)], dim=1),
        torch.rand(3, 64, 64),
    ])

    vae.train()
    for _ in range(300):
        loss, _ = vae.compute_loss(data)
        opt.zero_grad()
        loss.backward()
        opt.step()

    vae.eval()
    z = vae.encode(data)
    spread = z.std(dim=0)
    assert (spread > 1e-2).sum() >= 4, (
        f"posterior collapsed: only {(spread > 1e-2).sum()} active dims, "
        f"max per-dim std {spread.max():.2e}"
    )


def test_beta_scales_kl_term(obs_batch):
    """beta must actually weight the KL contribution to the total loss."""
    torch.manual_seed(1)
    vae = ConvVAE(latent_dim=LATENT_DIM, beta=1.0)
    torch.manual_seed(2)
    loss1, comps1 = vae.compute_loss(obs_batch)

    vae.beta = 3.0
    torch.manual_seed(2)
    loss3, comps3 = vae.compute_loss(obs_batch)

    expected_delta = 2.0 * comps1["kl"]
    assert abs((loss3.item() - loss1.item()) - expected_delta) < 1e-4
