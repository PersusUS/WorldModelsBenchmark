"""
ConvVAE from Ha & Schmidhuber 2018 (World Models), Appendix A.1.
Reference: https://arxiv.org/abs/1803.10122
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(recon: Tensor, target: Tensor) -> Tensor:
    """
    Squared error summed over pixels, averaged over the batch.

    Using `F.mse_loss(..., reduction="mean")` instead divides this term by
    3*64*64 = 12288 while the KL is summed over latent dimensions. That
    under-weights reconstruction by the same factor, and the cheapest way to
    reduce the total loss becomes driving the KL to zero — i.e. ignoring the
    input. The result is a fully collapsed posterior: every observation
    encodes to the same latent, and nothing downstream can learn dynamics.
    """
    return F.mse_loss(recon, target, reduction="sum") / target.shape[0]


class ConvVAE(nn.Module):
    """
    Convolutional VAE for encoding 64x64 RGB observations into a latent space.

    Encoder (input 3x64x64):
        Conv2d(3,32,4,2)  -> 32x31x31
        Conv2d(32,64,4,2) -> 64x14x14
        Conv2d(64,128,4,2)-> 128x6x6
        Conv2d(128,256,4,2)->256x2x2
        Flatten -> Linear(1024, latent_dim*2) -> mu, log_sigma

    Decoder (input latent_dim):
        Linear(latent_dim, 1024) -> Reshape(256,2,2)
        ConvTranspose2d(256,128,5,2) -> 128x7x7
        ConvTranspose2d(128,64,5,2)  -> 64x17x17
        ConvTranspose2d(64,32,6,2)   -> 32x38x38
        ConvTranspose2d(32,3,6,2)    -> 3x80x80
        Sigmoid, then center crop -> 3x64x64
    """

    def __init__(self, latent_dim: int = 32, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),   # -> 32x31x31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64x14x14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # -> 128x6x6
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), # -> 256x2x2
            nn.ReLU(),
        )
        self.enc_fc = nn.Linear(256 * 2 * 2, latent_dim * 2)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256 * 2 * 2)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),  # -> 128x5x5
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),   # -> 64x13x13
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),    # -> 32x30x30
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),     # -> 3x64x64
            nn.Sigmoid(),
        )

    def _encode_params(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """obs: (B, 3, 64, 64) -> mu: (B, latent_dim), log_sigma: (B, latent_dim)"""
        h = self.enc_conv(obs)                       # (B, 256, 2, 2)
        h = h.reshape(h.size(0), -1)                 # (B, 1024)
        params = self.enc_fc(h)                       # (B, latent_dim*2)
        mu, log_sigma = params.chunk(2, dim=-1)       # each (B, latent_dim)
        return mu, log_sigma

    def encode(self, obs: Tensor) -> Tensor:
        """
        obs: (B, 3, 64, 64) -> z: (B, latent_dim)
        Sample from posterior during training, use mu during eval.
        """
        mu, log_sigma = self._encode_params(obs)
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z

    def decode(self, z: Tensor) -> Tensor:
        """z: (B, latent_dim) -> obs_recon: (B, 3, 64, 64)"""
        h = self.dec_fc(z)                            # (B, 1024)
        h = h.reshape(-1, 256, 2, 2)                  # (B, 256, 2, 2)
        recon = self.dec_conv(h)                       # (B, 3, H, W)
        # Center crop to exactly 64x64 if needed
        if recon.shape[2] != 64 or recon.shape[3] != 64:
            ch = (recon.shape[2] - 64) // 2
            cw = (recon.shape[3] - 64) // 2
            recon = recon[:, :, ch:ch+64, cw:cw+64]
        return recon

    def compute_loss(self, obs: Tensor) -> Tuple[Tensor, dict]:
        """
        obs: (B, 3, 64, 64)
        Returns (total_loss, {"reconstruction": float, "kl": float})
        """
        mu, log_sigma = self._encode_params(obs)
        # Reparameterization
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Decode
        recon = self.decode(z)
        # Reconstruction loss: summed over pixels, averaged over batch, so it
        # stays commensurable with the KL term (see reconstruction_loss).
        recon_loss = reconstruction_loss(recon, obs)
        # KL divergence: -0.5 * sum(1 + log_sigma - mu^2 - exp(log_sigma))
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1)
        )
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, {
            "reconstruction": recon_loss.item(),
            "kl": kl_loss.item(),
        }
