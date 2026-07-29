"""Shared pytest fixtures for the cf_worldmodels test suite.

Tests use deliberately small latent/hidden dimensions so the whole suite runs
on CPU in a few seconds. The ConvVAE architecture is fixed to 64x64 RGB
inputs, so observation shapes are always (B, 3, 64, 64).
"""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

# Make `src` importable regardless of where pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LATENT_DIM = 8
HIDDEN_DIM = 16
ACTION_DIM = 3


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Seed every test so failures are reproducible."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dims():
    return LATENT_DIM, HIDDEN_DIM, ACTION_DIM


@pytest.fixture
def obs_batch():
    """A batch of 2 observations, (B, 3, 64, 64) in [0, 1]."""
    return torch.rand(2, 3, 64, 64)


@pytest.fixture
def seq_batch():
    """A (B, T, 3, 64, 64) sequence batch plus matching actions."""
    return {
        "obs": torch.rand(2, 3, 3, 64, 64),
        "actions": torch.rand(2, 3, ACTION_DIM),
    }


@pytest.fixture
def latent_dataset():
    """Evaluation dataset in the shape the benchmark metrics expect.

    Mirrors what protocol.build_latent_eval_dataset produces: (z, a, z')
    triples. `next_obs` is the target the NLL is scored against.
    """
    return {
        "obs": torch.randn(16, LATENT_DIM),
        "actions": torch.randn(16, ACTION_DIM),
        "next_obs": torch.randn(16, LATENT_DIM),
    }


@pytest.fixture
def ug_cfg():
    """Minimal UG-MTM config matching configs/models/ug_mtm.yaml's schema."""
    return OmegaConf.create(
        {
            "model": {
                "num_experts": 4,
                "latent_dim": LATENT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "action_dim": ACTION_DIM,
                "mc_dropout_T": 3,
                "gate_lambda": 10.0,
                "threshold_grad": 0.1,
                "threshold_window": 8,
                "dropout_rate": 0.1,
                "beta_kl": 1.0,
            }
        }
    )


@pytest.fixture
def episode():
    """A single 10-step episode in ReplayBuffer format."""
    import numpy as np

    return [
        {
            "obs": np.random.rand(64, 64, 3).astype(np.float32),
            "action": np.random.rand(ACTION_DIM).astype(np.float32),
            "reward": float(i),
            "done": i == 9,
        }
        for i in range(10)
    ]
