"""Tests for src/utils/checkpointing.py and src/utils/logging_utils.py."""
import json

import pytest
import torch

from src.models.rssm import RSSM
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.logging_utils import log_metrics, save_metrics

from conftest import LATENT_DIM, HIDDEN_DIM, ACTION_DIM


def build():
    return RSSM(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                action_dim=ACTION_DIM)


@pytest.fixture
def checkpoint_path(tmp_path):
    return tmp_path / "nested" / "checkpoint_final.pt"


def test_save_creates_parent_directories(checkpoint_path):
    model = build()
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, step=1, task_id=0, config={},
                    metrics={}, path=checkpoint_path)
    assert checkpoint_path.exists()


def test_checkpoint_has_the_documented_schema(checkpoint_path):
    model = build()
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, step=7, task_id=1,
                    config={"model": {"latent_dim": LATENT_DIM}},
                    metrics={"nll": 1.5, "kl": 0.5, "reconstruction": 1.0},
                    path=checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert set(ckpt) == {
        "step", "task_id", "model_state_dict", "optimizer_state_dict",
        "config", "metrics",
    }
    assert ckpt["step"] == 7
    assert ckpt["task_id"] == 1
    assert ckpt["config"]["model"]["latent_dim"] == LATENT_DIM
    assert set(ckpt["metrics"]) == {"nll", "kl", "reconstruction"}


def test_missing_metrics_default_to_zero(checkpoint_path):
    model = build()
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, step=0, task_id=0, config={},
                    metrics={"nll": 2.0}, path=checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert ckpt["metrics"]["nll"] == pytest.approx(2.0)
    assert ckpt["metrics"]["kl"] == pytest.approx(0.0)


def test_roundtrip_restores_model_weights(checkpoint_path):
    saved = build()
    optimizer = torch.optim.Adam(saved.parameters())
    save_checkpoint(saved, optimizer, step=0, task_id=0, config={},
                    metrics={}, path=checkpoint_path)

    restored = build()
    assert not torch.allclose(saved.gru.weight_ih, restored.gru.weight_ih)

    load_checkpoint(checkpoint_path, restored, device=torch.device("cpu"))
    for (name, a), (_, b) in zip(saved.state_dict().items(),
                                 restored.state_dict().items()):
        assert torch.allclose(a, b), name


def test_roundtrip_restores_optimizer_state(checkpoint_path, seq_batch):
    model = build()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, _ = model.compute_loss(seq_batch)
    loss.backward()
    optimizer.step()

    save_checkpoint(model, optimizer, step=1, task_id=0, config={},
                    metrics={}, path=checkpoint_path)

    fresh_model = build()
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    assert fresh_optimizer.state_dict()["state"] == {}

    load_checkpoint(checkpoint_path, fresh_model, fresh_optimizer,
                    device=torch.device("cpu"))
    assert fresh_optimizer.state_dict()["state"] != {}


def test_load_returns_the_full_checkpoint_dict(checkpoint_path):
    model = build()
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, step=42, task_id=3, config={"a": 1},
                    metrics={"nll": 0.25}, path=checkpoint_path)

    ckpt = load_checkpoint(checkpoint_path, build(), device=torch.device("cpu"))
    assert ckpt["step"] == 42
    assert ckpt["task_id"] == 3


# --- logging utils ---------------------------------------------------------

def test_save_metrics_writes_readable_json(tmp_path):
    path = tmp_path / "run" / "metrics.json"
    metrics = {"wmf": 0.5, "ft": -0.1, "pf": 0.3, "rd": 0.7, "pis": 0.0}
    save_metrics(metrics, path)

    assert json.loads(path.read_text()) == metrics


def test_save_metrics_emits_the_keys_the_plotting_scripts_read(tmp_path):
    path = tmp_path / "metrics.json"
    save_metrics(
        {"wmf": 0.1, "ft": 0.2, "pf": 0.3, "rd": 0.4, "pis": 0.0,
         "initial_reconstruction_loss": 1.0,
         "final_reconstruction_loss": 0.5},
        path,
    )
    loaded = json.loads(path.read_text())
    assert {"wmf", "ft", "pf", "rd", "pis"} <= set(loaded)


def test_log_metrics_without_wandb_is_a_noop():
    log_metrics(step=1, metrics={"loss": 1.0}, wandb_run=None, prefix="train")
