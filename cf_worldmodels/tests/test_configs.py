"""Tests that the shipped configs stay consistent with the code that reads them."""
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.models.ug_mtm import UG_MTM

BENCHMARK_CONFIGS = [
    "configs/benchmark/minigrid.yaml",
    "configs/benchmark/gymnasium.yaml",
    "configs/benchmark/dmcontrol.yaml",
]
MODEL_CONFIGS = [
    "configs/models/rssm_baseline.yaml",
    "configs/models/ug_mtm.yaml",
]
DISTANCE_LEVELS = ["distance_min", "distance_med", "distance_max"]


@pytest.mark.parametrize("path", BENCHMARK_CONFIGS + MODEL_CONFIGS)
def test_config_file_exists_and_parses(path):
    assert Path(path).exists(), f"missing config: {path}"
    assert OmegaConf.load(path) is not None


@pytest.mark.parametrize("path", BENCHMARK_CONFIGS)
def test_benchmark_config_defines_all_three_distance_levels(path):
    cfg = OmegaConf.load(path)
    assert set(cfg.benchmark.sequences) == set(DISTANCE_LEVELS)


@pytest.mark.parametrize("path", BENCHMARK_CONFIGS)
def test_every_sequence_defines_both_tasks(path):
    cfg = OmegaConf.load(path)
    for level in DISTANCE_LEVELS:
        seq = cfg.benchmark.sequences[level]
        assert "task_A" in seq, f"{path}:{level}"
        assert "task_B" in seq, f"{path}:{level}"


@pytest.mark.parametrize("path", BENCHMARK_CONFIGS)
def test_benchmark_config_declares_five_seeds(path):
    cfg = OmegaConf.load(path)
    assert list(cfg.protocol.seeds) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("path", ["configs/benchmark/minigrid.yaml",
                                  "configs/benchmark/gymnasium.yaml"])
def test_gym_style_families_identify_tasks_by_env_id(path):
    cfg = OmegaConf.load(path)
    for level in DISTANCE_LEVELS:
        seq = cfg.benchmark.sequences[level]
        assert "env_id" in seq.task_A
        assert "env_id" in seq.task_B


def test_dmcontrol_tasks_use_domain_and_task_names():
    cfg = OmegaConf.load("configs/benchmark/dmcontrol.yaml")
    for level in DISTANCE_LEVELS:
        seq = cfg.benchmark.sequences[level]
        for task in [seq.task_A, seq.task_B]:
            assert "domain_name" in task
            assert "task_name" in task


def test_gymnasium_physics_params_are_complete():
    """compute_d_param reads all three keys off every Gymnasium task."""
    cfg = OmegaConf.load("configs/benchmark/gymnasium.yaml")
    for level in DISTANCE_LEVELS:
        seq = cfg.benchmark.sequences[level]
        for task in [seq.task_A, seq.task_B]:
            assert set(task.params) == {"gravity", "mass_scale",
                                        "friction_scale"}


@pytest.mark.parametrize("path", BENCHMARK_CONFIGS)
def test_benchmark_config_action_dim_matches_its_family(path):
    """MiniGrid exposes 7 discrete actions; the MuJoCo families use 6-dim
    continuous actions."""
    cfg = OmegaConf.load(path)
    expected = 7 if cfg.benchmark.family == "minigrid" else 6
    assert cfg.model.action_dim == expected


def test_ug_mtm_config_supplies_every_field_the_model_reads():
    cfg = OmegaConf.load("configs/models/ug_mtm.yaml")
    required = {
        "num_experts", "latent_dim", "hidden_dim", "action_dim",
        "mc_dropout_T", "gate_lambda", "threshold_window",
        "dropout_rate", "beta_kl",
    }
    assert required <= set(cfg.model)


def test_ug_mtm_config_has_no_keys_the_model_ignores():
    """An unread config key reads as a knob that does something. It doesn't."""
    from omegaconf import OmegaConf as _OC

    cfg = _OC.load("configs/models/ug_mtm.yaml")
    known = {
        "num_experts", "latent_dim", "hidden_dim", "action_dim",
        "mc_dropout_T", "mc_dropout_T_eval", "gate_lambda", "threshold_window",
        "dropout_rate", "beta_kl",
    }
    assert set(cfg.model) <= known, f"unread keys: {set(cfg.model) - known}"


def test_ug_mtm_config_actually_constructs_the_model():
    cfg = OmegaConf.load("configs/models/ug_mtm.yaml")
    # Shrink only the expensive dimensions; keep the config's own semantics.
    cfg.model.hidden_dim = 16
    cfg.model.mc_dropout_T = 2
    model = UG_MTM(cfg)
    assert len(model.expert_pool.experts) == cfg.model.num_experts


def test_ug_mtm_uses_more_than_one_mc_sample():
    """T = 1 collapses the MC-dropout variance to exactly zero."""
    cfg = OmegaConf.load("configs/models/ug_mtm.yaml")
    assert cfg.model.mc_dropout_T > 1


def test_ug_mtm_dropout_is_enabled():
    """Zero dropout would make the uncertainty signal identically zero."""
    cfg = OmegaConf.load("configs/models/ug_mtm.yaml")
    assert cfg.model.dropout_rate > 0.0


def test_rssm_and_ug_mtm_share_capacity_settings():
    """Baseline and method must be compared at matched latent/hidden size."""
    rssm = OmegaConf.load("configs/models/rssm_baseline.yaml")
    ug = OmegaConf.load("configs/models/ug_mtm.yaml")
    assert rssm.model.latent_dim == ug.model.latent_dim
    assert rssm.model.hidden_dim == ug.model.hidden_dim
