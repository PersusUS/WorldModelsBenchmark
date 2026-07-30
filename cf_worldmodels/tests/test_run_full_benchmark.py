"""
Tests for the benchmark runner's protocol handling (I1).

The runner used to define the training protocol in module-level constants while
the configs declared different values and the paper a third set. These tests pin
down the fix: the config is the single source of truth, the runner reads every
field it needs from it, and nothing silently defaults.
"""
import json

import pytest
from omegaconf import OmegaConf

from experiments.run_full_benchmark import (
    FAMILY_CONFIGS,
    MODEL_FIELDS,
    PROTOCOL_FIELDS,
    check_protocol_consistency,
    create_model,
    load_reference,
    parse_args,
    protocol_overrides,
    reference_path,
    resolve_protocol,
)

import experiments.run_full_benchmark as runner

ACTION_DIM = 4


@pytest.fixture
def cfg():
    return OmegaConf.load(FAMILY_CONFIGS["minigrid"])


# --- the constants are gone ------------------------------------------------

@pytest.mark.parametrize("name", ["STEPS", "SEQ_LEN", "BATCH_SIZE", "N_COLLECT",
                                  "SEEDS", "N_EVAL_EPISODES",
                                  "N_EVAL_TRANSITIONS",
                                  "N_FISHER_TRANSITIONS"])
def test_runner_defines_no_protocol_constants(name):
    """Regression guard for I1: a constant here is a value the configs cannot
    reach, and every one of them drifted."""
    assert not hasattr(runner, name), (
        f"{name} is back as a module constant; read it from the config instead"
    )


# --- resolve_protocol ------------------------------------------------------

def test_reads_every_declared_field(cfg):
    protocol = resolve_protocol(cfg)
    for field in PROTOCOL_FIELDS:
        assert field in protocol
    for field in MODEL_FIELDS:
        assert field in protocol
    assert protocol["seeds"] == [0, 1, 2, 3, 4]
    assert set(protocol["wmf_weights"]) == {"alpha", "beta", "gamma"}


def test_result_is_json_serializable(cfg):
    """It is stored verbatim in every metrics.json, so no OmegaConf nodes."""
    round_tripped = json.loads(json.dumps(resolve_protocol(cfg)))
    assert round_tripped == resolve_protocol(cfg)


def test_missing_field_is_an_error_not_a_default(cfg):
    del cfg.protocol.n_train
    with pytest.raises(KeyError, match="n_train"):
        resolve_protocol(cfg)


def test_missing_protocol_block_is_an_error(cfg):
    del cfg.protocol
    with pytest.raises(KeyError, match="protocol"):
        resolve_protocol(cfg)


def test_overrides_replace_config_values(cfg):
    protocol = resolve_protocol(cfg, {"n_train": 7, "seeds": [42]})
    assert protocol["n_train"] == 7
    assert protocol["seeds"] == [42]


def test_none_overrides_are_ignored(cfg):
    """Unset command-line flags arrive as None and must not blank the config."""
    baseline = resolve_protocol(cfg)
    protocol = resolve_protocol(cfg, {"n_train": None, "batch_size": None})
    assert protocol == baseline


def test_unknown_override_is_rejected(cfg):
    with pytest.raises(KeyError, match="nonexistent"):
        resolve_protocol(cfg, {"nonexistent": 1})


@pytest.mark.parametrize("field", ["n_train", "batch_size", "seq_len",
                                   "n_collect", "rd_horizon", "hidden_dim"])
def test_non_positive_values_are_rejected(cfg, field):
    with pytest.raises(ValueError, match="positive"):
        resolve_protocol(cfg, {field: 0})


def test_empty_seed_list_is_rejected(cfg):
    with pytest.raises(ValueError, match="no seeds"):
        resolve_protocol(cfg, {"seeds": []})


def test_wmf_weights_must_sum_to_one(cfg):
    cfg.protocol.wmf_weights.alpha = 0.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        resolve_protocol(cfg)


def test_types_are_coerced(cfg):
    """YAML lets 1e-3 parse as a string; the protocol must still be numeric."""
    cfg.protocol.n_train = "250"
    cfg.protocol.learning_rate = "1e-4"
    protocol = resolve_protocol(cfg)
    assert protocol["n_train"] == 250
    assert protocol["learning_rate"] == pytest.approx(1e-4)


# --- the configs match what the runner reads -------------------------------

@pytest.mark.parametrize("family", list(FAMILY_CONFIGS))
def test_every_family_config_resolves(family):
    resolve_protocol(OmegaConf.load(FAMILY_CONFIGS[family]))


@pytest.mark.parametrize("family", list(FAMILY_CONFIGS))
def test_protocol_block_has_no_keys_the_runner_ignores(family):
    """An unread config key reads as a knob that does something. It doesn't —
    that is how `eval_every: 5000` survived for months."""
    cfg = OmegaConf.load(FAMILY_CONFIGS[family])
    known = set(PROTOCOL_FIELDS) | {"seeds", "wmf_weights"}
    unread = set(cfg.protocol) - known
    assert not unread, f"unread keys in {family}: {sorted(unread)}"


# --- create_model ----------------------------------------------------------

@pytest.mark.parametrize("method", ["finetuning", "replay_infinite", "ewc",
                                    "progressive_nets", "ug_mtm"])
def test_model_capacity_comes_from_the_protocol(cfg, method):
    """Baselines used to be built with hardcoded 32/512 while UG-MTM read its
    own config file. Matched capacity has to hold by construction."""
    protocol = resolve_protocol(cfg, {"n_train": 1})
    protocol["latent_dim"] = 6
    protocol["hidden_dim"] = 12
    model = create_model(method, ACTION_DIM, protocol)
    assert model.latent_dim == 6
    assert model.hidden_dim == 12
    assert model.action_dim == ACTION_DIM


def test_ewc_lambda_comes_from_the_protocol(cfg):
    protocol = resolve_protocol(cfg, None)
    protocol["ewc_lambda"] = 3.5
    assert create_model("ewc", ACTION_DIM, protocol).ewc_lambda == 3.5


def test_ug_mtm_mc_dropout_budget_comes_from_the_protocol(cfg):
    """The runner overrides the model config's training-time T. The override has
    to be visible in the protocol rather than buried in the runner."""
    protocol = resolve_protocol(cfg)
    protocol["latent_dim"] = 6
    protocol["hidden_dim"] = 12
    model = create_model("ug_mtm", ACTION_DIM, protocol)
    assert model.T == protocol["mc_dropout_T_train"]
    # Evaluation-time T is UG-MTM's own decision (D6) and is not overridden.
    assert model.T_eval == OmegaConf.load(
        runner.UG_MTM_CONFIG).model.mc_dropout_T_eval


def test_unknown_method_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown method"):
        create_model("does_not_exist", ACTION_DIM, resolve_protocol(cfg))


# --- cached results cannot mix protocols -----------------------------------

def _write(path, protocol):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"wmf": 1.0, "protocol": protocol}))
    return path


def test_matching_cached_protocol_passes(tmp_path, cfg):
    protocol = resolve_protocol(cfg)
    path = _write(tmp_path / "finetuning" / "minigrid_distance_min_0" /
                  "metrics.json", protocol)
    check_protocol_consistency([path], protocol)


def test_mismatched_cached_protocol_stops_the_run(tmp_path, cfg):
    """Skipping cached cells is what makes the runner resumable; it must not
    also average two training budgets into one table cell."""
    protocol = resolve_protocol(cfg)
    stale = dict(protocol, n_train=protocol["n_train"] * 2)
    path = _write(tmp_path / "finetuning" / "minigrid_distance_min_0" /
                  "metrics.json", stale)
    with pytest.raises(SystemExit, match="different protocol"):
        check_protocol_consistency([path], protocol)


def test_results_without_a_recorded_protocol_stop_the_run(tmp_path, cfg):
    """Pre-instrumentation results cannot be proven comparable, so they are
    treated as a mismatch rather than trusted."""
    path = tmp_path / "finetuning" / "minigrid_distance_min_0" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"wmf": 1.0}))
    with pytest.raises(SystemExit):
        check_protocol_consistency([path], resolve_protocol(cfg))


# --- the from-scratch task-B reference (P8/P10) -----------------------------

def _write_reference(root, protocol, **extra):
    path = reference_path(root, "minigrid", "distance_min", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"d_trans": 1.5, "heldout_reconstruction_B_from_scratch": 9.0,
               "protocol": protocol}
    payload.update(extra)
    path.write_text(json.dumps(payload))
    return path


def test_missing_reference_is_not_an_error(tmp_path, cfg):
    """--skip-reference is allowed; the cell then stores ft and d_trans null."""
    assert load_reference(tmp_path, "minigrid", "distance_min", 0,
                          resolve_protocol(cfg)) is None


def test_reference_is_read_back(tmp_path, cfg):
    protocol = resolve_protocol(cfg)
    _write_reference(tmp_path, protocol)
    stored = load_reference(tmp_path, "minigrid", "distance_min", 0, protocol)
    assert stored["d_trans"] == 1.5


def test_reference_from_another_protocol_stops_the_run(tmp_path, cfg):
    """The from-scratch arm is only a counterfactual at the same budget."""
    protocol = resolve_protocol(cfg)
    _write_reference(tmp_path, dict(protocol, n_train=protocol["n_train"] * 2))
    with pytest.raises(SystemExit, match="different protocol"):
        load_reference(tmp_path, "minigrid", "distance_min", 0, protocol)


def test_reference_is_shared_by_every_method(tmp_path, cfg):
    """One path per (family, distance, seed): d_trans is a property of the task
    pair, not of the continual-learning method."""
    paths = {reference_path(tmp_path, "minigrid", "distance_min", 0)
             for _ in runner.METHODS}
    assert len(paths) == 1
    assert "_reference" in str(paths.pop())


# --- command line ----------------------------------------------------------

def test_no_flags_means_the_config_decides():
    args = parse_args([])
    assert all(v is None for v in protocol_overrides(args).values())


def test_flags_map_onto_protocol_fields(cfg):
    args = parse_args(["--steps", "5", "--batch-size", "2", "--seq-len", "3",
                       "--n-collect", "4", "--seeds", "7", "8"])
    protocol = resolve_protocol(cfg, protocol_overrides(args))
    assert protocol["n_train"] == 5
    assert protocol["batch_size"] == 2
    assert protocol["seq_len"] == 3
    assert protocol["n_collect"] == 4
    assert protocol["seeds"] == [7, 8]


def test_override_keys_are_all_real_protocol_fields():
    """A typo in the flag-to-field mapping would silently do nothing."""
    known = set(PROTOCOL_FIELDS) | set(MODEL_FIELDS) | {"seeds", "wmf_weights"}
    assert set(protocol_overrides(parse_args([]))) <= known


def test_selection_flags_default_to_the_whole_grid():
    args = parse_args([])
    assert args.families == list(FAMILY_CONFIGS)
    assert args.methods == runner.METHODS
    assert args.distances == runner.DISTANCES


def test_the_reference_runs_by_default():
    """Forward transfer and d_trans are part of the benchmark, not an extra."""
    assert parse_args([]).skip_reference is False
    assert parse_args(["--skip-reference"]).skip_reference is True
