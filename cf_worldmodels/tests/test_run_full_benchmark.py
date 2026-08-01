"""
Tests for the benchmark runner's protocol handling (I1).

The runner used to define the training protocol in module-level constants while
the configs declared different values and the paper a third set. These tests pin
down the fix: the config is the single source of truth, the runner reads every
field it needs from it, and nothing silently defaults.
"""
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from experiments.run_full_benchmark import (
    FAMILY_CONFIGS,
    MODEL_FIELDS,
    PROTOCOL_FIELDS,
    check_protocol_consistency,
    collect_cell_buffers,
    create_model,
    family_subprocess_argv,
    preflight_action_dims,
    load_reference,
    parse_args,
    protocol_overrides,
    reference_path,
    resolve_protocol,
)

import experiments.run_full_benchmark as runner
from src.utils.seeding import set_seed

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


# --- the shared buffers ----------------------------------------------------

class StubEnv:
    """Minimal BaseEnv-shaped environment with an RNG of its own.

    Real environments keep their randomness in a generator the global seed
    never reaches -- that is what F16 was about -- so a stub that reads the
    global stream would test the opposite of what matters here.
    """

    EPISODE_LENGTH = 4

    def __init__(self, action_dim=2):  # noqa: D401 - see class docstring
        self._rng = np.random.default_rng()
        self._t = 0
        self.action_dim = action_dim
        self.obs_shape = (64, 64, 3)

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)

    def reset(self):
        self._t = 0
        return self._rng.random((64, 64, 3), dtype=np.float32)

    def step(self, action):
        self._t += 1
        done = self._t >= self.EPISODE_LENGTH
        return self._rng.random((64, 64, 3), dtype=np.float32), 0.0, done, {}

    def sample_action(self):
        return self._rng.random(self.action_dim, dtype=np.float32)

    def close(self):
        pass


@pytest.fixture
def tiny_protocol(cfg):
    return resolve_protocol(cfg, {"n_collect": 2, "seq_len": 2})


def _episodes(buffers):
    return [[step["obs"] for step in ep] for buf in buffers for ep in buf.episodes]


def test_the_same_seed_collects_the_same_episodes(tiny_protocol):
    """The five methods and the reference share one collection because a second
    one would reproduce it exactly. If that stops being true, they stop being
    comparable and the sharing becomes a bug rather than a saving."""
    env_A, env_B = StubEnv(), StubEnv()
    first = _episodes(collect_cell_buffers(env_A, env_B, 7, tiny_protocol))
    second = _episodes(collect_cell_buffers(env_A, env_B, 7, tiny_protocol))

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.array_equal(np.array(a), np.array(b))


def test_a_different_seed_collects_different_episodes(tiny_protocol):
    env_A, env_B = StubEnv(), StubEnv()
    first = _episodes(collect_cell_buffers(env_A, env_B, 7, tiny_protocol))
    other = _episodes(collect_cell_buffers(env_A, env_B, 8, tiny_protocol))
    assert not np.array_equal(np.array(first[0]), np.array(other[0]))


def test_held_out_episodes_are_not_the_training_ones(tiny_protocol):
    """Task-A quality is only evidence if the frames were never trained on."""
    env_A, env_B = StubEnv(), StubEnv()
    buf_A, _buf_B, buf_A_heldout, _buf_B_heldout = collect_cell_buffers(
        env_A, env_B, 7, tiny_protocol)
    trained = {ep[0]["obs"].tobytes() for ep in buf_A.episodes}
    heldout = {ep[0]["obs"].tobytes() for ep in buf_A_heldout.episodes}
    assert not (trained & heldout)


def test_collection_leaves_the_global_streams_alone(tiny_protocol):
    """Hoisting the collection out of run_cell only preserves results because
    it draws nothing from the streams that initialise and train the model."""
    env_A, env_B = StubEnv(), StubEnv()
    collect_cell_buffers(env_A, env_B, 7, tiny_protocol)
    after_collection = (torch.randn(4).tolist(), np.random.random(4).tolist())

    set_seed(7)
    direct = (torch.randn(4).tolist(), np.random.random(4).tolist())
    assert after_collection == direct


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


# --- action widths, checked before training (F25) ---------------------------

def test_a_pair_whose_tasks_disagree_is_rejected(monkeypatch, cfg):
    """F25: dmcontrol's distance_max paired cheetah (6 actions) with reacher
    (2). One world model spans both tasks, so its GRU has one action width;
    the mismatch surfaced 12 hours in as a torch shape error."""
    monkeypatch.setattr(runner, "create_env_pair",
                        lambda family, seq: (StubEnv(7), StubEnv(2)))
    with pytest.raises(SystemExit) as excinfo:
        preflight_action_dims("minigrid", cfg, ["distance_max"])
    assert "action_dim=2" in str(excinfo.value)
    assert "task_B" in str(excinfo.value)


def test_every_broken_pair_is_reported_at_once(monkeypatch, cfg):
    """Two broken levels should cost one fix, not two runs."""
    monkeypatch.setattr(runner, "create_env_pair",
                        lambda family, seq: (StubEnv(7), StubEnv(2)))
    with pytest.raises(SystemExit) as excinfo:
        preflight_action_dims("minigrid", cfg,
                              ["distance_min", "distance_max"])
    assert "distance_min" in str(excinfo.value)
    assert "distance_max" in str(excinfo.value)


def test_matching_pairs_pass_and_close_their_environments(monkeypatch, cfg):
    closed = []

    class ClosingStub(StubEnv):
        def close(self):
            closed.append(self)

    monkeypatch.setattr(runner, "create_env_pair",
                        lambda family, seq: (ClosingStub(7), ClosingStub(7)))
    preflight_action_dims("minigrid", cfg, ["distance_min"])
    assert len(closed) == 2


# --- one process per family (F24) ------------------------------------------

def test_each_family_gets_its_own_process(monkeypatch, tmp_path):
    """dm_control cannot build an OpenGL context in a process where MuJoCo
    already has one, so a single process cannot run the whole grid."""
    calls = []

    class Result:
        returncode = 0

    monkeypatch.setattr(runner.subprocess, "run",
                        lambda argv, **kw: calls.append(argv) or Result())
    monkeypatch.setattr(runner, "print_results_table", lambda *a: None)
    runner.main(["--results-dir", str(tmp_path)])

    assert [argv[argv.index("--families") + 1] for argv in calls] ==         list(FAMILY_CONFIGS)


def test_a_failing_family_stops_the_run(monkeypatch, tmp_path):
    class Result:
        returncode = 1

    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kw: Result())
    with pytest.raises(SystemExit, match="subprocess exited"):
        runner.main(["--results-dir", str(tmp_path)])


def test_one_family_runs_in_this_process(monkeypatch, tmp_path):
    """No subprocess when there is nothing to isolate -- and the child call,
    which asks for exactly one family, must not recurse forever."""
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **kw: pytest.fail("should not spawn"))
    runner.main(["--families", "minigrid", "--dry-run",
                 "--results-dir", str(tmp_path)])


def test_dry_run_stays_in_one_process(monkeypatch, tmp_path):
    """--dry-run only prints; spawning three interpreters to print would make
    the plan harder to read, not easier."""
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *a, **kw: pytest.fail("should not spawn"))
    runner.main(["--dry-run", "--results-dir", str(tmp_path)])


def test_every_flag_reaches_the_subprocess():
    """F9 was overrides that were built, printed, and never passed on. The
    generated command line is parsed back and compared field by field."""
    argv = ["--methods", "ewc", "ug_mtm", "--distances", "distance_max",
            "--results-dir", "somewhere", "--skip-reference",
            "--steps", "7", "--batch-size", "3", "--seq-len", "2",
            "--n-collect", "4", "--seeds", "11", "12"]
    parent = parse_args(argv)
    child = parse_args(family_subprocess_argv(parent, "dmcontrol")[2:])

    assert child.families == ["dmcontrol"]
    assert child.methods == parent.methods
    assert child.distances == parent.distances
    assert str(child.results_dir) == str(parent.results_dir)
    assert child.skip_reference == parent.skip_reference
    assert protocol_overrides(child) == protocol_overrides(parent)


def test_defaults_survive_the_round_trip():
    """A None override must stay None, not become an explicit value."""
    parent = parse_args([])
    child = parse_args(family_subprocess_argv(parent, "minigrid")[2:])
    assert protocol_overrides(child) == protocol_overrides(parent)
    assert all(v is None for v in protocol_overrides(child).values())


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
