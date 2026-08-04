"""
Tests for experiments/run_sequence.py.

The sequence runner exists to answer the one question the paired grid cannot,
so what is pinned here is the shape of that answer: that every earlier task is
scored at every later stage, that each task is compared against the model as it
stood when that task finished rather than against some other snapshot, and that
each method is handed its task boundary the same way the paired runner hands it
one.
"""
import numpy as np
import pytest
import torch

from experiments.run_sequence import (
    advance_method,
    make_env,
    result_path,
    run_sequence,
    training_buffer,
)
from src.utils.buffer import ReplayBuffer

from conftest import ACTION_DIM, HIDDEN_DIM, LATENT_DIM

PROTOCOL = {
    "n_collect": 2, "n_train": 2, "batch_size": 2, "seq_len": 2,
    "learning_rate": 1e-3, "curve_points": 0, "n_eval_episodes": 1,
    "n_eval_transitions": 4, "n_fisher_transitions": 2, "n_recon_frames": 2,
    "rd_horizon": 2, "rd_samples": 2, "ewc_lambda": 1000.0,
    "mc_dropout_T_train": 2, "seeds": [0],
    "latent_dim": LATENT_DIM, "hidden_dim": HIDDEN_DIM, "beta_kl": 1.0,
}


def make_episode(length=6):
    return [
        {
            "obs": np.random.rand(64, 64, 3).astype(np.float32),
            "action": np.random.rand(ACTION_DIM).astype(np.float32),
            "reward": 0.0,
            "done": i == length - 1,
        }
        for i in range(length)
    ]


def make_buffer(n_episodes=2):
    buf = ReplayBuffer(max_episodes=n_episodes, seq_len=PROTOCOL["seq_len"])
    for _ in range(n_episodes):
        buf.add_episode(make_episode())
    return buf


# --- what each method is handed at a boundary ------------------------------

def test_replay_trains_on_every_task_seen_so_far():
    """At stage i replay's buffer is the union of tasks 0..i; everyone else
    gets the current task alone. This is the difference that makes replay's
    forward transfer confounded with a divided budget, so it has to be right."""
    buffers = [make_buffer(2) for _ in range(3)]
    alone = training_buffer("finetuning", buffers, 2, PROTOCOL)
    assert len(alone.episodes) == 2

    combined = training_buffer("replay_infinite", buffers, 2, PROTOCOL)
    assert len(combined.episodes) == 6


def test_replay_is_told_about_every_earlier_task_not_just_the_last():
    """add_task_data is called once per task index up to the one just
    finished. Passing only the newest would quietly make an unbounded buffer
    behave like a buffer of size one."""
    class Recorder:
        def __init__(self):
            self.seen = []

        def add_task_data(self, task_id, episodes):
            self.seen.append(task_id)

    model = Recorder()
    buffers = [make_buffer(1) for _ in range(4)]
    advance_method(model, "replay_infinite", None, buffers, buffers, None, 0,
                   PROTOCOL, finished=2)
    assert model.seen == [0, 1, 2]


def test_progressive_nets_gets_one_column_per_boundary():
    class Recorder:
        def __init__(self):
            self.columns = 0

        def add_column(self):
            self.columns += 1

    model = Recorder()
    for finished in range(3):
        advance_method(model, "progressive_nets", None, [], [], None, 0,
                       PROTOCOL, finished=finished)
    assert model.columns == 3


def test_a_family_without_a_constructor_fails_loudly():
    """Silently running minigrid envs for a dmcontrol request would produce a
    plausible-looking result file for an experiment that never happened."""
    with pytest.raises(ValueError, match="minigrid only"):
        make_env("dmcontrol", object())


def test_result_path_keeps_methods_and_seeds_apart():
    from pathlib import Path
    a = result_path(Path("r"), "minigrid", 3, "ewc")
    b = result_path(Path("r"), "minigrid", 3, "finetuning")
    assert a != b
    assert a.name == "metrics.json"


# --- the forgetting matrix -------------------------------------------------

@pytest.mark.slow
def test_every_earlier_task_is_scored_at_every_later_stage():
    """The point of the whole script: stage i reports on tasks 0..i, so the
    retention matrix is lower-triangular and the last stage covers everything."""
    k = 3
    buffers = [make_buffer(2) for _ in range(k)]
    heldouts = [make_buffer(1) for _ in range(k)]
    tasks = [{"env_id": f"T{i}"} for i in range(k)]

    metrics = run_sequence("finetuning", "minigrid", tasks, buffers, heldouts,
                           ACTION_DIM, torch.device("cpu"), 0, PROTOCOL)

    assert metrics["k"] == k
    assert len(metrics["stages"]) == k
    for i, stage in enumerate(metrics["stages"]):
        assert [r["task"] for r in stage["retention"]] == list(range(i + 1))


@pytest.mark.slow
def test_each_task_is_compared_against_its_own_snapshot():
    """PF(i,i) and RD(i,i) are a model against itself and must be exactly
    zero. A nonzero diagonal would mean the snapshot taken after task i is not
    the one task i is scored against -- the failure that would silently make
    every off-diagonal entry meaningless."""
    k = 3
    buffers = [make_buffer(2) for _ in range(k)]
    heldouts = [make_buffer(1) for _ in range(k)]
    tasks = [{"env_id": f"T{i}"} for i in range(k)]

    metrics = run_sequence("finetuning", "minigrid", tasks, buffers, heldouts,
                           ACTION_DIM, torch.device("cpu"), 0, PROTOCOL)

    for i, stage in enumerate(metrics["stages"]):
        diagonal = [r for r in stage["retention"] if r["task"] == i][0]
        assert diagonal["pf"] == 0.0
        assert diagonal["rd"] == 0.0


@pytest.mark.slow
def test_the_protocol_travels_with_the_result():
    """Same invariant the paired runner holds: a result file says what budget
    produced it, so two of them can never be pooled by accident."""
    buffers = [make_buffer(2) for _ in range(2)]
    heldouts = [make_buffer(1) for _ in range(2)]
    metrics = run_sequence("finetuning", "minigrid",
                           [{"env_id": "A"}, {"env_id": "B"}], buffers,
                           heldouts, ACTION_DIM, torch.device("cpu"), 0,
                           PROTOCOL)
    assert metrics["protocol"] == PROTOCOL
    assert metrics["tasks"] == [{"env_id": "A"}, {"env_id": "B"}]
