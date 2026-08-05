"""
Tests for experiments/summarize_sequence.py.

The sequence table makes one claim -- that retention peaks somewhere other than
the end -- so what is pinned here is that the peak is counted per seed rather
than read off an average, that the structural zeros on the diagonal stay out of
the curve, and that a directory mixing two sequences is refused.
"""
import pytest

from experiments.summarize_sequence import (
    check_sequence_runs,
    encoder_factor,
    peak_stage,
    retention_curve,
    task_difficulty,
)

PROTOCOL = {"n_train": 5000, "seeds": [0, 1, 2, 3, 4]}
TASKS = [{"env_id": "A"}, {"env_id": "B"}, {"env_id": "C"}]


def make_run(method, seed, rd_by_stage, recon_by_stage=None, tasks=None,
             protocol=None):
    """One k=3 run: task 0 measured at stages 0, 1 and 2."""
    recon_by_stage = recon_by_stage or {0: 1.0, 1: 1.0, 2: 1.0}
    stages = []
    for stage in range(3):
        retention = [{
            "task": 0,
            "pf": 0.0 if stage == 0 else 1.0,
            "rd": 0.0 if stage == 0 else rd_by_stage[stage],
            "heldout_reconstruction": recon_by_stage[stage],
        }]
        if stage > 0:  # later tasks, so task_difficulty has something to read
            retention.append({"task": stage, "pf": 0.0, "rd": 0.0,
                              "heldout_reconstruction": 10.0 * stage})
        stages.append({"stage": stage, "task": (tasks or TASKS)[stage],
                       "n_nan_steps": 0, "retention": retention})
    return {"method": method, "seed": seed, "k": 3,
            "tasks": tasks or TASKS, "stages": stages,
            "protocol": protocol or PROTOCOL}


def test_the_diagonal_stays_out_of_the_curve():
    """RD(i, i) is zero by construction. Leaving it in would put a structural
    zero at the start of every curve and invent a rise that is not measured."""
    runs = [make_run("finetuning", 0, {1: 5.0, 2: 3.0})]
    curve = retention_curve(runs, "finetuning", task=0, key="rd")
    assert sorted(curve) == [1, 2]


def test_the_peak_is_counted_per_seed_not_off_the_median():
    """A peak that only exists after averaging is a property of the average.
    Here four of five seeds peak at stage 1 and one peaks at stage 2."""
    runs = [make_run("finetuning", s, {1: 100.0, 2: 10.0}) for s in range(4)]
    runs.append(make_run("finetuning", 4, {1: 10.0, 2: 100.0}))
    peak, agreeing, n_seeds = peak_stage(runs, "finetuning", task=0)
    assert (peak, agreeing, n_seeds) == (1, 4, 5)


def test_a_monotone_curve_peaks_at_the_end():
    runs = [make_run("replay_infinite", s, {1: 5.0, 2: 9.0}) for s in range(3)]
    peak, agreeing, n_seeds = peak_stage(runs, "replay_infinite", task=0)
    assert (peak, agreeing, n_seeds) == (2, 3, 3)


def test_encoder_factor_is_relative_to_the_moment_the_task_finished():
    """Same quantity the paired grid reports: what the model has now over what
    it had when it stopped training on that task, not an absolute error."""
    runs = [make_run("finetuning", 0, {1: 1.0, 2: 1.0},
                     recon_by_stage={0: 2.0, 1: 20.0, 2: 200.0})]
    assert encoder_factor(runs, "finetuning", task=0, stage=2) == [100.0]


def test_task_difficulty_reads_each_task_when_it_finished():
    runs = [make_run("finetuning", s, {1: 1.0, 2: 1.0}) for s in range(3)]
    assert task_difficulty(runs, 1) == [10.0, 10.0, 10.0]


def test_two_sequences_in_one_directory_are_refused():
    """The same failure F26 was in the paired grid: a sequence was edited and
    stale results stayed behind. Every number here medians over the runs."""
    other = [{"env_id": "A"}, {"env_id": "B"}, {"env_id": "Z"}]
    runs = [make_run("finetuning", 0, {1: 1.0, 2: 1.0}),
            make_run("finetuning", 1, {1: 1.0, 2: 1.0}, tasks=other)]
    with pytest.raises(ValueError, match="different task sequences"):
        check_sequence_runs(runs)


def test_two_budgets_in_one_directory_are_refused():
    runs = [make_run("finetuning", 0, {1: 1.0, 2: 1.0}),
            make_run("finetuning", 1, {1: 1.0, 2: 1.0},
                     protocol=dict(PROTOCOL, n_train=1000))]
    with pytest.raises(ValueError, match="different protocols"):
        check_sequence_runs(runs)


def test_a_different_seed_list_is_not_a_different_protocol():
    """Same rule as the paired grid (I22): `seeds` is provenance, not budget."""
    runs = [make_run("finetuning", 0, {1: 1.0, 2: 1.0}),
            make_run("finetuning", 5, {1: 1.0, 2: 1.0},
                     protocol=dict(PROTOCOL, seeds=[5, 6]))]
    check_sequence_runs(runs)  # must not raise
