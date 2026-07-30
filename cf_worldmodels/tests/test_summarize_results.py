"""
Tests for experiments/summarize_results.py.

The aggregation is what every table in the paper is built from, so the parts
that could silently produce a wrong number — pairing seeds, detecting mixed
protocols, the exact permutation test — are pinned down here.
"""
import json

import numpy as np
import pytest

from experiments.summarize_results import (
    DEFAULT_METRICS,
    cell_values,
    rd_share,
    effect_size,
    load_runs,
    paired_differences,
    permutation_p,
    shared_protocol,
)

PROTOCOL = {"n_train": 1000, "n_collect": 20, "batch_size": 8, "seq_len": 5,
            "seeds": [0, 1, 2, 3, 4]}


def make_run(method, seed, wmf, family="minigrid", distance="distance_med",
             protocol=None, **extra):
    run = {"method": method, "family": family, "distance": distance,
           "seed": seed, "wmf": wmf, "protocol": protocol or PROTOCOL}
    run.update(extra)
    return run


def write_runs(tmp_path, runs):
    for run in runs:
        d = (tmp_path / run["method"] /
             f"{run['family']}_{run['distance']}_{run['seed']}")
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps(run))
    return tmp_path


# --- loading ---------------------------------------------------------------

def test_loads_every_run_directory(tmp_path):
    write_runs(tmp_path, [make_run("finetuning", s, float(s)) for s in range(3)])
    runs = load_runs(tmp_path)
    assert len(runs) == 3
    assert {r["seed"] for r in runs} == {0, 1, 2}


def test_empty_directory_loads_nothing(tmp_path):
    assert load_runs(tmp_path) == []


# --- protocol consistency --------------------------------------------------

def test_shared_protocol_returned_when_all_agree():
    runs = [make_run("finetuning", s, 1.0) for s in range(3)]
    assert shared_protocol(runs) == PROTOCOL


def test_mixed_protocols_are_detected():
    """Averaging two training budgets into one cell is the failure mode."""
    other = dict(PROTOCOL, n_train=5000)
    runs = [make_run("finetuning", 0, 1.0),
            make_run("finetuning", 1, 2.0, protocol=other)]
    assert shared_protocol(runs) is None


def test_protocol_comparison_ignores_key_order():
    reordered = {k: PROTOCOL[k] for k in reversed(list(PROTOCOL))}
    runs = [make_run("finetuning", 0, 1.0),
            make_run("finetuning", 1, 2.0, protocol=reordered)]
    assert shared_protocol(runs) == PROTOCOL


# --- cell selection --------------------------------------------------------

def test_cell_values_selects_one_cell_only():
    runs = [make_run("finetuning", 0, 1.0),
            make_run("finetuning", 1, 2.0),
            make_run("ewc", 0, 9.0),
            make_run("finetuning", 0, 5.0, distance="distance_max")]
    assert cell_values(runs, "finetuning", "minigrid", "distance_med",
                       "wmf") == {0: 1.0, 1: 2.0}


def test_missing_key_is_skipped_not_defaulted():
    runs = [make_run("finetuning", 0, 1.0), {"method": "finetuning",
                                             "family": "minigrid",
                                             "distance": "distance_med",
                                             "seed": 1}]
    assert cell_values(runs, "finetuning", "minigrid", "distance_med",
                       "wmf") == {0: 1.0}


def test_null_values_are_skipped_like_missing_ones():
    """ft and d_trans are null in runs made with --skip-reference. Averaging a
    null in as zero would invent a measurement that was never taken."""
    runs = [make_run("finetuning", 0, 1.0, ft=3.0),
            make_run("finetuning", 1, 1.0, ft=None)]
    assert cell_values(runs, "finetuning", "minigrid", "distance_med",
                       "ft") == {0: 3.0}


# --- the aggregate, and how much of it is RD (P7) --------------------------

def test_default_tables_lead_with_the_components_not_the_aggregate():
    """PF and RD are the reported metrics; WMF gets its own labelled section."""
    assert DEFAULT_METRICS[:2] == ["pf", "rd"]
    assert "wmf" not in DEFAULT_METRICS


def test_rd_share_is_one_when_pf_is_zero():
    runs = [make_run("finetuning", 0, 1.0, pf=0.0, rd=20.0)]
    assert rd_share(runs, "finetuning", "minigrid",
                    "distance_med") == pytest.approx(1.0)


def test_rd_share_is_half_when_the_two_contribute_equally():
    runs = [make_run("finetuning", 0, 1.0, pf=5.0, rd=5.0)]
    assert rd_share(runs, "finetuning", "minigrid",
                    "distance_med") == pytest.approx(0.5)


def test_rd_share_uses_magnitudes_so_a_negative_pf_cannot_inflate_it():
    """PF goes negative on fine-tuning (F18); signed sums would cancel and
    report a share above 1."""
    runs = [make_run("finetuning", 0, 1.0, pf=-5.0, rd=5.0)]
    assert rd_share(runs, "finetuning", "minigrid",
                    "distance_med") == pytest.approx(0.5)


def test_rd_share_is_nan_without_data():
    assert np.isnan(rd_share([], "finetuning", "minigrid", "distance_med"))


# --- pairing ---------------------------------------------------------------

def test_pairing_aligns_by_seed_not_by_order():
    runs = [make_run("a", 0, 1.0), make_run("a", 1, 2.0),
            make_run("b", 1, 20.0), make_run("b", 0, 10.0)]
    seeds, a, b = paired_differences(runs, "a", "b", "minigrid",
                                     "distance_med", "wmf")
    assert seeds == [0, 1]
    assert list(a) == [1.0, 2.0]
    assert list(b) == [10.0, 20.0]


def test_unpaired_seeds_are_dropped():
    """A 5-vs-3 comparison dressed up as paired would be worse than a small one."""
    runs = [make_run("a", 0, 1.0), make_run("a", 1, 2.0), make_run("a", 2, 3.0),
            make_run("b", 0, 10.0), make_run("b", 2, 30.0)]
    seeds, a, b = paired_differences(runs, "a", "b", "minigrid",
                                     "distance_med", "wmf")
    assert seeds == [0, 2]
    assert len(a) == len(b) == 2


# --- statistics ------------------------------------------------------------

def test_permutation_p_is_at_its_floor_for_a_unanimous_effect():
    """Every difference the same sign gives the smallest p n allows: 2/2^n."""
    diffs = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    assert permutation_p(diffs) == pytest.approx(2 / 32)


def test_permutation_p_cannot_beat_005_at_n_5():
    """The reason the paper's n=5, p<0.001 claim deserves a second look."""
    diffs = np.array([1e6, 1e6, 1e6, 1e6, 1e6])
    assert permutation_p(diffs) > 0.05


def test_permutation_p_is_one_for_a_symmetric_split():
    diffs = np.array([1.0, -1.0])
    assert permutation_p(diffs) == pytest.approx(1.0)


def test_permutation_p_is_two_sided():
    """Sign of the effect must not change the p-value."""
    diffs = np.array([2.0, 3.0, 4.0])
    assert permutation_p(diffs) == permutation_p(-diffs)


def test_permutation_p_of_no_samples_is_nan():
    assert np.isnan(permutation_p(np.array([])))


def test_effect_size_is_mean_over_sd():
    diffs = np.array([1.0, 2.0, 3.0])
    expected = np.mean(diffs) / np.std(diffs, ddof=1)
    assert effect_size(diffs) == pytest.approx(expected)


def test_effect_size_is_infinite_when_every_difference_is_equal():
    """Zero variance is a real outcome here, not a division to crash on."""
    assert effect_size(np.array([3.0, 3.0, 3.0])) == float("inf")


def test_effect_size_of_a_single_sample_is_nan():
    assert np.isnan(effect_size(np.array([1.0])))
