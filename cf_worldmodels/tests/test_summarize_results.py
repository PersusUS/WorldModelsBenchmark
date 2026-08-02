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
    AXIS_METHODS,
    DEFAULT_METRICS,
    axis_value,
    cell_summary,
    cell_values,
    control_cells,
    rd_share,
    effect_size,
    is_right_skewed,
    load_runs,
    paired_differences,
    permutation_p,
    rank,
    shared_protocol,
    spearman,
    task_a_loss,
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


def test_the_reported_suite_is_pf_rd_and_ft():
    """PIS was announced and never implemented, and is withdrawn rather than
    reported as a zero (D18/F6). A table of it would be a column of nulls
    presenting an unimplemented metric as a measured one."""
    assert DEFAULT_METRICS == ["pf", "rd", "ft"]
    assert "pis" not in DEFAULT_METRICS


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


# --- how a cell is summarised (D15/P12) ------------------------------------

def test_a_symmetric_cell_is_not_flagged_as_skewed():
    """EWC's PF straddles zero with a small spread. A criterion built on the
    mean/median gap or on a coefficient of variation would fire here."""
    assert not is_right_skewed([-0.12, -0.03, -0.005, 0.04, 0.086])


def test_one_seed_an_order_of_magnitude_out_is_flagged():
    """The cell P12 exists for: ug_mtm on minigrid/distance_max (F23)."""
    assert is_right_skewed([17.7, 40.0, 520.0, 574.0, 4364.0])


def test_skew_is_one_sided():
    """A long left tail is not what the median policy is guarding against, and
    calling it the same thing would make the marker mean two things."""
    assert not is_right_skewed([-4364.0, -574.0, -520.0, -40.0, -17.7])


def test_identical_values_are_not_skewed():
    """ug_mtm freezes its encoder, so its pixel columns repeat one value."""
    assert not is_right_skewed([7.66, 7.66, 7.66, 7.66, 7.66])


def test_two_samples_are_never_called_skewed():
    assert not is_right_skewed([1.0, 1000.0])


def test_cell_summary_reports_median_and_range_not_mean():
    summary = cell_summary([17.7, 40.0, 520.0, 574.0, 4364.0])
    assert summary.startswith("+520")
    assert "17.7" in summary and "4364" in summary
    assert "1103" not in summary  # the mean, which describes none of the five


def test_cell_summary_marks_a_skewed_cell():
    assert cell_summary([17.7, 40.0, 520.0, 574.0, 4364.0]).endswith("!")
    assert not cell_summary([1.0, 2.0, 3.0, 4.0, 5.0]).endswith("!")


def test_cell_summary_of_no_values():
    assert cell_summary([]) == "N/A"


def test_the_skew_marker_can_be_turned_off_for_bounded_columns():
    """Pixel reconstruction is bounded by the data; the marker is about metrics
    that are unbounded above, and firing it everywhere would make it mean
    nothing. The range is still printed."""
    values = [17.7, 40.0, 520.0, 574.0, 4364.0]
    assert "!" not in cell_summary(values, flag_skew=False)
    assert "17.7" in cell_summary(values, flag_skew=False)


# --- the distance axis (F27) -----------------------------------------------

def test_axis_leaves_out_the_method_with_the_frozen_encoder():
    """The exclusion is the one editorial choice in F27, so it is pinned: it
    must be declared in a constant, not applied wherever the section is drawn."""
    assert "ug_mtm" not in AXIS_METHODS
    assert set(AXIS_METHODS) == {"finetuning", "replay_infinite", "ewc",
                                 "progressive_nets"}


def test_axis_value_takes_the_median_over_methods_of_median_over_seeds():
    runs = [make_run("finetuning", 0, 0.0, rd=10.0),
            make_run("finetuning", 1, 0.0, rd=90.0),   # method median 50
            make_run("ewc", 0, 0.0, rd=20.0),
            make_run("ewc", 1, 0.0, rd=20.0),          # method median 20
            make_run("replay_infinite", 0, 0.0, rd=30.0)]  # method median 30
    value = axis_value(runs, ["finetuning", "ewc", "replay_infinite"],
                       "minigrid", "distance_med", "rd")
    assert value == pytest.approx(30.0)


def test_axis_value_is_not_dragged_by_one_extreme_method():
    """A mean over methods would let the outlier method set the family's
    number, which is the same failure P12 fixes one level down."""
    runs = [make_run("finetuning", 0, 0.0, rd=10.0),
            make_run("ewc", 0, 0.0, rd=12.0),
            make_run("replay_infinite", 0, 0.0, rd=4000.0)]
    value = axis_value(runs, ["finetuning", "ewc", "replay_infinite"],
                       "minigrid", "distance_med", "rd")
    assert value == pytest.approx(12.0)


def test_axis_value_of_an_absent_cell_is_nan():
    assert np.isnan(axis_value([], AXIS_METHODS, "minigrid", "distance_med",
                               "rd"))


# --- rank correlation, written out because scipy is undeclared (F12) -------

def test_rank_averages_ties():
    assert list(rank([10.0, 20.0, 20.0, 40.0])) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_one_for_any_monotone_relation():
    """Rank correlation, not Pearson: the relation is monotone but not linear."""
    assert spearman([1, 2, 3, 4], [1, 10, 100, 1000]) == pytest.approx(1.0)


def test_spearman_is_minus_one_when_reversed():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_matches_the_hand_computed_value():
    rho = 1 - 6 * (1 + 1) / (4 * (4 ** 2 - 1))  # two adjacent ranks swapped
    assert spearman([1, 2, 3, 4], [2, 1, 3, 4]) == pytest.approx(rho)


def test_spearman_of_a_constant_predictor_is_nan():
    """Not 0: a predictor with no variation has no rank correlation to report,
    and printing +0.00 would read as 'measured, and it does not predict'."""
    assert np.isnan(spearman([1, 2, 3], [5, 5, 5]))


def test_spearman_rejects_mismatched_series():
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


# --- control cells (F22/P13) -----------------------------------------------

def quality_run(method, seed, before, after, family="gymnasium",
                distance="distance_min", **extra):
    return make_run(method, seed, 0.0, family=family, distance=distance,
                    heldout_reconstruction_A_after_task_A=before,
                    heldout_reconstruction_A_after_task_B=after, **extra)


def test_task_a_loss_is_relative_to_what_the_model_had():
    """+0.23 is 1.6% of DMControl's task-A error and 44% of MiniGrid's. An
    absolute threshold would call the same number forgetting in one family and
    noise in the other."""
    runs = [quality_run("finetuning", 0, 14.74, 14.97)]
    assert task_a_loss(runs, "finetuning", "gymnasium",
                       "distance_min") == pytest.approx(0.0156, abs=1e-3)


def test_task_a_loss_is_negative_when_task_b_helped():
    runs = [quality_run("finetuning", 0, 26.49, 24.0)]
    assert task_a_loss(runs, "finetuning", "gymnasium", "distance_min") < 0


def test_a_cell_is_a_control_only_if_no_method_lost_anything():
    runs = [quality_run("finetuning", 0, 10.0, 9.0),
            quality_run("ewc", 0, 10.0, 10.05),
            quality_run("progressive_nets", 0, 10.0, 400.0)]
    assert control_cells(runs, ["finetuning", "ewc", "progressive_nets"],
                         ["gymnasium"], ["distance_min"]) == []
    assert control_cells(runs, ["finetuning", "ewc"],
                         ["gymnasium"], ["distance_min"]) == [("gymnasium",
                                                               "distance_min")]


def test_a_control_cell_is_about_pixels_and_says_nothing_about_rd():
    """gymnasium/distance_med loses nothing in pixels and still has the
    family's highest RD. The two scales disagreeing is F18, so the control
    criterion must not consult RD and quietly rule the cell out."""
    runs = [quality_run("finetuning", 0, 26.15, 26.2, distance="distance_med",
                        rd=113.0)]
    assert control_cells(runs, ["finetuning"], ["gymnasium"],
                         ["distance_med"]) == [("gymnasium", "distance_med")]


def test_control_threshold_is_a_parameter_not_a_constant():
    runs = [quality_run("finetuning", 0, 10.0, 10.5)]  # +5%
    assert control_cells(runs, ["finetuning"], ["gymnasium"],
                         ["distance_min"], threshold=0.10)
    assert not control_cells(runs, ["finetuning"], ["gymnasium"],
                             ["distance_min"], threshold=0.01)
