"""
Tests for experiments/export_tables.py.

The point of this script is that the paper's tables and the console summary
cannot disagree, so what is pinned here is that the LaTeX carries the same
marks and the same aggregation the summary uses — the peak level, the control
dagger, the skew asterisk — and that nothing in it is hand-written.
"""
import numpy as np
import pytest

from experiments.export_tables import (
    HEADER,
    axis_table,
    encoder_table,
    fmt,
    method_table,
    predictor_table,
    protocol_table,
    task_label,
    tasks_table,
)

PROTOCOL = {"n_train": 5000, "n_collect": 20, "batch_size": 8, "seq_len": 5,
            "seeds": [0, 1, 2, 3, 4]}


def run(method, family, distance, seed, **extra):
    entry = {"method": method, "family": family, "distance": distance,
             "seed": seed, "protocol": PROTOCOL}
    entry.update(extra)
    return entry


def grid(values_by_level, method="finetuning", family="minigrid", key="rd"):
    """One method, three levels, one seed each."""
    return [run(method, family, f"distance_{level}", 0, **{key: value})
            for level, value in values_by_level.items()]


LEVELS = ["distance_min", "distance_med", "distance_max"]


def test_every_table_says_it_is_generated():
    """A number that gets edited in place stops matching the runs it claims to
    describe, which is the failure this whole script exists to prevent."""
    runs = grid({"min": 10.0, "med": 30.0, "max": 20.0})
    for table in (axis_table(runs, ["minigrid"], LEVELS, ["finetuning"]),
                  method_table(runs, ["finetuning"], ["minigrid"], LEVELS,
                               "rd")):
        assert table.startswith(HEADER)
        assert "do not edit" in table


def test_fmt_renders_missing_data_as_a_dash_not_a_zero():
    assert fmt(float("nan")) == "--"
    assert fmt(None) == "--"
    assert fmt(1.234) == "1.23"


def test_axis_table_bolds_and_names_the_peak_level():
    """The claim of the section is where the peak is, so the table has to make
    it visible without the reader scanning three numbers."""
    runs = grid({"min": 10.0, "med": 30.0, "max": 20.0})
    table = axis_table(runs, ["minigrid"], LEVELS, ["finetuning"])
    assert r"\textbf{30.00}" in table
    assert table.rstrip().splitlines()[-3].endswith(r"med \\")


def test_predictor_table_carries_the_rank_correlations():
    runs = []
    for level, rd, dt in [("min", 10.0, 1.0), ("med", 30.0, 3.0),
                          ("max", 20.0, 2.0)]:
        runs.append(run("finetuning", "minigrid", f"distance_{level}", 0,
                        rd=rd, d_trans=dt,
                        heldout_reconstruction_B_after_task_B=rd))
    table = predictor_table(runs, ["minigrid"], LEVELS, ["finetuning"])
    assert "Spearman" in table
    # d_trans and RD are in the same order here, so rho is 1; the labelled
    # level is min < med < max while RD peaks in the middle, so it is not.
    assert "1.00" in table
    assert "0.50" in table


def test_encoder_table_reports_a_factor_so_the_scales_fit_in_one_column():
    """745 and 1.00 have to sit in the same column; +74434.6% and +0.0% do
    not."""
    runs = [run("finetuning", "minigrid", "distance_min", 0,
                heldout_reconstruction_A_after_task_A=1.0,
                heldout_reconstruction_A_after_task_B=745.0)]
    table = encoder_table(runs, ["finetuning"], ["minigrid"], ["distance_min"])
    assert "745" in table
    assert "74434" not in table


def test_encoder_table_daggers_the_control_cells():
    runs = [run("finetuning", "gymnasium", "distance_min", 0,
                heldout_reconstruction_A_after_task_A=26.5,
                heldout_reconstruction_A_after_task_B=24.9)]
    table = encoder_table(runs, ["finetuning"], ["gymnasium"],
                          ["distance_min"])
    assert r"min$^{\dagger}$" in table


def test_method_table_marks_a_skewed_cell():
    """Same mark the console summary uses (D15): the median is reported, and
    the reader is told when the sample behind it has a tail."""
    runs = [run("ug_mtm", "minigrid", "distance_max", seed, rd=value)
            for seed, value in enumerate([17.7, 40.0, 520.4, 574.5, 4364.0])]
    table = method_table(runs, ["ug_mtm"], ["minigrid"], ["distance_max"], "rd")
    assert r"520$^{\ast}$" in table


def test_method_table_leaves_an_absent_cell_empty():
    runs = [run("finetuning", "minigrid", "distance_min", 0, rd=10.0)]
    table = method_table(runs, ["finetuning", "ewc"], ["minigrid"],
                         ["distance_min"], "rd")
    assert "--" in table


def test_protocol_table_reports_the_budget_the_runs_carry():
    """Table 1 comes from the results, not from the config: the previous
    version of this paper described a budget it had never executed."""
    table = protocol_table([run("finetuning", "minigrid", "distance_min", 0,
                                rd=10.0)])
    assert "5000" in table
    assert "Seeds & 5 (0, 1, 2, 3, 4)" in table


def test_protocol_table_refuses_to_describe_two_budgets_at_once():
    other = dict(PROTOCOL, n_train=1000)
    runs = [run("finetuning", "minigrid", "distance_min", 0, rd=10.0),
            run("finetuning", "minigrid", "distance_med", 0, rd=10.0)]
    runs[1]["protocol"] = other
    with pytest.raises(ValueError, match="different protocols"):
        protocol_table(runs)


def test_task_label_drops_a_scale_of_one_and_keeps_gravity():
    assert task_label({"env_id": "HalfCheetah-v4",
                       "params": {"gravity": 4.0, "mass_scale": 1.0}}) == (
        r"\texttt{HalfCheetah-v4} ($g=4$)")
    assert task_label({"domain_name": "cheetah", "task_name": "run",
                       "params": {}}) == r"\texttt{cheetah/run}"


def test_d_param_is_left_empty_where_differencing_physics_would_lie():
    """cheetah -> walker changes everything and leaves the physics vector at
    its default, so Eq. 8 would score the largest change in the family as a
    zero. That gap is the reason d_trans exists."""
    from experiments.export_tables import d_param_of

    swap = {"task_A": {"domain_name": "cheetah", "task_name": "run",
                       "params": {}},
            "task_B": {"domain_name": "walker", "task_name": "run",
                       "params": {}}}
    assert np.isnan(d_param_of(swap))

    physics = {"task_A": {"env_id": "HalfCheetah-v4",
                          "params": {"gravity": 9.8, "mass_scale": 1.0,
                                     "friction_scale": 1.0}},
               "task_B": {"env_id": "HalfCheetah-v4",
                          "params": {"gravity": 4.0, "mass_scale": 1.0,
                                     "friction_scale": 1.0}}}
    assert d_param_of(physics) == pytest.approx(0.5858, abs=1e-4)


def test_tasks_table_refuses_a_cell_that_mixes_environments():
    """The failure this catches is F26: a level was edited, and half a cell's
    seeds came from the environment pair it used to name."""
    pair = {"task_A": {"domain_name": "cheetah", "task_name": "run"},
            "task_B": {"domain_name": "walker", "task_name": "run"}}
    stale = {"task_A": pair["task_A"],
             "task_B": {"domain_name": "walker", "task_name": "stand"}}
    runs = [run("finetuning", "dmcontrol", "distance_max", 0, tasks=pair),
            run("finetuning", "dmcontrol", "distance_max", 1, tasks=stale)]
    with pytest.raises(ValueError, match="different task pairs"):
        tasks_table(runs, ["dmcontrol"], ["distance_max"])
