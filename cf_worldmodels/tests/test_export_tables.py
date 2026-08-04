"""
Tests for experiments/export_tables.py.

The point of this script is that the paper's tables and the console summary
cannot disagree, so what is pinned here is that the LaTeX carries the same
marks and the same aggregation the summary uses — the peak level, the control
dagger, the skew asterisk — and that nothing in it is hand-written.
"""
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
    runs = [run("finetuning", "minigrid", "distance_min", s, rd=10.0)
            for s in range(5)]
    table = protocol_table(runs)
    assert "5000" in table
    assert r"Seeds per cell & 5 \\" in table


def test_protocol_table_counts_seeds_off_the_runs_not_the_protocol_block():
    """After seeds are added to a finished grid the block records what one
    invocation was asked for. Only the cells know what exists, and they need
    not agree -- the six discriminating cells got ten, the controls kept five."""
    runs = [run("finetuning", "minigrid", "distance_min", s, rd=10.0)
            for s in range(10)]
    runs += [run("finetuning", "gymnasium", "distance_min", s, rd=10.0)
             for s in range(5)]
    table = protocol_table(runs)
    assert r"Seeds per cell & 5--10 \\" in table


def test_protocol_labels_cover_every_field_the_runner_records():
    """A field the runner starts recording must show up in Table 1 or be
    named as a deliberate omission — never just quietly missing."""
    from experiments.export_tables import (
        PROTOCOL_OMITTED, PROTOCOL_ROWS, _check_the_labels_cover_the_fields)
    from experiments.run_full_benchmark import MODEL_FIELDS, PROTOCOL_FIELDS

    _check_the_labels_cover_the_fields()  # the real lists, as imported
    labelled = {row[0] for row in PROTOCOL_ROWS}
    assert not (set(PROTOCOL_FIELDS) | set(MODEL_FIELDS)) - labelled \
        - PROTOCOL_OMITTED


def test_task_label_drops_a_scale_of_one_and_keeps_gravity():
    assert task_label({"env_id": "HalfCheetah-v4",
                       "params": {"gravity": 4.0, "mass_scale": 1.0}}) == (
        r"\texttt{HalfCheetah-v4} ($g=4$)")
    assert task_label({"domain_name": "cheetah", "task_name": "run",
                       "params": {}}) == r"\texttt{cheetah/run}"


def test_task_label_escapes_the_underscores_dm_control_uses():
    """`point_mass` and `ball_in_cup` are real dm_control domains, and a bare
    underscore is a LaTeX error, not a typo in the output."""
    assert task_label({"domain_name": "point_mass", "task_name": "easy"}) == (
        r"\texttt{point\_mass/easy}")


def test_tasks_table_reads_the_pair_and_both_distances():
    tasks = {"task_A": {"env_id": "HalfCheetah-v4", "params": {"gravity": 9.8}},
             "task_B": {"env_id": "HalfCheetah-v4", "params": {"gravity": 4.0}}}
    runs = [run("finetuning", "gymnasium", "distance_med", seed,
                tasks=tasks, d_trans=30.0 + seed) for seed in range(5)]
    table = tasks_table(runs, ["gymnasium"], ["distance_med"])
    assert r"\texttt{HalfCheetah-v4} ($g=9.8$)" in table
    assert "0.586" in table   # d_param, Eq. 8
    assert "32.00" in table   # d_trans, median over the five seeds
