"""
Tests for experiments/plot_final.py.

The figure is the one artefact a reader looks at before anything else, so what
is pinned here is what it plots (PF and RD, never WMF) and what it puts on the X
axis (the measured d_trans when the runs carry it, an ordinal fallback when they
do not).
"""
import json

import pytest

from experiments.plot_final import (
    METRICS,
    cell,
    distance_axis,
    draw,
    main,
    parse_args,
)

DISTANCES = ["distance_min", "distance_med", "distance_max"]


def write_run(root, method, family, distance, seed, **fields):
    directory = root / method / f"{family}_{distance}_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    run = {"method": method, "family": family, "distance": distance,
           "seed": seed}
    run.update(fields)
    (directory / "metrics.json").write_text(json.dumps(run))


def write_grid(root, family="minigrid", d_trans=None):
    for i, distance in enumerate(DISTANCES):
        for seed in (0, 1):
            write_run(root, "finetuning", family, distance, seed,
                      pf=1.0 + i, rd=10.0 + i,
                      d_trans=None if d_trans is None else d_trans[i])


def test_the_figure_plots_the_reported_metrics_not_the_aggregate():
    assert [key for key, _ in METRICS] == ["pf", "rd"]


def test_cell_averages_over_seeds(tmp_path):
    write_run(tmp_path, "finetuning", "minigrid", "distance_min", 0, pf=1.0)
    write_run(tmp_path, "finetuning", "minigrid", "distance_min", 1, pf=3.0)
    mean, std = cell(tmp_path, "finetuning", "minigrid", "distance_min", "pf")
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)


def test_cell_is_empty_when_the_key_is_null(tmp_path):
    """ft and d_trans are null under --skip-reference; null is not zero."""
    write_run(tmp_path, "finetuning", "minigrid", "distance_min", 0, ft=None)
    assert cell(tmp_path, "finetuning", "minigrid", "distance_min", "ft") \
        == (None, None)


def test_x_axis_uses_the_measured_distance_when_it_is_there(tmp_path):
    write_grid(tmp_path, d_trans=[2.0, 5.0, 9.0])
    xs, labels, xlabel = distance_axis(tmp_path, "minigrid")
    assert xs == [2.0, 5.0, 9.0]
    assert labels == ["2.00", "5.00", "9.00"]
    assert "d_{trans}" in xlabel


def test_x_axis_falls_back_to_levels_and_says_so(tmp_path):
    """Runs made before d_trans existed, or with --skip-reference."""
    write_grid(tmp_path, d_trans=None)
    xs, labels, xlabel = distance_axis(tmp_path, "minigrid")
    assert xs == [1, 2, 3]
    assert labels == ["Min", "Med", "Max"]
    assert "level" in xlabel


def test_a_partial_distance_column_does_not_mix_scales(tmp_path):
    """Two levels on d_trans and one on an ordinal position would put points
    from different scales on the same axis."""
    write_grid(tmp_path, d_trans=[2.0, 5.0, 9.0])
    for seed in (0, 1):
        write_run(tmp_path, "finetuning", "minigrid", "distance_max", seed,
                  pf=3.0, rd=12.0, d_trans=None)
    xs, _labels, xlabel = distance_axis(tmp_path, "minigrid")
    assert xs == [1, 2, 3]
    assert "level" in xlabel


def test_draw_writes_both_formats(tmp_path):
    write_grid(tmp_path / "runs", d_trans=[2.0, 5.0, 9.0])
    paths = draw(tmp_path / "runs", tmp_path / "figures", "figure")
    assert [p.name for p in paths] == ["figure.pdf", "figure.png"]
    for path in paths:
        assert path.stat().st_size > 0


def test_draw_survives_a_family_with_no_runs(tmp_path):
    """Only one family is ever run during development."""
    write_grid(tmp_path / "runs", family="minigrid", d_trans=[1.0, 2.0, 3.0])
    assert draw(tmp_path / "runs", tmp_path / "figures", "figure")


def test_main_refuses_an_empty_results_directory(tmp_path):
    with pytest.raises(SystemExit, match="no metrics.json"):
        main(["--results-dir", str(tmp_path), "--out", str(tmp_path)])


def test_defaults_point_at_the_results_directory():
    args = parse_args([])
    assert str(args.results_dir) == "results"
