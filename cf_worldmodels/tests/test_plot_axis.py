"""
Tests for experiments/plot_axis.py.

Figure 1 makes the paper's headline claim visually, so what is pinned here is
that it makes *that* claim: the X axis is the labelled level (not the measured
distance, which would reorder the cells and hide the peak), the peak is
annotated where the aggregate actually peaks, and the aggregate excludes
UG-MTM for the same reason the table does.
"""
import matplotlib
matplotlib.use("Agg")

from experiments.plot_axis import LEVEL_TICK, draw

LEVELS = ["distance_min", "distance_med", "distance_max"]


def run(method, family, distance, seed, rd):
    return {"method": method, "family": family, "distance": distance,
            "seed": seed, "rd": rd,
            "protocol": {"n_train": 5000, "seeds": [0]}}


def grid(values_by_level, method="finetuning", family="minigrid"):
    return [run(method, family, f"distance_{level}", 0, value)
            for level, value in values_by_level.items()]


def test_the_x_axis_is_the_label_not_the_measured_distance():
    """Sorting by d_trans reorders the cells and the peak stops being visible;
    that is what plot_final.py does and why this figure exists separately."""
    runs = grid({"min": 10.0, "med": 30.0, "max": 20.0})
    fig = draw(runs, ["minigrid"], LEVELS, ["finetuning"], ["finetuning"])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        [LEVEL_TICK[d] for d in LEVELS]
    assert "Labelled distance level" in ax.get_xlabel()


def test_the_peak_is_annotated_where_the_aggregate_peaks():
    runs = grid({"min": 10.0, "med": 30.0, "max": 20.0})
    fig = draw(runs, ["minigrid"], LEVELS, ["finetuning"], ["finetuning"])
    ax = fig.axes[0]
    peaks = [a for a in ax.texts if a.get_text() == "peak"]
    assert len(peaks) == 1
    assert peaks[0].xy[0] == 1          # the medium level


def test_a_monotone_family_annotates_its_last_level():
    runs = grid({"min": 10.0, "med": 20.0, "max": 30.0})
    fig = draw(runs, ["minigrid"], LEVELS, ["finetuning"], ["finetuning"])
    peaks = [a for a in fig.axes[0].texts if a.get_text() == "peak"]
    assert peaks[0].xy[0] == 2


def test_ug_mtm_is_drawn_but_kept_out_of_the_aggregate():
    """It freezes its encoder, so pooling it would average two scales rather
    than read the axis -- the same exclusion summarize_results declares."""
    runs = grid({"min": 10.0, "med": 30.0, "max": 20.0})
    runs += grid({"min": 0.001, "med": 0.002, "max": 0.003}, method="ug_mtm")
    fig = draw(runs, ["minigrid"], LEVELS, ["finetuning", "ug_mtm"],
               ["finetuning"])
    ax = fig.axes[0]
    # errorbar() labels its container, not the line inside it.
    labels = [c.get_label() for c in ax.containers]
    assert "UG-MTM" in labels
    assert "Fine-tuning" in labels
    # The aggregate follows fine-tuning alone: its peak is still the medium
    # level, which a pool including ug_mtm's three orders of magnitude below
    # would not disturb but would misrepresent.
    peaks = [a for a in ax.texts if a.get_text() == "peak"]
    assert peaks[0].xy[1] == 30.0


def test_the_y_axis_is_logarithmic():
    """RD spans four orders of magnitude across methods; on a linear axis
    every panel but one is a flat line."""
    runs = grid({"min": 10.0, "med": 30.0, "max": 20.0})
    fig = draw(runs, ["minigrid"], LEVELS, ["finetuning"], ["finetuning"])
    assert fig.axes[0].get_yscale() == "log"


def test_one_panel_per_family():
    runs = grid({"min": 10.0, "med": 30.0, "max": 20.0})
    runs += grid({"min": 5.0, "med": 9.0, "max": 7.0}, family="gymnasium")
    fig = draw(runs, ["minigrid", "gymnasium"], LEVELS, ["finetuning"],
               ["finetuning"])
    assert len(fig.axes) == 2
