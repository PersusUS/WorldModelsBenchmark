"""
Figure 1: forgetting against the labelled distance level, one panel per family.

This is the paper's headline in a picture — RD peaks at the *medium* level in
all three families, so the ordinal axis the benchmark was built around does not
order forgetting.

It is a different figure from `plot_final.py`, which puts the measured distance
`d_trans` on the X axis. That was the right choice when `d_trans` was believed
to be the better axis; it is the wrong choice for showing this result, because
sorting the cells by `d_trans` reorders them and the peak stops being visible.
`plot_final.py` survives as the appendix figure, where the point is precisely
that the measured axis does not separate the levels either (F29).

Two presentation decisions, both deliberate:

  * UG-MTM is drawn but not emphasised, and is not part of the aggregate. It
    freezes its encoder, which puts its RD orders of magnitude below the other
    four in DMControl; a bold "ours" line here would invite reading the figure
    as a method comparison, which is not what it is about.
  * The Y axis is logarithmic. RD spans 1.15 to 86 across cells and 0.002 to
    8135 across methods; on a linear axis every panel but one is a flat line.

    python experiments/plot_axis.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_full_benchmark import DISTANCES, FAMILY_CONFIGS
from experiments.summarize_results import (
    AXIS_METHODS,
    axis_value,
    cell_values,
    check_runs_consistent,
    load_runs,
)

FAMILY_TITLE = {"minigrid": "MiniGrid (discrete)",
                "gymnasium": "Gymnasium (continuous)",
                "dmcontrol": "DMControl (visual)"}
LEVEL_TICK = {"distance_min": "min", "distance_med": "med",
              "distance_max": "max"}
METHOD_STYLE = {
    "finetuning": ("Fine-tuning", "#e41a1c", "s", "--"),
    "replay_infinite": ("Replay", "#377eb8", "D", "-."),
    "ewc": ("EWC", "#ff7f00", "^", ":"),
    "progressive_nets": ("Prog. Nets", "#4daf4a", "v", "--"),
    "ug_mtm": ("UG-MTM", "#999999", "o", "-"),
}


def draw(runs, families, distances, methods, axis_methods, key="rd"):
    """Returns the figure. Split out from main so a test can inspect it."""
    fig, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 3.6))
    axes = np.atleast_1d(axes)
    x = np.arange(len(distances))

    for ax, family in zip(axes, families):
        for method in methods:
            label, colour, marker, style = METHOD_STYLE.get(
                method, (method, "#666666", "o", "-"))
            medians, lo, hi = [], [], []
            for distance in distances:
                values = list(cell_values(runs, method, family, distance,
                                          key).values())
                if values:
                    medians.append(float(np.median(values)))
                    lo.append(medians[-1] - float(np.min(values)))
                    hi.append(float(np.max(values)) - medians[-1])
                else:
                    medians.append(np.nan)
                    lo.append(0.0)
                    hi.append(0.0)
            ax.errorbar(x, medians, yerr=[lo, hi], label=label, color=colour,
                        marker=marker, linestyle=style, linewidth=1.4,
                        markersize=5, capsize=3, alpha=0.85)

        # The aggregate the finding is stated over, drawn on top so the peak is
        # legible without tracing four lines.
        aggregate = [axis_value(runs, axis_methods, family, d, key)
                     for d in distances]
        ax.plot(x, aggregate, color="black", linewidth=2.6, marker="o",
                markersize=7, zorder=5,
                label="Median over RSSM methods" if family == families[0] else None)
        if not all(np.isnan(v) for v in aggregate):
            peak = int(np.nanargmax(aggregate))
            ax.annotate("peak", (x[peak], aggregate[peak]),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_TICK.get(d, d) for d in distances])
        ax.set_yscale("log")
        # Headroom for the "peak" annotation, which otherwise collides with the
        # panel title wherever the peak is also the tallest point drawn.
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top * 2.2)
        ax.set_title(FAMILY_TITLE.get(family, family), fontweight="bold")
        ax.set_xlabel("Labelled distance level")
        ax.grid(alpha=0.3, linestyle=":")
    axes[0].set_ylabel("RD (rollout divergence), log scale")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default=Path("results"), type=Path)
    parser.add_argument("--out", default=Path("../paper/figures"), type=Path)
    parser.add_argument("--axis-methods", nargs="+", default=AXIS_METHODS)
    args = parser.parse_args(argv)

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no metrics.json found under {args.results_dir}")
    check_runs_consistent(runs)

    present = {(r.get("method"), r.get("family"), r.get("distance"))
               for r in runs}
    families = [f for f in FAMILY_CONFIGS if any(p[1] == f for p in present)]
    distances = [d for d in DISTANCES if any(p[2] == d for p in present)]
    methods = [m for m in METHOD_STYLE if any(p[0] == m for p in present)]

    fig = draw(runs, families, distances, methods, args.axis_methods)
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = args.out / f"axis_peak.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
