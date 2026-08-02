"""
The benchmark's main figure: forgetting as a function of dynamic distance.

One row per reported metric (PF and RD) and one column per family. There is no
WMF panel on purpose: with PIS hardcoded to zero the aggregate is
0.4*PF + 0.4*RD, of which RD supplies 78-97%, so a WMF panel is an RD panel
with a different label. `summarize_results.py` still prints WMF next to the
share of it that comes from RD; every table belongs there, and this file only
draws.

The X axis is the measured distance between the two environments, not an
ordinal label. `d_trans` (Eq. 9) is computed for all three families now, so it
is used wherever the runs carry it; a family whose runs predate that falls back
to Min/Med/Max positions, and the axis label says which of the two is being
shown.

    python experiments/plot_final.py
    python experiments/plot_final.py --results-dir results --out results/figures
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

METHODS = ["finetuning", "replay_infinite", "ewc", "progressive_nets", "ug_mtm"]
LABELS = ["Finetuning", "Replay (Infinite)", "EWC", "Progressive Nets",
          "UG-MTM (ours)"]
COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#4daf4a", "#000000"]
MARKERS = ["s", "D", "^", "v", "o"]
LINES = ["--", "-.", ":", "--", "-"]
LW = [1.8, 1.8, 1.8, 1.8, 2.5]

FAMILIES = ["minigrid", "gymnasium", "dmcontrol"]
FAM_LABELS = ["MiniGrid (Discrete)", "Gymnasium (Continuous)",
              "DMControl (Visual)"]
DISTANCES = ["distance_min", "distance_med", "distance_max"]
METRICS = [("pf", "PF (prediction fidelity)"),
           ("rd", "RD (rollout divergence)")]


def cell(results_root, method, family, distance, key):
    """
    (median, (down, up)) of `key` over the seeds of one cell, or (None, None).

    Median and full seed range, the same policy the tables use (D15/P12), so
    the figure and the table cannot disagree about the same cell. The bars are
    deliberately asymmetric: RD is an unbounded KL and a method that ends up
    overconfident gets a long tail upward only (F23), which a symmetric std
    would draw as an equal error in both directions.
    """
    values = []
    for path in results_root.glob(f"{method}/{family}_{distance}_*/metrics.json"):
        value = json.load(open(path)).get(key)
        if value is not None:
            values.append(float(value))
    if not values:
        return None, None
    median = float(np.median(values))
    return median, (median - min(values), max(values) - median)


def distance_axis(results_root, family):
    """
    (x positions, tick labels, axis label) for one family.

    d_trans is a property of the task pair, so it is read from whichever runs
    of that family carry it, regardless of method.
    """
    xs = []
    for distance in DISTANCES:
        values = [
            json.load(open(path)).get("d_trans")
            for path in results_root.glob(f"*/{family}_{distance}_*/metrics.json")
        ]
        values = [v for v in values if v is not None]
        xs.append(float(np.median(values)) if values else None)

    if all(x is not None for x in xs):
        return xs, [f"{x:.2f}" for x in xs], "Dynamic distance $d_{trans}$"
    return [1, 2, 3], ["Min", "Med", "Max"], "Dynamic distance level"


def draw(results_root, out_dir, stem):
    fig, axes = plt.subplots(len(METRICS), len(FAMILIES),
                             figsize=(14, 4.2 * len(METRICS)))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(wspace=0.35, hspace=0.35)

    for col, (family, fam_label) in enumerate(zip(FAMILIES, FAM_LABELS)):
        xvals, xticklabels, xlabel = distance_axis(results_root, family)

        for row, (key, metric_label) in enumerate(METRICS):
            ax = axes[row][col]
            extent_seen = []
            for i, (method, label) in enumerate(zip(METHODS, LABELS)):
                medians, spreads, xs = [], [], []
                for x, distance in zip(xvals, DISTANCES):
                    median, spread = cell(results_root, method, family,
                                          distance, key)
                    if median is not None:
                        medians.append(median)
                        spreads.append(spread)
                        xs.append(x)
                if not medians:
                    continue
                # What the panel has to fit is the bars, not the centres: the
                # seed that makes a cell heavy-tailed is a bar end, and it is
                # the reason the axis needs a log scale at all.
                extent_seen.extend(m - d for m, (d, _) in zip(medians, spreads))
                extent_seen.extend(m + u for m, (_, u) in zip(medians, spreads))
                ax.errorbar(
                    xs, medians, yerr=np.array(spreads).T, label=label,
                    color=COLORS[i], linestyle=LINES[i], marker=MARKERS[i],
                    linewidth=LW[i], markersize=6, capsize=4,
                    zorder=5 if method == "ug_mtm" else 2,
                )

            # Zero is the no-forgetting line, and PF goes negative often
            # enough that it has to be visible. RD is a KL and cannot be
            # negative, so the line only means something on the PF row.
            if key != "rd":
                ax.axhline(0, color="gray", linewidth=0.8, linestyle=":",
                           alpha=0.6)
            if row == 0:
                ax.set_title(fam_label, fontsize=13, fontweight="bold", pad=8)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(metric_label if col == 0 else "", fontsize=10)
            ax.set_xticks(xvals)
            # Two levels can land on the same measured distance -- dmcontrol's
            # med and max do -- and overlapping labels look like a plotting
            # bug rather than the result it is.
            ax.set_xticklabels(xticklabels, fontsize=9, rotation=30,
                               ha="right")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.tick_params(labelsize=9)

            if key == "rd" and extent_seen and min(extent_seen) > 0 and \
                    max(extent_seen) / min(extent_seen) > 50:
                # RD is a KL, so it is non-negative, and it spans three orders
                # of magnitude once UG-MTM's outlier cell is in (F23). On a
                # linear axis that one point flattens every other method to
                # the zero line.
                ax.set_yscale("log")
                ax.set_ylabel(f"{metric_label} (log)" if col == 0 else "",
                              fontsize=10)
            else:
                # Independent Y axis per panel: PF and RD differ by an order of
                # magnitude, which is the whole reason they are not aggregated.
                ax.autoscale(axis="y")
                ymin, ymax = ax.get_ylim()
                pad = (ymax - ymin) * 0.15
                ax.set_ylim(ymin - pad, ymax + pad)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", ncol=5,
                   fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.06))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved: {path}")
    plt.close(fig)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--out", default=Path("results/figures"), type=Path)
    parser.add_argument("--stem", default="forgetting_vs_distance",
                        help="basename of the .pdf and .png written")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not any(args.results_dir.glob("*/*/metrics.json")):
        raise SystemExit(f"no metrics.json found under {args.results_dir}")
    draw(args.results_dir, args.out, args.stem)
    print("Tables come from summarize_results.py, not from here.")


if __name__ == "__main__":
    main()
