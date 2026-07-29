"""
experiments/plot_final.py
Final publication-quality figure for WMF Benchmark paper.
- Independent Y-axis per panel (makes each family readable)
- Gymnasium X-axis shows d_param values instead of ordinal labels
- Clean style suitable for workshop paper
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark.distances import compute_d_param
from omegaconf import OmegaConf

# --- Config ---
METHODS = ["finetuning", "replay_infinite", "ewc", "progressive_nets", "ug_mtm"]
LABELS  = ["Finetuning", "Replay (Infinite)", "EWC", "Progressive Nets", "UG-MTM (ours)"]
COLORS  = ["#e41a1c", "#377eb8", "#ff7f00", "#4daf4a", "#000000"]
MARKERS = ["s",        "D",       "^",       "v",       "o"]
LINES   = ["--",       "-.",      ":",       "--",      "-"]
LW      = [1.8,        1.8,       1.8,       1.8,       2.5]

FAMILIES   = ["minigrid",    "gymnasium",       "dmcontrol"]
FAM_LABELS = ["MiniGrid (Discrete)", "Gymnasium (Continuous)", "DMControl (Visual)"]
DISTANCES  = ["distance_min", "distance_med", "distance_max"]

# Compute Gymnasium d_param values for X-axis
gym_cfg = OmegaConf.load("configs/benchmark/gymnasium.yaml")
gym_xvals = []
for dist in DISTANCES:
    seq = gym_cfg.benchmark.sequences[dist]
    d = compute_d_param(dict(seq.task_A.params), dict(seq.task_B.params))
    gym_xvals.append(round(d, 3))
print(f"Gymnasium d_param values: {gym_xvals}")

# --- Load results ---
def load_wmf(method, family, distance):
    files = list(Path("results").glob(
        f"{method}/{family}_{distance}_*/metrics.json"))
    if not files:
        return None, None
    vals = [json.load(open(f))["wmf"] for f in files]
    return np.mean(vals), np.std(vals)

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.subplots_adjust(wspace=0.35)

for ax_i, (family, fam_label) in enumerate(zip(FAMILIES, FAM_LABELS)):
    ax = axes[ax_i]

    # X-axis values
    if family == "gymnasium":
        xvals  = gym_xvals
        xlabel = "Dynamic distance $d_{param}$"
        xticks = gym_xvals
        xticklabels = [f"{v:.3f}" for v in gym_xvals]
    else:
        xvals  = [1, 2, 3]
        xlabel = "Dynamic distance level"
        xticks = [1, 2, 3]
        xticklabels = ["Min", "Med", "Max"]

    for m_i, (method, label) in enumerate(zip(METHODS, LABELS)):
        means, stds, xs = [], [], []
        for x, dist in zip(xvals, DISTANCES):
            mean, std = load_wmf(method, family, dist)
            if mean is not None:
                means.append(mean)
                stds.append(std)
                xs.append(x)
        if not means:
            continue
        ax.errorbar(
            xs, means, yerr=stds,
            label=label,
            color=COLORS[m_i],
            linestyle=LINES[m_i],
            marker=MARKERS[m_i],
            linewidth=LW[m_i],
            markersize=6,
            capsize=4,
            zorder=5 if method == "ug_mtm" else 2
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_title(fam_label, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("WMF (World Model Forgetting)" if ax_i == 0 else "", fontsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(labelsize=9)

    # Independent Y-axis: auto-scale per panel with 10% padding
    ax.autoscale(axis="y")
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.15
    ax.set_ylim(ymin - pad, ymax + pad)

# Shared legend below figure
handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels_leg,
    loc="lower center",
    ncol=5,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.12),
)

Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("results/figures/figurasfinalv2.pdf",
            bbox_inches="tight", dpi=150)
fig.savefig("results/figures/figurasfinalv2.png",
            bbox_inches="tight", dpi=150)
print("Saved: results/figures/figurasfinalv2.pdf")
print("Saved: results/figures/figurasfinalv2.png")

# --- Print full results table ---
print("\nFULL RESULTS TABLE (mean WMF, 5 seeds)")
print(f"{'Method':<22}", end="")
for fam in FAMILIES:
    for dist in ["min","med","max"]:
        print(f" {fam[:3]}_{dist[:3]}", end="")
print()
for method, label in zip(METHODS, LABELS):
    print(f"{label:<22}", end="")
    for family in FAMILIES:
        for distance in DISTANCES:
            mean, std = load_wmf(method, family, distance)
            if mean is not None:
                print(f" {mean:+.4f}", end="")
            else:
                print(f"    N/A", end="")
    print()
