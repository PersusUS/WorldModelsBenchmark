"""
Aggregate benchmark results from the stored metrics.json files.

Every table in the paper comes from here, never from a transcribed number. The
runner writes one metrics.json per (method, family, distance, seed); this reads
them back, checks they were produced under one protocol, and prints:

  * the protocol they share, once;
  * mean +- std per cell for the forgetting metrics;
  * the task-A quality columns (F17/F18), which are what tell you whether there
    was anything to forget and whether the metrics saw it;
  * an optional paired comparison between two methods.

On the paired comparison: it is seed-paired, because two methods on the same seed
share their data and their initialisation, and pairing removes that variance. Two
things are reported instead of a t-test — with n = 5 an exact test cannot reach
p < 0.05 at all (the smallest two-sided permutation p is 2/2^5 = 0.0625), so a
parametric p-value in that regime is a statement about the normality assumption
rather than about the data:

  * `perm p`: exact two-sided paired permutation p over all 2^n sign flips.
  * `d_z`: mean difference over its standard deviation (paired effect size).

Usage:

    python experiments/summarize_results.py
    python experiments/summarize_results.py --compare replay_infinite finetuning
    python experiments/summarize_results.py --results-dir ../_devlog/archive/results-R12-finding4
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_full_benchmark import DISTANCES, FAMILY_CONFIGS, METHODS

# What the benchmark reports. PF and RD are the forgetting metrics and they are
# printed side by side rather than folded into WMF: they live on different
# scales and RD alone supplies 78-97% of the aggregate, so a single WMF column
# is a report on RD wearing a suit (F14/P7). WMF is still printed, below, under
# a heading that says what it is. FT is forward transfer measured against a
# model trained on task B from scratch (F20/P10).
DEFAULT_METRICS = ["pf", "rd", "ft"]
# family[:3] would abbreviate minigrid to "min", and a column headed "min_med"
# reads as "minimum median" rather than "MiniGrid, medium distance".
FAMILY_ABBREV = {"minigrid": "mgrid", "gymnasium": "gym", "dmcontrol": "dmc"}
QUALITY_COLUMNS = [
    ("heldout_reconstruction_A_after_task_A", "recon A|after A"),
    ("heldout_reconstruction_A_after_task_B", "recon A|after B"),
    ("heldout_reconstruction_B_after_task_B", "recon B|after B"),
    ("final_reconstruction_loss_A", "train recon A"),
    ("final_reconstruction_loss_B", "train recon B"),
]


def load_runs(results_root: Path) -> list:
    """Every metrics.json under `results_root`, as a list of dicts."""
    runs = []
    for path in sorted(results_root.glob("*/*/metrics.json")):
        with open(path) as f:
            run = json.load(f)
        run.setdefault("_path", str(path))
        runs.append(run)
    return runs


def shared_protocol(runs: list):
    """
    The protocol every run shares, or None if they disagree.

    Mixing budgets in one aggregate is the failure this returns None for; the
    caller decides how loudly to complain.
    """
    protocols = [json.dumps(r.get("protocol"), sort_keys=True) for r in runs]
    if not protocols or len(set(protocols)) != 1:
        return None
    return json.loads(protocols[0])


def cell_values(runs: list, method: str, family: str, distance: str,
                key: str) -> dict:
    """
    {seed: value} for one cell, skipping runs that lack the key.

    A stored null counts as absent: ft and d_trans are null in runs made with
    --skip-reference, and averaging them in as zero would invent a result.
    """
    return {
        r["seed"]: r[key]
        for r in runs
        if r.get("method") == method and r.get("family") == family
        and r.get("distance") == distance and r.get(key) is not None
    }


def paired_differences(runs: list, method_a: str, method_b: str, family: str,
                       distance: str, key: str):
    """
    (seeds, values_a, values_b) restricted to seeds both methods ran.

    Unpaired seeds are dropped rather than filled: a comparison that silently
    mixes 5 seeds of one method with 3 of the other is worse than a smaller one.
    """
    a = cell_values(runs, method_a, family, distance, key)
    b = cell_values(runs, method_b, family, distance, key)
    seeds = sorted(set(a) & set(b))
    return (seeds,
            np.array([a[s] for s in seeds], dtype=float),
            np.array([b[s] for s in seeds], dtype=float))


def permutation_p(diffs: np.ndarray) -> float:
    """
    Exact two-sided paired permutation p over all 2^n sign flips.

    Exact rather than asymptotic because n is 5. Note the floor: the smallest
    value this can return is 2 / 2^n, so at n = 5 no result can come out below
    0.0625 however large the effect.
    """
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(diffs)))
    count = sum(
        1 for signs in itertools.product([1, -1], repeat=n)
        if abs(float(np.mean(diffs * np.array(signs)))) >= observed - 1e-12
    )
    return count / 2 ** n


def effect_size(diffs: np.ndarray) -> float:
    """Paired Cohen's d_z: mean difference over its standard deviation."""
    if len(diffs) < 2:
        return float("nan")
    sd = float(np.std(diffs, ddof=1))
    return float("inf") if sd == 0 else float(np.mean(diffs)) / sd


def print_metric_table(runs, methods, families, distances, key):
    # The metric name goes on its own line: as a corner cell it would be wider
    # than the method column for the longer keys and skew every row under it.
    print(f"{key}:")
    header = f"{'':<20}"
    for family in families:
        for distance in distances:
            col = f"{FAMILY_ABBREV.get(family, family[:5])}_{distance.split('_')[1]}"
            header += f" | {col:>17}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<20}"
        for family in families:
            for distance in distances:
                values = list(cell_values(runs, method, family, distance,
                                          key).values())
                if values:
                    cell = f"{np.mean(values):+.3f}+-{np.std(values):.3f}"
                else:
                    cell = "N/A"
                row += f" | {cell:>17}"
        print(row)
    print()


def rd_share(runs, method, family, distance, weights=(0.4, 0.4)):
    """
    Fraction of |alpha*PF + beta*RD| that RD contributes, averaged over seeds.

    This is the number behind P7: if it sits near 1, WMF is RD with extra
    steps, and reporting the aggregate as a single headline hides which metric
    discriminates.
    """
    alpha, beta = weights
    pf = cell_values(runs, method, family, distance, "pf")
    rd = cell_values(runs, method, family, distance, "rd")
    shares = []
    for seed in sorted(set(pf) & set(rd)):
        total = abs(alpha * pf[seed]) + abs(beta * rd[seed])
        if total > 0:
            shares.append(abs(beta * rd[seed]) / total)
    return float(np.mean(shares)) if shares else float("nan")


def print_aggregate_section(runs, methods, families, distances):
    """WMF, plus the diagnostic that says how much of it is RD."""
    print("=" * 78)
    print("LEGACY AGGREGATE (Eq. 6): WMF = 0.4*PF + 0.4*RD + 0.2*PIS, with PIS")
    print("hardcoded to 0. Printed for continuity with the paper, not as the")
    print("headline number - the share column says why.")
    print("=" * 78)
    print_metric_table(runs, methods, families, distances, "wmf")

    print("share of |WMF| contributed by RD:")
    header = f"{'':<20}"
    for family in families:
        for distance in distances:
            col = f"{FAMILY_ABBREV.get(family, family[:5])}_{distance.split('_')[1]}"
            header += f" | {col:>17}"
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:<20}"
        for family in families:
            for distance in distances:
                share = rd_share(runs, method, family, distance)
                cell = "N/A" if np.isnan(share) else f"{100 * share:.1f}%"
                row += f" | {cell:>17}"
        print(row)
    print()


def print_distance_table(runs, families, distances):
    """
    d_trans per (family, distance): Eq. 9, between one model per environment.

    It is a property of the task pair and the seed, not of the method, so every
    method's cell stores the same value and this table collapses them.
    """
    values = {}
    for family in families:
        for distance in distances:
            per_seed = {r["seed"]: r["d_trans"] for r in runs
                        if r.get("family") == family
                        and r.get("distance") == distance
                        and r.get("d_trans") is not None}
            if per_seed:
                values[(family, distance)] = list(per_seed.values())
    if not values:
        return

    print("=" * 78)
    print("DYNAMIC DISTANCE d_trans (Eq. 9): KL between the transition models")
    print("of two environments, each trained on its own task from scratch.")
    print("=" * 78)
    for (family, distance), vals in values.items():
        print(f"  {family:<10} {distance:<14} "
              f"d_trans = {np.mean(vals):8.4f} +- {np.std(vals):.4f} "
              f"(n={len(vals)})")
    print()


def print_comparison(runs, method_a, method_b, families, distances, keys):
    print("=" * 78)
    print(f"PAIRED COMPARISON: {method_a} - {method_b}   (per seed)")
    print("=" * 78)
    for family in families:
        for distance in distances:
            printed_header = False
            for key in keys:
                seeds, a, b = paired_differences(runs, method_a, method_b,
                                                 family, distance, key)
                if len(seeds) == 0:
                    continue
                if not printed_header:
                    print(f"\n### {family} / {distance}  (n={len(seeds)}, "
                          f"seeds={seeds})")
                    printed_header = True
                diffs = a - b
                wins = int(np.sum(diffs > 0))
                print(f"  {key:<6} {method_a}={np.mean(a):+8.3f}  "
                      f"{method_b}={np.mean(b):+8.3f}  "
                      f"delta={np.mean(diffs):+8.3f}  "
                      f"higher in {wins}/{len(seeds)}  "
                      f"perm p={permutation_p(diffs):.4f}  "
                      f"d_z={effect_size(diffs):+.2f}")
    print()
    print(f"With n=5 the exact two-sided permutation p cannot go below "
          f"{2 / 2 ** 5:.4f}; read d_z and the win count alongside it.")
    print()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--families", nargs="+", default=list(FAMILY_CONFIGS))
    parser.add_argument("--distances", nargs="+", default=DISTANCES)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--compare", nargs=2, metavar=("METHOD_A", "METHOD_B"),
                        help="seed-paired comparison of two methods")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no metrics.json found under {args.results_dir}")

    present = {(r.get("method"), r.get("family"), r.get("distance"))
               for r in runs}
    methods = [m for m in args.methods if any(p[0] == m for p in present)]
    families = [f for f in args.families if any(p[1] == f for p in present)]
    distances = [d for d in args.distances if any(p[2] == d for p in present)]

    print(f"{len(runs)} run(s) from {args.results_dir}")
    protocol = shared_protocol(runs)
    if protocol is None:
        print("\nWARNING: these runs do NOT share one protocol. Any mean below "
              "averages different training budgets — fix the directory before "
              "reading the numbers.\n")
    else:
        print(f"protocol: n_train={protocol['n_train']} "
              f"n_collect={protocol['n_collect']} "
              f"batch_size={protocol['batch_size']} "
              f"seq_len={protocol['seq_len']} "
              f"seeds={protocol['seeds']}")

    print("\n" + "=" * 78)
    print("FORGETTING METRICS (mean +- std across seeds)")
    print("PF and RD are measured on latents encoded once by the post-task-A")
    print("model, so they score the transition component M in a fixed latent")
    print("basis and are blind to drift in the encoder itself (F18). The pixel")
    print("columns further down are the scale that sees it.")
    print("=" * 78)
    for key in args.metrics:
        print_metric_table(runs, methods, families, distances, key)

    print_aggregate_section(runs, methods, families, distances)
    print_distance_table(runs, families, distances)

    print("=" * 78)
    print("TASK QUALITY IN PIXELS (F17/F18): was there anything to forget, and")
    print("did the metrics above see it? Squared error per held-out frame, the")
    print("one scale that is comparable across models and across budgets.")
    print("=" * 78)
    for key, label in QUALITY_COLUMNS:
        if any(r.get(key) is not None for r in runs):
            print_metric_table(runs, methods, families, distances, key)

    nan_runs = [r for r in runs
                if r.get("n_nan_steps_A", 0) or r.get("n_nan_steps_B", 0)]
    if nan_runs:
        print(f"WARNING: {len(nan_runs)} run(s) dropped NaN steps:")
        for r in nan_runs:
            print(f"  {r['_path']}: A={r.get('n_nan_steps_A')} "
                  f"B={r.get('n_nan_steps_B')}")
        print()

    if args.compare:
        print_comparison(runs, args.compare[0], args.compare[1], families,
                         distances, args.metrics)


if __name__ == "__main__":
    main()
