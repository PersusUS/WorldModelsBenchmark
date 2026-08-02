"""
Aggregate benchmark results from the stored metrics.json files.

Every table in the paper comes from here, never from a transcribed number. The
runner writes one metrics.json per (method, family, distance, seed); this reads
them back, checks they were produced under one protocol, and prints:

  * the protocol they share, once;
  * median [min, max] per cell for the forgetting metrics;
  * whether the labelled distance level predicts forgetting at all (F27);
  * which cells produced no forgetting in any method, i.e. the controls (F22);
  * the task-A quality columns (F17/F18), which are what tell you whether there
    was anything to forget and whether the metrics saw it;
  * an optional paired comparison between two methods.

On the median: cells are summarised by their median and their full seed range,
not by mean +- std (D15/P12). RD is a KL and is unbounded above, so a method
that produces an overconfident post-task-B model gets a heavy right tail by
construction — ug_mtm on minigrid/distance_max runs 17.7, 40.0, 520, 574, 4364
across its five seeds (F23), and a mean of those describes none of them. The
skew is not noise to be smoothed away: cells whose upper half stretches five
times further than their lower half are marked with `!` and listed underneath,
because that is a finding about the method, not a nuisance.

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
# The four methods that share the RSSM architecture. The distance-axis section
# (F27) is about the axis, not about the methods, so it aggregates over these
# and leaves ug_mtm out: ug_mtm freezes its encoder, which puts its RD two
# orders of magnitude below the others in dmcontrol (0.002 against 1.2), and an
# aggregate over all five would be an average of two scales rather than a
# reading of the axis. The exclusion is declared here and printed in the
# section header; --axis-methods overrides it.
AXIS_METHODS = ["finetuning", "replay_infinite", "ewc", "progressive_nets"]
# family[:3] would abbreviate minigrid to "min", and a column headed "min_med"
# reads as "minimum median" rather than "MiniGrid, medium distance".
FAMILY_ABBREV = {"minigrid": "mgrid", "gymnasium": "gym", "dmcontrol": "dmc"}
CELL_WIDTH = 23
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


def check_runs_consistent(runs: list) -> None:
    """
    Refuse a results directory whose runs cannot be aggregated together.

    Two ways a cell stops meaning one thing, and both have happened: runs made
    under different budgets (the runner's own check, `check_protocol_consistency`)
    and runs made on different environment pairs after a level was edited
    (F26). The runner catches both before training; this catches them before
    *reading*, so every consumer of `load_runs` inherits one notion of a
    consistent directory instead of each table builder inventing its own.

    Raises ValueError. A warning would be wrong here: the failure is silent
    averaging, and a warning is what gets scrolled past.
    """
    with_protocol = [r for r in runs if r.get("protocol")]
    if with_protocol and shared_protocol(with_protocol) is None:
        distinct = len({json.dumps(r["protocol"], sort_keys=True)
                        for r in with_protocol})
        raise ValueError(
            f"{distinct} different protocols across {len(with_protocol)} "
            "runs: they were produced under different budgets and cannot be "
            "aggregated. Archive the odd ones or restrict the results "
            "directory."
        )

    by_cell = {}
    for run in runs:
        if run.get("tasks") is None:
            continue
        by_cell.setdefault((run.get("family"), run.get("distance")),
                           []).append(run)
    for (family, distance), cell in by_cell.items():
        first = cell[0]["tasks"]
        if any(r["tasks"] != first for r in cell):
            raise ValueError(
                f"{family}/{distance} contains runs from more than one task "
                "pair; a level was edited and stale results were left behind."
            )


def cell_tasks(runs: list, family: str, distance: str):
    """The task pair one cell was run on, or None if no run recorded it."""
    for run in runs:
        if (run.get("family") == family and run.get("distance") == distance
                and run.get("tasks")):
            return run["tasks"]
    return None


def cell_d_trans(runs: list, family: str, distance: str) -> dict:
    """
    {seed: d_trans} for one cell.

    d_trans is a property of the task pair and the seed, not of the method, so
    every method's run in a cell stores the same value and this collapses them.
    A stored null is absent, not zero, for the reason `cell_values` gives.
    """
    return {r["seed"]: r["d_trans"] for r in runs
            if r.get("family") == family and r.get("distance") == distance
            and r.get("d_trans") is not None}


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


def is_right_skewed(values, factor: float = 5.0) -> bool:
    """
    True when the upper half of the sample stretches `factor` times further
    from the median than the lower half does.

    The test is on order statistics rather than on mean/median or a coefficient
    of variation, both of which fire on cells that merely straddle zero — EWC's
    PF sits at -0.005 with a spread of 0.083 and is perfectly symmetric. What
    this catches is the shape that makes a mean meaningless: one seed an order
    of magnitude above the rest (F23).
    """
    v = sorted(float(x) for x in values)
    if len(v) < 3:
        return False
    median = float(np.median(v))
    return (v[-1] - median) > factor * (median - v[0])


def cell_summary(values, flag_skew: bool = True) -> str:
    """
    One cell as `median [min, max]`, with `!` if the sample is right-skewed.

    Four significant figures rather than three decimals: the same column has to
    hold -0.062 and 4364, and a fixed decimal count either loses EWC's PF or
    prints eight characters of noise on ug_mtm's RD.
    """
    v = sorted(float(x) for x in values)
    if not v:
        return "N/A"
    mark = "!" if flag_skew and is_right_skewed(v) else ""
    return f"{np.median(v):+.4g} [{v[0]:.4g},{v[-1]:.4g}]{mark}"


def print_metric_table(runs, methods, families, distances, key,
                       flag_skew: bool = True):
    """
    One table of `key`, and the list of cells whose sample is skewed.

    `flag_skew` is off for the pixel columns. The marker is there to say that a
    mean would have been the wrong summary, which is a statement about metrics
    that are unbounded above — RD is a KL, PF a difference of NLLs. A held-out
    reconstruction error is bounded by the data, and marking the ordinary 25-35
    spread of a Gymnasium baseline would point at everything and therefore at
    nothing. The range is printed either way.
    """
    # The metric name goes on its own line: as a corner cell it would be wider
    # than the method column for the longer keys and skew every row under it.
    print(f"{key}:")
    header = f"{'':<20}"
    for family in families:
        for distance in distances:
            col = f"{FAMILY_ABBREV.get(family, family[:5])}_{distance.split('_')[1]}"
            header += f" | {col:>{CELL_WIDTH}}"
    print(header)
    print("-" * len(header))

    skewed = []
    for method in methods:
        row = f"{method:<20}"
        for family in families:
            for distance in distances:
                values = list(cell_values(runs, method, family, distance,
                                          key).values())
                if flag_skew and values and is_right_skewed(values):
                    skewed.append((method, family, distance, sorted(values)))
                row += f" | {cell_summary(values, flag_skew):>{CELL_WIDTH}}"
        print(row)

    for method, family, distance, values in skewed:
        seeds = ", ".join(f"{v:.4g}" for v in values)
        print(f"  ! {method} / {family} / {distance}: {seeds}"
              f"  (mean {np.mean(values):.4g} vs median {np.median(values):.4g})")
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
    print("LEGACY AGGREGATE (Eq. 6): WMF = 0.4*PF + 0.4*RD + 0.2*PIS. The")
    print("benchmark's suite is PF, RD and FT; PIS was withdrawn, never having")
    print("been implemented (D18), and its term is evaluated at zero - which is")
    print("what the previous paper's WMF numbers were computed with too. This")
    print("is printed to reproduce them, not as a headline: the share column")
    print("says why.")
    print("=" * 78)
    print_metric_table(runs, methods, families, distances, "wmf")

    print("share of |WMF| contributed by RD:")
    header = f"{'':<20}"
    for family in families:
        for distance in distances:
            col = f"{FAMILY_ABBREV.get(family, family[:5])}_{distance.split('_')[1]}"
            header += f" | {col:>{CELL_WIDTH}}"
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:<20}"
        for family in families:
            for distance in distances:
                share = rd_share(runs, method, family, distance)
                cell = "N/A" if np.isnan(share) else f"{100 * share:.1f}%"
                row += f" | {cell:>{CELL_WIDTH}}"
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
            per_seed = cell_d_trans(runs, family, distance)
            if per_seed:
                values[(family, distance)] = list(per_seed.values())
    if not values:
        return

    print("=" * 78)
    print("DYNAMIC DISTANCE d_trans (Eq. 9): KL between the transition models")
    print("of two environments, each trained on its own task from scratch.")
    print("=" * 78)
    # d_trans keeps its mean and std alongside the median: the claim it has to
    # support is whether two levels separate, and that is a statement about
    # spread, not about the centre.
    for (family, distance), vals in values.items():
        print(f"  {family:<10} {distance:<14} "
              f"d_trans = {np.median(vals):8.4f} "
              f"[{min(vals):.4f},{max(vals):.4f}]  "
              f"mean {np.mean(vals):.4f} +- {np.std(vals):.4f} "
              f"(n={len(vals)})")
    print()


def rank(values) -> np.ndarray:
    """Ranks of `values`, ties averaged. The half of Spearman that scipy would
    have supplied, written out because scipy is installed here but not declared
    in requirements.txt, and importing an undeclared dependency is F12."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(1, len(v) + 1, dtype=float)
    for value in np.unique(v):
        tied = v == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x, y) -> float:
    """Spearman's rho: Pearson's r on the ranks."""
    if len(x) != len(y):
        raise ValueError("spearman needs two series of the same length")
    if len(x) < 2:
        return float("nan")
    rx, ry = rank(x) - np.mean(rank(x)), rank(y) - np.mean(rank(y))
    denominator = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    return float("nan") if denominator == 0 else float(np.sum(rx * ry) / denominator)


def axis_value(runs, methods, family, distance, key):
    """
    One cell of the distance axis: the median over methods of each method's
    median over seeds.

    Median twice rather than mean twice, for the two reasons separately. Over
    seeds, because of the heavy tail (P12). Over methods, because the methods
    are not five samples of one quantity — they are five different ones, and a
    mean would let whichever method is furthest out set the family's number.
    """
    per_method = []
    for method in methods:
        values = list(cell_values(runs, method, family, distance, key).values())
        if values:
            per_method.append(float(np.median(values)))
    return float(np.median(per_method)) if per_method else float("nan")


def print_distance_axis_section(runs, axis_methods, families, distances,
                                key="rd"):
    """
    F27: does the labelled distance level predict forgetting, and what does?

    This is the benchmark grading its own design axis, so it prints the axis
    first (one row per family, so the reader can see the peak for themselves)
    and only then the rank correlations that summarise it.
    """
    cells = [(family, distance) for family in families for distance in distances
             if not np.isnan(axis_value(runs, axis_methods, family, distance, key))]
    if len(cells) < 2:
        return

    print("=" * 78)
    print(f"DISTANCE AXIS (F27): does the labelled level predict {key.upper()}?")
    print("Each cell is the median over methods of each method's median over")
    print("seeds. Methods aggregated: " + ", ".join(axis_methods) + ".")
    if set(axis_methods) != set(METHODS):
        left_out = [m for m in METHODS if m not in axis_methods]
        print("Left out, deliberately: " + ", ".join(left_out) + " - a method")
        print("that freezes its encoder lands two orders of magnitude away in")
        print("dmcontrol, so pooling it here would average two scales instead")
        print("of reading the axis. Its own numbers are in the tables above.")
    print("=" * 78)

    header = f"  {'family':<12}" + "".join(
        f"{d.split('_')[1]:>12}" for d in distances) + f"{'peak at':>12}"
    print(header)
    for family in families:
        values = [axis_value(runs, axis_methods, family, d, key)
                  for d in distances]
        if all(np.isnan(v) for v in values):
            continue
        peak = distances[int(np.nanargmax(values))].split("_")[1]
        row = f"  {family:<12}" + "".join(
            ("N/A" if np.isnan(v) else f"{v:.2f}").rjust(12) for v in values)
        print(row + f"{peak:>12}")
    print()

    # The predictors, over the same cells. d_trans is a property of the task
    # pair, so it is read off any run that carries it rather than per method.
    observed, labels, d_trans, difficulty = [], [], [], []
    for family, distance in cells:
        observed.append(axis_value(runs, axis_methods, family, distance, key))
        labels.append(distances.index(distance))
        per_seed = [r["d_trans"] for r in runs
                    if r.get("family") == family and r.get("distance") == distance
                    and r.get("d_trans") is not None]
        d_trans.append(float(np.median(per_seed)) if per_seed else float("nan"))
        difficulty.append(axis_value(runs, axis_methods, family, distance,
                                     "heldout_reconstruction_B_after_task_B"))

    # The same nine rows the correlations are computed from. Printed because a
    # correlation over nine points is a summary of a table small enough to
    # read, and the reader should be able to check it against the ranks.
    print(f"  {'cell':<26}{key:>10}{'d_trans':>12}{'recon B|after B':>18}")
    for (family, distance), obs, dt, diff in zip(cells, observed, d_trans,
                                                 difficulty):
        cell_name = f"{family}/{distance.split('_')[1]}"
        print(f"  {cell_name:<26}{obs:>10.2f}{dt:>12.2f}{diff:>18.2f}")
    print()

    print(f"Rank correlation (Spearman) with {key}, over the "
          f"{len(cells)} cells:")
    for name, predictor in (("labelled level (min<med<max)", labels),
                            ("d_trans (Eq. 9)", d_trans),
                            ("task-B difficulty (recon B|after B)", difficulty)):
        usable = [(o, p) for o, p in zip(observed, predictor) if not np.isnan(p)]
        if len(usable) < 2:
            continue
        rho = spearman([o for o, _ in usable], [p for _, p in usable])
        print(f"  {name:<38} {rho:+.2f}  (n={len(usable)})")
    print()
    print(f"With {len(cells)} cells and families on different scales these are")
    print("indicative, not conclusive; the repeated peak in the table above is")
    print("the robust half of the result.")
    print()


def task_a_loss(runs, method, family, distance):
    """
    How much of task A the encoder lost, as a fraction of what it had.

    Relative, not absolute: a cell of MiniGrid reconstructs task A at 0.52 and
    one of DMControl at 14.7, so +0.23 means opposite things in the two. Per
    seed first and median after, so one seed cannot set the ratio.
    """
    before = cell_values(runs, method, family, distance,
                         "heldout_reconstruction_A_after_task_A")
    after = cell_values(runs, method, family, distance,
                        "heldout_reconstruction_A_after_task_B")
    ratios = [(after[s] - before[s]) / before[s]
              for s in sorted(set(before) & set(after)) if before[s] > 0]
    return float(np.median(ratios)) if ratios else float("nan")


def control_cells(runs, methods, families, distances, threshold: float = 0.10):
    """
    Cells where no method's encoder lost task A (F22/P13).

    `threshold` is a fraction of the task-A error, and 10% is not a knife edge:
    the cells that do forget sit between +1359% and +78197%, so anything from a
    few percent to a few hundred would select the same cells. It is stated as a
    number rather than left implicit because it is the kind of choice a reader
    is entitled to move.
    """
    controls = []
    for family in families:
        for distance in distances:
            losses = [task_a_loss(runs, m, family, distance) for m in methods]
            losses = [x for x in losses if not np.isnan(x)]
            if losses and max(losses) <= threshold:
                controls.append((family, distance))
    return controls


def print_control_section(runs, methods, families, distances,
                          threshold: float = 0.10):
    """
    P13: what each cell did to the encoder's grip on task A, and which cells
    therefore had nothing to lose.

    Printed as the numbers rather than as a verdict, because the interesting
    case is the one a verdict would hide: a cell can leave the encoder intact
    and still move RD, which is F18 seen from the other side.
    """
    print("=" * 78)
    print("TASK-A LOSS PER CELL (F22/P13): median change in task-A")
    print("reconstruction after training on B, as a fraction of what the model")
    print("had. Range over methods. This is the pixel scale, so a cell can read")
    print("~0 here and still move PF and RD - that is F18, not a contradiction.")
    print("=" * 78)
    controls = control_cells(runs, methods, families, distances, threshold)
    for family in families:
        for distance in distances:
            losses = {m: task_a_loss(runs, m, family, distance)
                      for m in methods}
            losses = {m: v for m, v in losses.items() if not np.isnan(v)}
            if not losses:
                continue
            worst = max(losses, key=losses.get)
            flag = "  <- nothing lost by anyone" \
                if (family, distance) in controls else ""
            print(f"  {family:<10} {distance:<14} "
                  f"{100 * min(losses.values()):+9.1f}% .. "
                  f"{100 * losses[worst]:+11.1f}% ({worst}){flag}")
    print(f"\n  {len(controls)} of {len(families) * len(distances)} cells lose "
          f"nothing at the {100 * threshold:.0f}% threshold.")
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
                # The two cell values are medians, to agree with the tables
                # (D15). `delta` is the mean paired difference, because that is
                # the statistic the permutation test and d_z are computed on -
                # it is not the difference of the two medians printed beside it.
                print(f"  {key:<6} {method_a}={np.median(a):+8.3f}  "
                      f"{method_b}={np.median(b):+8.3f}  "
                      f"mean delta={np.mean(diffs):+8.3f}  "
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
    parser.add_argument("--axis-methods", nargs="+", default=AXIS_METHODS,
                        help="methods aggregated by the distance-axis section "
                             "(F27); the default leaves ug_mtm out and says so")
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
    # This tool warns where `export_tables.py` raises, and on purpose: the
    # numbers below are how you diagnose an inconsistent directory, so
    # refusing to print them is refusing to help. The paper's tables are the
    # opposite case and stop.
    try:
        check_runs_consistent(runs)
    except ValueError as problem:
        print(f"\nWARNING: {problem}\nEvery aggregate below mixes them — fix "
              "the directory before reading the numbers.\n")

    protocol = shared_protocol(runs)
    if protocol is not None:
        print(f"protocol: n_train={protocol['n_train']} "
              f"n_collect={protocol['n_collect']} "
              f"batch_size={protocol['batch_size']} "
              f"seq_len={protocol['seq_len']} "
              f"seeds={protocol['seeds']}")

    print("\n" + "=" * 78)
    print("FORGETTING METRICS (median [min, max] across seeds; ! = right-skewed)")
    print("PF and RD are measured on latents encoded once by the post-task-A")
    print("model, so they score the transition component M in a fixed latent")
    print("basis and are blind to drift in the encoder itself (F18). The pixel")
    print("columns further down are the scale that sees it.")
    print("=" * 78)
    for key in args.metrics:
        print_metric_table(runs, methods, families, distances, key)

    axis_methods = [m for m in args.axis_methods if m in methods]
    print_distance_axis_section(runs, axis_methods, families, distances)
    print_control_section(runs, methods, families, distances)
    print_aggregate_section(runs, methods, families, distances)
    print_distance_table(runs, families, distances)

    print("=" * 78)
    print("TASK QUALITY IN PIXELS (F17/F18): was there anything to forget, and")
    print("did the metrics above see it? Squared error per held-out frame, the")
    print("one scale that is comparable across models and across budgets.")
    print("=" * 78)
    for key, label in QUALITY_COLUMNS:
        if any(r.get(key) is not None for r in runs):
            print_metric_table(runs, methods, families, distances, key,
                               flag_skew=False)

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
