"""
Aggregate the k-task sequence runs produced by `experiments/run_sequence.py`.

Same rule as `summarize_results.py`: this is where the numbers come from, and
nothing downstream is allowed to retype them. The sequence results have their
own schema -- a retention matrix per run rather than one cell's metrics -- so
they get their own loader and their own aggregation, and share the reporting
policy (medians over seeds) with the paired grid.

    python experiments/summarize_sequence.py

The question these runs answer is the one the paired grid cannot: whether
forgetting of the first task accumulates as the sequence goes on.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_full_benchmark import METHODS, protocol_identity


def load_sequence_runs(results_root: Path) -> list:
    """Every metrics.json under `results_root`, as a list of dicts."""
    runs = []
    for path in sorted(Path(results_root).glob("*/*/metrics.json")):
        with open(path) as f:
            run = json.load(f)
        run.setdefault("_path", str(path))
        runs.append(run)
    return runs


def check_sequence_runs(runs: list) -> None:
    """Refuse runs that cannot be aggregated: two budgets, or two sequences."""
    if not runs:
        return
    protocols = {json.dumps(protocol_identity(r.get("protocol")), sort_keys=True)
                 for r in runs}
    if len(protocols) > 1:
        raise ValueError(
            f"{len(protocols)} different protocols across {len(runs)} sequence "
            "runs; they cannot be aggregated."
        )
    sequences = {json.dumps(r.get("tasks"), sort_keys=True) for r in runs}
    if len(sequences) > 1:
        raise ValueError(
            f"{len(sequences)} different task sequences in one directory; the "
            "sequence was edited and stale results were left behind."
        )


def retention(run: dict, task: int, stage: int, key: str):
    """`key` of `task` as measured at `stage`, or None if not measured there."""
    if stage >= len(run.get("stages", [])):
        return None
    for entry in run["stages"][stage]["retention"]:
        if entry["task"] == task:
            return entry[key]
    return None


def retention_curve(runs: list, method: str, task: int, key: str) -> dict:
    """{stage: [value per seed]} for one task, over the stages that measure it.

    The diagonal (stage == task) is excluded: it is zero by construction for
    pf and rd, and including it would put a structural zero in a curve of
    measured values.
    """
    curve = {}
    method_runs = [r for r in runs if r.get("method") == method]
    for run in method_runs:
        for stage in range(task + 1, run.get("k", 0)):
            value = retention(run, task, stage, key)
            if value is not None:
                curve.setdefault(stage, []).append(value)
    return curve


def encoder_factor(runs: list, method: str, task: int, stage: int) -> list:
    """Held-out reconstruction of `task` at `stage`, over what it was when
    `task` finished. Same quantity the paired grid's encoder table reports."""
    factors = []
    for run in runs:
        if run.get("method") != method:
            continue
        base = retention(run, task, task, "heldout_reconstruction")
        now = retention(run, task, stage, "heldout_reconstruction")
        if base and now is not None:
            factors.append(now / base)
    return factors


def peak_stage(runs: list, method: str, task: int, key: str = "rd"):
    """
    Where the curve peaks, and in how many seeds it peaks there.

    Reported per seed rather than off the median curve: a peak that only exists
    after averaging is a property of the average.
    """
    curve = retention_curve(runs, method, task, key)
    if not curve:
        return None, 0, 0
    stages = sorted(curve)
    medians = [float(np.median(curve[s])) for s in stages]
    peak = stages[int(np.argmax(medians))]

    n_seeds = min(len(curve[s]) for s in stages)
    agreeing = 0
    for seed_index in range(n_seeds):
        per_seed = [curve[s][seed_index] for s in stages]
        if stages[int(np.argmax(per_seed))] == peak:
            agreeing += 1
    return peak, agreeing, n_seeds


def task_difficulty(runs: list, task: int) -> list:
    """How hard each task was to fit, measured the moment it finished. Pooled
    over methods: at that point every method has just trained on it."""
    values = [retention(r, task, task, "heldout_reconstruction") for r in runs]
    return [v for v in values if v is not None]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default=Path("results-seq"), type=Path)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--task", type=int, default=0,
                        help="which task's retention to report (0 = the first)")
    args = parser.parse_args(argv)

    runs = load_sequence_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no metrics.json found under {args.results_dir}")
    check_sequence_runs(runs)
    methods = [m for m in args.methods if any(r.get("method") == m for r in runs)]

    k = runs[0]["k"]
    tasks = runs[0]["tasks"]
    print(f"{len(runs)} sequence run(s) from {args.results_dir}, k={k}")
    for i, task in enumerate(tasks):
        difficulty = task_difficulty(runs, i)
        name = task.get("env_id") or f"{task.get('domain_name')}/{task.get('task_name')}"
        print(f"  T{i + 1}  {name:<32} "
              f"difficulty {np.median(difficulty):8.2f}" if difficulty else name)

    nan = sum(s["n_nan_steps"] for r in runs for s in r["stages"])
    print(f"NaN steps across every stage of every run: {nan}")

    stages = list(range(args.task + 1, k))
    print("\n" + "=" * 78)
    print(f"RETENTION OF T{args.task + 1}: rollout divergence from the model as it")
    print("stood when that task finished, median over seeds.")
    print("=" * 78)
    header = "".join(f"  after T{s + 1}" for s in stages)
    print(f"  {'method':<18}{header}   peak")
    for method in methods:
        curve = retention_curve(runs, method, args.task, "rd")
        if not curve:
            continue
        row = "".join(f"{np.median(curve[s]):10.1f}" if s in curve else "        --"
                      for s in stages)
        peak, agreeing, n_seeds = peak_stage(runs, method, args.task)
        print(f"  {method:<18}{row}   T{peak + 1} ({agreeing}/{n_seeds} seeds)")

    print("\n" + "=" * 78)
    print(f"HELD-OUT RECONSTRUCTION OF T{args.task + 1}, as a factor of what the")
    print("model had when that task finished. The pixel scale, which PF and RD")
    print("cannot see (F18).")
    print("=" * 78)
    print(f"  {'method':<18}{header}")
    for method in methods:
        row = ""
        for stage in stages:
            factors = encoder_factor(runs, method, args.task, stage)
            row += f"{np.median(factors):10.2f}" if factors else "        --"
        print(f"  {method:<18}{row}")


if __name__ == "__main__":
    main()
