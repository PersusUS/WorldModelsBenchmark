"""
Run full benchmark for a method across seeds.

Usage:
    python experiments/run_benchmark.py \
        --method finetuning \
        --config configs/benchmark/minigrid.yaml \
        --distance distance_min \
        --seeds 0 1 2 3 4 \
        --no_wandb
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["finetuning", "replay_infinite", "ewc",
                                 "progressive_nets", "ug_mtm"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--distance", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    script = "experiments/train_baseline.py"
    if args.method == "ug_mtm":
        script = "experiments/train_ug_mtm.py"

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running {args.method} | distance={args.distance} | seed={seed}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, script,
            "--method", args.method,
            "--config", args.config,
            "--distance", args.distance,
            "--seed", str(seed),
            "--steps", str(args.steps),
        ]
        if args.no_wandb:
            cmd.append("--no_wandb")

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"WARNING: seed {seed} failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
