"""
Convergence probe for task A: does the model learn task A before we measure
forgetting on it?

This answers the objection that decides whether the rest of the benchmark means
anything. PF and RD measure how far M_theta_k has drifted from M_theta_i on data
from task A. If M_theta_i never fit task A, they measure the distance between
two bad models, and calling that catastrophic forgetting does not hold up.

What it does: train ONE task-A run at the largest budget requested and evaluate
at each multiple of the benchmark's own `n_train` along the way. Because batches
are drawn sequentially from a seeded stream, the state at checkpoint m is
exactly what an independent run of m * n_train steps would have reached, so this
costs one 10x run instead of 1x + 2x + 5x + 10x. Evaluations are wrapped in
preserve_rng_state, which is what makes that equivalence hold.

Three signals are recorded at each checkpoint:

  * `heldout_reconstruction` — squared error per frame, in pixel space, on
    held-out task-A frames. **This is the one to read.** It is the only signal
    comparable across budgets: it is measured on fixed pixels.
  * `nll_own` — NLL of the transition model on held-out task-A transitions,
    encoded by the model itself. This is what the benchmark's PF is built from,
    but it is not comparable across budgets on its own: the encoder produces
    the targets, so the target moves as training proceeds. A model that encoded
    every frame to the same latent would score a superb NLL — that is exactly
    the failure F0 was.
  * `nll_random_init` and `nll_gap` — NLL of an untrained model on the same
    dataset, and the gap to `nll_own`. The gap is computed exactly as the
    benchmark's FT is, and reads as "how much better than an untrained model,
    in this latent space". It is the honest way to look at NLL across budgets.
    It is not numerically the FT of the matching benchmark cell: the untrained
    reference is drawn at a different point in the random stream.

Usage:

    python experiments/convergence_A.py
    python experiments/convergence_A.py --family gymnasium --multipliers 1 2 5 10
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.benchmark.metrics import compute_nll
from src.benchmark.protocol import (
    build_latent_eval_dataset,
    collect_rollouts,
    evaluate_reconstruction,
)
from src.utils.buffer import ReplayBuffer
from src.utils.seeding import preserve_rng_state, set_seed

from experiments.run_full_benchmark import (
    FAMILY_CONFIGS,
    create_env_pair,
    create_model,
    resolve_protocol,
)


def measure(model, model_rand, buf_heldout, device, protocol, seed):
    """
    Evaluate task-A fit without disturbing the training run.

    Everything here runs inside preserve_rng_state: UG-MTM draws MC-dropout
    masks even in eval mode, and the whole point of the probe is that
    checkpoint m equals an independent run of m steps.
    """
    with preserve_rng_state():
        was_training = model.training
        eval_ds = build_latent_eval_dataset(
            model, buf_heldout, device,
            n_transitions=protocol["n_eval_transitions"], seed=seed,
        )
        nll_own = compute_nll(model, eval_ds, device)
        nll_random = compute_nll(model_rand, eval_ds, device)
        recon = evaluate_reconstruction(
            model, buf_heldout, device,
            n_frames=protocol["n_recon_frames"], seed=seed,
        )
        model.train(was_training)

    return {
        "heldout_reconstruction": recon,
        "nll_own": nll_own,
        "nll_random_init": nll_random,
        "nll_gap": nll_random - nll_own,
    }


def run_probe(family, distance, method, seed, multipliers, protocol, device):
    cfg = OmegaConf.load(FAMILY_CONFIGS[family])
    seq_cfg = cfg.benchmark.sequences[distance]
    env_A, env_B = create_env_pair(family, seq_cfg)

    try:
        # Same seeding as run_cell, so the 1x checkpoint reproduces the
        # benchmark's own task-A training for this cell.
        set_seed(seed)
        env_A.seed(seed)
        env_B.seed(seed + 1)

        action_dim = env_A.action_dim
        model = create_model(method, action_dim, protocol).to(device)

        buf_A = ReplayBuffer(max_episodes=protocol["n_collect"],
                             seq_len=protocol["seq_len"])
        buf_B = ReplayBuffer(max_episodes=protocol["n_collect"],
                             seq_len=protocol["seq_len"])
        buf_A_heldout = ReplayBuffer(max_episodes=protocol["n_eval_episodes"],
                                     seq_len=2)
        collect_rollouts(env_A, buf_A, n_rollouts=protocol["n_collect"])
        # Task B rollouts are collected but never trained on: they advance the
        # RNG exactly as run_cell does, which is what keeps the 1x checkpoint
        # comparable with the benchmark.
        collect_rollouts(env_B, buf_B, n_rollouts=protocol["n_collect"])
        collect_rollouts(env_A, buf_A_heldout,
                         n_rollouts=protocol["n_eval_episodes"])

        opt = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
        base_steps = protocol["n_train"]
        checkpoints = sorted({int(m * base_steps) for m in multipliers})

        # The untrained reference model is built once, before training, so every
        # checkpoint is compared against the same weights.
        with preserve_rng_state():
            model_rand = create_model(method, action_dim, protocol).to(device)

        records = []
        model.train()
        n_nan = 0
        recon_sum = 0.0
        recon_n = 0
        step = 0

        for target in checkpoints:
            while step < target:
                step += 1
                batch = buf_A.sample(batch_size=protocol["batch_size"],
                                     seq_len=protocol["seq_len"])
                obs = batch["obs"].permute(0, 1, 4, 2, 3).to(device)
                actions = batch["actions"].to(device)
                loss, comps = model.compute_loss({"obs": obs, "actions": actions})
                if torch.isnan(loss):
                    n_nan += 1
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                recon_sum += comps["reconstruction"]
                recon_n += 1

            record = {
                "steps": step,
                "multiplier": step / base_steps,
                # Mean over the steps since the previous checkpoint, not over
                # the whole run: a run-long mean hides a plateau.
                "train_reconstruction_interval_mean": recon_sum / max(recon_n, 1),
                "n_nan_steps_cumulative": n_nan,
            }
            record.update(measure(model, model_rand, buf_A_heldout, device,
                                  protocol, seed))
            records.append(record)
            recon_sum, recon_n = 0.0, 0

            print(f"  steps={record['steps']:>7} "
                  f"({record['multiplier']:g}x) | "
                  f"held-out recon={record['heldout_reconstruction']:10.3f} | "
                  f"train recon={record['train_reconstruction_interval_mean']:10.3f} | "
                  f"NLL(own)={record['nll_own']:9.3f} | "
                  f"gap vs random={record['nll_gap']:9.3f}")

        return records
    finally:
        env_A.close()
        env_B.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--family", default="minigrid", choices=list(FAMILY_CONFIGS))
    parser.add_argument("--distance", default="distance_med",
                        choices=["distance_min", "distance_med", "distance_max"])
    parser.add_argument("--method", default="finetuning",
                        help="which model to probe; the plain RSSM baseline by "
                             "default, since task-A convergence is a property "
                             "of the architecture and not of the mitigation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multipliers", type=float, nargs="+",
                        default=[1, 2, 5, 10],
                        help="multiples of the config's n_train to evaluate at")
    parser.add_argument("--results-dir", default="results/convergence", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(FAMILY_CONFIGS[args.family])
    protocol = resolve_protocol(cfg)
    base_steps = protocol["n_train"]
    largest = max(args.multipliers)

    print(f"Device: {device}")
    print(f"Probe: {args.method} | {args.family} | {args.distance} | "
          f"seed={args.seed}")
    print(f"Base budget from {FAMILY_CONFIGS[args.family]}: "
          f"n_train={base_steps}, n_collect={protocol['n_collect']}, "
          f"batch_size={protocol['batch_size']}, seq_len={protocol['seq_len']}")
    print(f"Checkpoints at {args.multipliers} x n_train "
          f"(one training run of {int(largest * base_steps)} steps)\n")

    records = run_probe(args.family, args.distance, args.method, args.seed,
                        args.multipliers, protocol, device)

    out = args.results_dir / (
        f"{args.method}_{args.family}_{args.distance}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": args.method,
        "family": args.family,
        "distance": args.distance,
        "seed": args.seed,
        "multipliers": list(args.multipliers),
        "protocol": protocol,
        "checkpoints": records,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out}")

    first, last = records[0], records[-1]
    start, end = first["heldout_reconstruction"], last["heldout_reconstruction"]
    change = (end - start) / start if start else 0.0
    print(f"Held-out reconstruction from {first['multiplier']:g}x to "
          f"{last['multiplier']:g}x: {start:.3f} -> {end:.3f} "
          f"({change:+.1%}; negative means still improving)")
    print("Read the held-out reconstruction column: it is the only one "
          "comparable across budgets.")


if __name__ == "__main__":
    main()
