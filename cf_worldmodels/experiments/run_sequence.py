"""
The benchmark over a sequence of k tasks instead of a pair.

The paper states its formalism for `T_1 ... T_k` and its metrics as `PF(i, k)`,
but the paired runner executes one task switch. This runs the sequence and
produces the full forgetting matrix: for every task `i` and every later stage
`k`, how much the model has lost of `T_i` by the time it has finished `T_k`.

It is a separate entry point rather than a generalisation of
`run_full_benchmark.py`, and that is deliberate. The paired runner produced the
375 results the paper reports; rewriting it to carry an index where it now says
`task_A` would put every one of those numbers at risk to answer a question
asked on the side. This shares the models, the training loop, the metrics and
the protocol loader with it, and shares no state.

    python experiments/run_sequence.py

Writes results-seq/<method>/<family>_seq_<seed>/metrics.json. Resumable: a seed
whose file exists is skipped.

Definitions follow the paired runner exactly. `D_i` is built once, from held-out
episodes of `T_i`, encoded by the model as it stood immediately after training
on `T_i` (D9's declared scope: PF and RD score the transition component in a
fixed latent basis). The reference NLL and the model snapshot taken at that
moment are what every later stage is compared against.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from experiments.run_full_benchmark import (
    METHODS,
    create_model,
    resolve_protocol,
    train_task,
)
from src.benchmark.metrics import compute_nll, compute_pf, compute_rd
from src.benchmark.protocol import (
    build_latent_eval_dataset,
    collect_rollouts,
    evaluate_reconstruction,
)
from src.envs.minigrid_env import MiniGridEnv
from src.utils.buffer import ReplayBuffer
from src.utils.logging_utils import save_metrics
from src.utils.seeding import preserve_rng_state, set_seed

DEFAULT_CONFIG = "configs/benchmark/minigrid_sequence.yaml"


def make_env(family, spec):
    if family != "minigrid":
        raise ValueError(
            f"sequence runs are implemented for minigrid only, got {family!r}. "
            "The other two families need a physics-parameterised constructor "
            "here; see create_env_pair in run_full_benchmark.py."
        )
    return MiniGridEnv(spec.env_id)


def collect_task(env, protocol, seed):
    """Training and held-out buffers for one task, from disjoint rollouts."""
    train_buf = ReplayBuffer(max_episodes=protocol["n_collect"],
                             seq_len=protocol["seq_len"])
    heldout_buf = ReplayBuffer(max_episodes=protocol["n_eval_episodes"],
                               seq_len=protocol["seq_len"])
    env.seed(seed)
    collect_rollouts(env, train_buf, protocol["n_collect"])
    collect_rollouts(env, heldout_buf, protocol["n_eval_episodes"])
    return train_buf, heldout_buf


def advance_method(model, method, snapshot, buffers, heldouts, device, seed,
                   protocol, finished):
    """
    The method's task-switching step, for a sequence.

    Same operations the paired runner applies at its single boundary, applied
    at every boundary. All four accumulate by construction: EWC appends a
    Fisher diagonal per task and sums the penalties, progressive nets add a
    column, replay accumulates episodes, and UG-MTM activates one more expert.
    `finished` is the index of the task just completed.
    """
    if method == "ewc":
        fisher_ds = build_latent_eval_dataset(
            snapshot, heldouts[finished], device,
            n_transitions=protocol["n_fisher_transitions"],
            seed=seed + 10_000 + finished,
        )
        model.consolidate(fisher_ds, device)
    elif method == "progressive_nets":
        model.add_column()
    elif method == "replay_infinite":
        for task_id in range(finished + 1):
            model.add_task_data(task_id=task_id,
                                episodes=buffers[task_id].episodes)
    elif method == "ug_mtm":
        if model.K_active < model.K - 1:
            model.K_active += 1
        for name, param in model.named_parameters():
            if f"experts.{model.K_active}" in name:
                param.requires_grad_(True)
            elif "uncertainty_head" in name or "threshold_net" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)


def training_buffer(method, buffers, upto, protocol):
    """What the model trains on at stage `upto`: everything so far, for replay,
    and the current task alone for everyone else."""
    if method != "replay_infinite":
        return buffers[upto]
    combined = ReplayBuffer(max_episodes=protocol["n_collect"] * (upto + 1),
                            seq_len=protocol["seq_len"])
    for task_id in range(upto + 1):
        for episode in buffers[task_id].episodes:
            combined.add_episode(episode)
    return combined


def run_sequence(method, family, tasks, buffers, heldouts, action_dim, device,
                 seed, protocol):
    """Train through the whole sequence, measuring every earlier task at every
    stage. Returns the metrics dict written to disk."""
    set_seed(seed)
    model = create_model(method, action_dim, protocol).to(device)

    k = len(tasks)
    eval_sets = {}      # i -> D_i, encoded once by the post-T_i model
    snapshots = {}      # i -> the post-T_i model, for PF's and RD's reference
    baseline_nll = {}   # i -> NLL(M_i, D_i)
    stages = []

    for i in range(k):
        if i > 0:
            advance_method(model, method, snapshots[i - 1], buffers, heldouts,
                           device, seed, protocol, finished=i - 1)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=protocol["learning_rate"],
        )
        stats = train_task(model, training_buffer(method, buffers, i, protocol),
                           optimizer, device, protocol)

        # Everything below only measures, so it runs with the RNG stream
        # restored afterwards: UG-MTM's gate keeps MC-dropout live at
        # evaluation time by design, and an extra evaluation would otherwise
        # move every later training step.
        with preserve_rng_state():
            snapshot = copy.deepcopy(model).to(device)
            snapshot.eval()
            snapshots[i] = snapshot
            eval_sets[i] = build_latent_eval_dataset(
                snapshot, heldouts[i], device,
                n_transitions=protocol["n_eval_transitions"], seed=seed + i,
            )
            baseline_nll[i] = compute_nll(snapshot, eval_sets[i], device)

            stage = {
                "stage": i,
                "task": tasks[i],
                "n_nan_steps": stats["n_nan_steps"],
                "n_update_steps": stats["n_update_steps"],
                "final_reconstruction_loss": stats["final_reconstruction_loss"],
                "recon_curve": stats["recon_curve"],
                "retention": [],
            }
            # Every task seen so far, scored by the model as it stands now.
            for earlier in range(i + 1):
                stage["retention"].append({
                    "task": earlier,
                    "pf": compute_pf(snapshots[earlier], model,
                                     eval_sets[earlier], device),
                    "rd": compute_rd(snapshots[earlier], model,
                                     eval_sets[earlier],
                                     horizon=protocol["rd_horizon"],
                                     n_samples=protocol["rd_samples"]),
                    "heldout_reconstruction": evaluate_reconstruction(
                        model, heldouts[earlier], device,
                        n_frames=protocol["n_recon_frames"], seed=seed + earlier,
                    ),
                })
            stages.append(stage)

    return {
        "method": method,
        "family": family,
        "seed": seed,
        "k": k,
        "tasks": tasks,
        "stages": stages,
        # PF(i, i) is zero by construction and RD(i, i) is zero likewise; the
        # diagonal is stored anyway so a reader can check that rather than
        # trust it.
        "baseline_nll": baseline_nll,
        "protocol": protocol,
    }


def result_path(results_root, family, seed, method):
    return (results_root / method / f"{family}_seq_{seed}" / "metrics.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--results-dir", default=Path("results-seq"), type=Path)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, dest="n_train")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = OmegaConf.load(args.config)
    protocol = resolve_protocol(cfg, {"n_train": args.n_train,
                                      "seeds": args.seeds})
    family = cfg.benchmark.family
    task_specs = list(cfg.benchmark.tasks)
    tasks = [OmegaConf.to_container(t, resolve=True) for t in task_specs]
    declared = int(cfg.model.action_dim)

    print(f"=== sequence: {family}, k={len(tasks)} ===")
    for i, task in enumerate(tasks):
        print(f"  T{i + 1}  {task.get('env_id')}")
    print(f"protocol: n_train={protocol['n_train']} "
          f"seeds={protocol['seeds']} methods={args.methods}")

    # One model spans the whole sequence, so its GRU has a single action width.
    # Checked before training rather than after (F25).
    for i, spec in enumerate(task_specs):
        env = make_env(family, spec)
        try:
            if env.action_dim != declared:
                raise SystemExit(
                    f"T{i + 1} ({spec.env_id}) exposes action_dim="
                    f"{env.action_dim}, config declares {declared}. One model "
                    "spans the sequence and cannot have two action widths."
                )
        finally:
            env.close()

    if args.dry_run:
        planned = sum(
            1 for seed in protocol["seeds"] for method in args.methods
            if not result_path(args.results_dir, family, seed, method).exists()
        )
        print(f"{planned} run(s) planned, "
              f"{len(protocol['seeds']) * len(args.methods) - planned} cached.")
        print("--dry-run: nothing was trained.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in protocol["seeds"]:
        pending = [m for m in args.methods
                   if not result_path(args.results_dir, family, seed,
                                      m).exists()]
        if not pending:
            print(f"\n=== seed {seed}: all cached ===")
            continue

        # Collected once and shared by every method of this seed, exactly as
        # the paired runner shares its rollouts across the five cells.
        print(f"\n=== seed {seed}: collecting {len(tasks)} task(s) ===")
        set_seed(seed)
        buffers, heldouts = [], []
        for i, spec in enumerate(task_specs):
            env = make_env(family, spec)
            try:
                train_buf, heldout_buf = collect_task(env, protocol, seed + i)
            finally:
                env.close()
            buffers.append(train_buf)
            heldouts.append(heldout_buf)

        for method in pending:
            metrics = run_sequence(method, family, tasks, buffers, heldouts,
                                   declared, device, seed, protocol)
            path = result_path(args.results_dir, family, seed, method)
            save_metrics(metrics, path)
            final = metrics["stages"][-1]["retention"]
            summary = "  ".join(
                f"T{r['task'] + 1}: pf={r['pf']:+.2f} rd={r['rd']:.1f}"
                for r in final if r["task"] < len(tasks) - 1
            )
            print(f"  {method:<17} after T{len(tasks)} | {summary}")


if __name__ == "__main__":
    main()
