"""
Run the full benchmark: 5 methods x 3 families x 3 distance levels x N seeds.

The training protocol is not defined in this file. Every value comes from the
`protocol:` block of `configs/benchmark/<family>.yaml`, and each metrics.json
records the protocol it was produced with. Two consequences worth knowing:

  * Table 1 of the paper must be generated from the results, never written by
    hand. Hardcoded constants here were the origin of three different sets of
    numbers circulating for the same experiment.
  * The runner refuses to reuse cached results that were produced under a
    different protocol, so one results directory can never mix budgets.

Command-line flags override individual protocol fields; the effective protocol
is printed before anything trains.

Runs whose metrics.json already exists are skipped, so it is safe to interrupt
and resume.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.envs.minigrid_env import MiniGridEnv
from src.envs.gymnasium_env import GymnasiumEnv
from src.envs.dmcontrol_env import DMControlEnv
from src.models.ug_mtm import UG_MTM
from src.baselines.finetuning import FineTuningWorldModel
from src.baselines.replay import InfiniteReplayWorldModel
from src.baselines.ewc import EWCWorldModel
from src.baselines.progressive_nets import ProgressiveNetWorldModel
from src.utils.buffer import ReplayBuffer
from src.benchmark.protocol import (
    build_latent_eval_dataset,
    collect_rollouts,
    evaluate_reconstruction,
)
from src.benchmark.metrics import compute_pf, compute_rd, compute_wmf, compute_ft, compute_nll
from src.benchmark.distances import compute_d_param
from src.utils.logging_utils import save_metrics
from src.utils.seeding import preserve_rng_state, set_seed

FAMILY_CONFIGS = {
    "minigrid": "configs/benchmark/minigrid.yaml",
    "gymnasium": "configs/benchmark/gymnasium.yaml",
    "dmcontrol": "configs/benchmark/dmcontrol.yaml",
}
METHODS = ["finetuning", "replay_infinite", "ewc", "progressive_nets", "ug_mtm"]
DISTANCES = ["distance_min", "distance_med", "distance_max"]
# Abbreviations for table headers. Truncating "minigrid" to "min" would produce
# a column headed "min_med", which reads as "minimum median".
FAMILY_ABBREV = {"minigrid": "mgrid", "gymnasium": "gym", "dmcontrol": "dmc"}

UG_MTM_CONFIG = "configs/models/ug_mtm.yaml"

# Every field the runner reads out of the config's `protocol:` block, with the
# type it is coerced to. A missing field is an error rather than a default:
# a silent default here is exactly how the protocol drifted away from the
# configs in the first place.
PROTOCOL_FIELDS = {
    "n_collect": int,
    "n_train": int,
    "batch_size": int,
    "seq_len": int,
    "learning_rate": float,
    "curve_points": int,
    "n_eval_episodes": int,
    "n_eval_transitions": int,
    "n_fisher_transitions": int,
    "n_recon_frames": int,
    "rd_horizon": int,
    "rd_samples": int,
    "ewc_lambda": float,
    "mc_dropout_T_train": int,
}
MODEL_FIELDS = {"latent_dim": int, "hidden_dim": int, "beta_kl": float}


def resolve_protocol(cfg, overrides=None) -> dict:
    """
    Read the protocol block of a family config, apply overrides, validate.

    Returns a plain JSON-serializable dict; it is stored verbatim in every
    metrics.json the run produces.
    """
    if "protocol" not in cfg:
        raise KeyError("config has no `protocol:` block")

    block = cfg.protocol
    missing = [k for k in PROTOCOL_FIELDS if k not in block]
    if missing:
        raise KeyError(
            f"config protocol block is missing: {', '.join(sorted(missing))}"
        )

    protocol = {k: caster(block[k]) for k, caster in PROTOCOL_FIELDS.items()}
    protocol["seeds"] = [int(s) for s in block.seeds]
    protocol["wmf_weights"] = {
        "alpha": float(block.wmf_weights.alpha),
        "beta": float(block.wmf_weights.beta),
        "gamma": float(block.wmf_weights.gamma),
    }
    for key, caster in MODEL_FIELDS.items():
        protocol[key] = caster(cfg.model[key])

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key not in protocol:
            raise KeyError(f"cannot override unknown protocol field: {key}")
        protocol[key] = value

    positive = [k for k in ("n_collect", "n_train", "batch_size", "seq_len",
                            "n_eval_episodes", "n_eval_transitions",
                            "n_fisher_transitions", "n_recon_frames",
                            "rd_horizon", "rd_samples", "latent_dim",
                            "hidden_dim")
                if protocol[k] <= 0]
    if positive:
        raise ValueError(
            f"protocol fields must be positive: {', '.join(sorted(positive))}"
        )
    if not protocol["seeds"]:
        raise ValueError("protocol declares no seeds")
    weight_sum = sum(protocol["wmf_weights"].values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"wmf_weights must sum to 1.0, got {weight_sum:.6f}")

    return protocol


def format_protocol(protocol: dict) -> str:
    """One line per field, for the banner printed before training."""
    lines = []
    for key, value in protocol.items():
        lines.append(f"    {key:<22} {value}")
    return "\n".join(lines)


def create_env_pair(family, seq_cfg):
    """Create (env_A, env_B) for a given family and sequence config."""
    if family == "minigrid":
        env_A = MiniGridEnv(seq_cfg.task_A.env_id)
        env_B = MiniGridEnv(seq_cfg.task_B.env_id)
    elif family == "gymnasium":
        params_A = dict(seq_cfg.task_A.get("params", {}))
        params_B = dict(seq_cfg.task_B.get("params", {}))
        env_A = GymnasiumEnv(seq_cfg.task_A.env_id, physics_params=params_A)
        env_B = GymnasiumEnv(seq_cfg.task_B.env_id, physics_params=params_B)
    elif family == "dmcontrol":
        env_A = DMControlEnv(seq_cfg.task_A.domain_name, seq_cfg.task_A.task_name)
        env_B = DMControlEnv(seq_cfg.task_B.domain_name, seq_cfg.task_B.task_name)
    return env_A, env_B


def create_model(method, action_dim, protocol):
    """
    Build a model at the capacity the protocol declares.

    UG-MTM reads its architecture from its own config file, but latent size,
    hidden size, beta_kl and the MC-dropout budget are overridden from the
    protocol so that method and baselines are compared at matched capacity by
    construction rather than by coincidence.
    """
    latent_dim = protocol["latent_dim"]
    hidden_dim = protocol["hidden_dim"]
    beta_kl = protocol["beta_kl"]

    if method == "finetuning":
        return FineTuningWorldModel(latent_dim, hidden_dim, action_dim, beta_kl=beta_kl)
    if method == "replay_infinite":
        return InfiniteReplayWorldModel(latent_dim, hidden_dim, action_dim, beta_kl=beta_kl)
    if method == "ewc":
        return EWCWorldModel(latent_dim, hidden_dim, action_dim,
                             ewc_lambda=protocol["ewc_lambda"], beta_kl=beta_kl)
    if method == "progressive_nets":
        return ProgressiveNetWorldModel(latent_dim, hidden_dim, action_dim, beta_kl=beta_kl)
    if method == "ug_mtm":
        cfg = OmegaConf.load(UG_MTM_CONFIG)
        cfg.model.action_dim = int(action_dim)
        cfg.model.latent_dim = latent_dim
        cfg.model.hidden_dim = hidden_dim
        cfg.model.beta_kl = beta_kl
        cfg.model.mc_dropout_T = protocol["mc_dropout_T_train"]
        return UG_MTM(cfg)
    raise ValueError(f"unknown method: {method}")


def train_task(model, buffer, optimizer, device, protocol, steps=None):
    """
    Train `model` on `buffer` for `steps` gradient updates.

    Returns a dict with the reconstruction loss at the first and last accepted
    step, a subsampled curve of it, and how many steps were dropped for
    producing a NaN loss. Those NaNs used to be skipped in silence, which made
    a run where most steps diverged look exactly like a healthy one.
    """
    steps = protocol["n_train"] if steps is None else steps
    curve_points = protocol["curve_points"]
    every = max(1, steps // curve_points) if curve_points > 0 else 0

    model.train()
    curve = []
    initial_loss = None
    final_loss = None
    n_nan = 0
    n_updates = 0

    for step in range(1, steps + 1):
        batch = buffer.sample(batch_size=protocol["batch_size"],
                              seq_len=protocol["seq_len"])
        obs = batch["obs"].permute(0, 1, 4, 2, 3).to(device)
        actions = batch["actions"].to(device)
        loss, comps = model.compute_loss({"obs": obs, "actions": actions})
        if torch.isnan(loss):
            n_nan += 1
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_updates += 1
        recon = comps["reconstruction"]
        if initial_loss is None:
            initial_loss = recon
        final_loss = recon
        if every and (step % every == 0 or step == steps):
            curve.append([step, recon])

    return {
        "initial_reconstruction_loss": initial_loss,
        "final_reconstruction_loss": final_loss,
        "recon_curve": curve,
        "n_nan_steps": n_nan,
        "n_update_steps": n_updates,
    }


def switch_task(model, method, model_i, buf_A, buf_B, buf_A_heldout,
                device, seed, protocol):
    """Apply the method's task-switching step between task A and task B."""
    if method == "ewc":
        fisher_ds = build_latent_eval_dataset(
            model_i, buf_A_heldout, device,
            n_transitions=protocol["n_fisher_transitions"], seed=seed + 10_000,
        )
        model.consolidate(fisher_ds, device)
    elif method == "progressive_nets":
        model.add_column()
    elif method == "replay_infinite":
        model.add_task_data(task_id=0, episodes=buf_A.episodes)
        model.add_task_data(task_id=1, episodes=buf_B.episodes)
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


def run_cell(method, env_A, env_B, device, seed, family, distance, protocol,
             results_root):
    """Train one (method, family, distance, seed) cell and write its metrics."""
    # Seeding the global RNGs is not enough: the environments own RNGs that no
    # global seed reaches, and cuDNN needs to be put in deterministic mode.
    # See src/utils/seeding.py. The two environments are seeded once here, not
    # per episode, so successive resets walk a deterministic stream of distinct
    # initial states — that is also what keeps the held-out task-A episodes
    # different from the ones used for training.
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
    collect_rollouts(env_B, buf_B, n_rollouts=protocol["n_collect"])
    collect_rollouts(env_A, buf_A_heldout, n_rollouts=protocol["n_eval_episodes"])

    # Train Task A
    opt = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    task_A = train_task(model, buf_A, opt, device, protocol)
    model_i_state = {k: v.clone() for k, v in model.state_dict().items()}

    # D_i: held-out task-A transitions, encoded once by the post-task-A model
    # so that model_i and model_k are scored on identical inputs and targets.
    model_i = create_model(method, action_dim, protocol).to(device)
    model_i.load_state_dict(model_i_state)
    eval_ds = build_latent_eval_dataset(
        model_i, buf_A_heldout, device,
        n_transitions=protocol["n_eval_transitions"], seed=seed,
    )

    # How well task A was actually learned, in pixel space, on frames that were
    # never trained on. A forgetting benchmark has to show there was something
    # to forget: if model_i never learned task A, PF and RD are the distance
    # between two bad models. Measured under preserve_rng_state so that
    # instrumenting the run cannot change its results.
    with preserve_rng_state():
        recon_A_after_task_A = evaluate_reconstruction(
            model_i, buf_A_heldout, device,
            n_frames=protocol["n_recon_frames"], seed=seed,
        )

    switch_task(model, method, model_i, buf_A, buf_B, buf_A_heldout,
                device, seed, protocol)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=protocol["learning_rate"])

    # Train Task B
    if method == "replay_infinite":
        combined = ReplayBuffer(max_episodes=protocol["n_collect"] * 2,
                                seq_len=protocol["seq_len"])
        for ep in buf_A.episodes:
            combined.add_episode(ep)
        for ep in buf_B.episodes:
            combined.add_episode(ep)
        task_B = train_task(model, combined, opt, device, protocol)
    else:
        task_B = train_task(model, buf_B, opt, device, protocol)

    # Compute metrics on the held-out task-A dataset built above.
    #
    # The order of the five calls below is load-bearing. For UG-MTM the gate
    # keeps MC-dropout active at evaluation time by design, so every NLL
    # evaluation draws dropout masks; inserting or reordering a call changes the
    # random stream that compute_rd and the random reference model draw from,
    # and the run stops matching. Anything added here goes inside
    # preserve_rng_state.
    pf = compute_pf(model_i, model, eval_ds, device)
    rd = compute_rd(model_i, model, eval_ds,
                    horizon=protocol["rd_horizon"],
                    n_samples=protocol["rd_samples"])
    weights = protocol["wmf_weights"]
    wmf = compute_wmf([pf], [rd], [0.0], alpha=weights["alpha"],
                      beta=weights["beta"], gamma=weights["gamma"])

    model_rand = create_model(method, action_dim, protocol).to(device)
    # model_i is the frozen post-task-A snapshot, so its NLL on D_A is the
    # task-A fit regardless of when it is evaluated.
    nll_A_after_task_A = compute_nll(model_i, eval_ds, device)
    nll_A_random_init = compute_nll(model_rand, eval_ds, device)
    ft = compute_ft(nll_A_after_task_A, nll_A_random_init)

    with preserve_rng_state():
        nll_A_after_task_B = compute_nll(model, eval_ds, device)
        recon_A_after_task_B = evaluate_reconstruction(
            model, buf_A_heldout, device,
            n_frames=protocol["n_recon_frames"], seed=seed,
        )

    metrics = {
        # Forgetting metrics (Section 3.2). PIS is still unimplemented.
        "wmf": wmf,
        "pf": pf,
        "rd": rd,
        "pis": 0.0,
        "ft": ft,
        # The three NLLs on the held-out task-A set that PF and FT are built
        # from, stored so both are decomposable after the fact. For UG-MTM the
        # evaluation-time gate is stochastic, so pf differs from
        # nll_A_after_task_B - nll_A_after_task_A by the MC-dropout noise of two
        # independent evaluations; for every other method they agree exactly.
        "nll_A_after_task_A": nll_A_after_task_A,
        "nll_A_after_task_B": nll_A_after_task_B,
        "nll_A_random_init": nll_A_random_init,
        # Task-A quality, reported alongside the forgetting metrics rather than
        # instead of them. The reconstruction figures are in pixel space and so
        # are comparable across training budgets; the NLLs above are not.
        "initial_reconstruction_loss_A": task_A["initial_reconstruction_loss"],
        "final_reconstruction_loss_A": task_A["final_reconstruction_loss"],
        "heldout_reconstruction_A_after_task_A": recon_A_after_task_A,
        "heldout_reconstruction_A_after_task_B": recon_A_after_task_B,
        "recon_curve_A": task_A["recon_curve"],
        # Task B, for symmetry: whether the new task was learned at all.
        "initial_reconstruction_loss_B": task_B["initial_reconstruction_loss"],
        "final_reconstruction_loss_B": task_B["final_reconstruction_loss"],
        "recon_curve_B": task_B["recon_curve"],
        # Run health.
        "n_nan_steps_A": task_A["n_nan_steps"],
        "n_nan_steps_B": task_B["n_nan_steps"],
        "n_update_steps_A": task_A["n_update_steps"],
        "n_update_steps_B": task_B["n_update_steps"],
        # Provenance.
        "method": method,
        "family": family,
        "distance": distance,
        "seed": seed,
        "protocol": protocol,
    }
    results_dir = results_root / method / f"{family}_{distance}_{seed}"
    save_metrics(metrics, results_dir / "metrics.json")

    n_nan = task_A["n_nan_steps"] + task_B["n_nan_steps"]
    if n_nan:
        print(f"    WARNING: {n_nan} of {2 * protocol['n_train']} steps "
              f"produced a NaN loss and were dropped")

    return metrics


def check_protocol_consistency(paths, protocol):
    """
    Refuse to reuse results produced under a different protocol.

    Skipping on the mere existence of metrics.json is what makes the runner
    resumable, but it would also silently average two training budgets into one
    table cell. Mismatches are reported all at once so the fix is a single
    move-or-delete.
    """
    mismatched = []
    for path in paths:
        stored = json.load(open(path)).get("protocol")
        if stored != protocol:
            mismatched.append(path)
    if mismatched:
        listing = "\n".join(f"  {p}" for p in mismatched[:10])
        more = "" if len(mismatched) <= 10 else \
            f"\n  ... and {len(mismatched) - 10} more"
        raise SystemExit(
            f"{len(mismatched)} cached result(s) were produced under a "
            f"different protocol than the one requested:\n{listing}{more}\n\n"
            "Archive or delete them before running with a new protocol — "
            "averaging two budgets into one table cell is exactly the kind of "
            "silent mixing this check exists to prevent."
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--families", nargs="+", default=list(FAMILY_CONFIGS),
                        choices=list(FAMILY_CONFIGS))
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--distances", nargs="+", default=DISTANCES,
                        choices=DISTANCES)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the effective protocol and run plan, then exit")
    # Protocol overrides. Anything left as None keeps the config's value.
    parser.add_argument("--steps", type=int, dest="n_train",
                        help="gradient updates per task (overrides n_train)")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--seq-len", type=int, dest="seq_len")
    parser.add_argument("--n-collect", type=int, dest="n_collect",
                        help="episodes collected per task")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="seeds to run (overrides the config's list)")
    return parser.parse_args(argv)


def protocol_overrides(args):
    """The subset of CLI flags that override protocol fields."""
    return {
        "n_train": args.n_train,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_collect": args.n_collect,
        "seeds": args.seeds,
    }


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    overrides = protocol_overrides(args)
    applied = {k: v for k, v in overrides.items() if v is not None}
    protocols = {}
    for family in args.families:
        cfg = OmegaConf.load(FAMILY_CONFIGS[family])
        protocols[family] = resolve_protocol(cfg, overrides)

    for family in args.families:
        print(f"\n=== effective protocol: {family} "
              f"({FAMILY_CONFIGS[family]}) ===")
        print(format_protocol(protocols[family]))
    if applied:
        print("\nOverridden from the command line: "
              + ", ".join(f"{k}={v}" for k, v in sorted(applied.items())))

    # d_param is only defined for the Gymnasium family, whose tasks differ by
    # physics parameters rather than by identity.
    if "gymnasium" in args.families:
        gym_cfg = OmegaConf.load(FAMILY_CONFIGS["gymnasium"])
        print("\n=== Gymnasium d_param values ===")
        for distance in args.distances:
            seq = gym_cfg.benchmark.sequences[distance]
            d = compute_d_param(dict(seq.task_A.params), dict(seq.task_B.params))
            print(f"  {distance}: d_param = {d:.4f}")

    results_root = args.results_dir
    planned = []
    for family in args.families:
        for distance in args.distances:
            for method in args.methods:
                for seed in protocols[family]["seeds"]:
                    planned.append((family, distance, method, seed))

    cached = [results_root / method / f"{family}_{distance}_{seed}" / "metrics.json"
              for family, distance, method, seed in planned]
    existing = [p for p in cached if p.exists()]
    for family in args.families:
        family_cached = [p for p in existing if f"{family}_" in p.parent.name]
        check_protocol_consistency(family_cached, protocols[family])

    print(f"\n{len(planned)} cell(s) planned, {len(existing)} already cached.")
    if args.dry_run:
        print("--dry-run: nothing was trained.")
        return

    for family in args.families:
        cfg = OmegaConf.load(FAMILY_CONFIGS[family])
        protocol = protocols[family]

        for distance in args.distances:
            seq_cfg = cfg.benchmark.sequences[distance]
            env_A, env_B = create_env_pair(family, seq_cfg)

            declared = int(cfg.model.action_dim)
            if declared != env_A.action_dim:
                raise SystemExit(
                    f"{FAMILY_CONFIGS[family]} declares action_dim="
                    f"{declared} but {family} exposes {env_A.action_dim}"
                )

            for method in args.methods:
                seeds = protocol["seeds"]
                paths = [results_root / method / f"{family}_{distance}_{s}" / "metrics.json"
                         for s in seeds]
                if all(p.exists() for p in paths):
                    print(f"  skip {method}/{family}/{distance} (all seeds exist)")
                    continue

                print(f"\n=== {method} | {family} | {distance} ===")
                for seed, path in zip(seeds, paths):
                    if path.exists():
                        m = json.load(open(path))
                        print(f"  seed={seed} | WMF={m['wmf']:.4f} (cached)")
                        continue

                    m = run_cell(method, env_A, env_B, device, seed, family,
                                 distance, protocol, results_root)
                    print(f"  seed={seed} | WMF={m['wmf']:.4f} PF={m['pf']:.4f} "
                          f"RD={m['rd']:.4f} FT={m['ft']:.4f} | "
                          f"task-A held-out recon={m['heldout_reconstruction_A_after_task_A']:.2f}"
                          f" -> {m['heldout_reconstruction_A_after_task_B']:.2f}")

            env_A.close()
            env_B.close()

    print_results_table(args, results_root)


def print_results_table(args, results_root):
    """Mean WMF per cell, read back from the metrics.json files."""
    print("\n" + "=" * 120)
    print("FULL RESULTS TABLE (mean WMF across seeds)")
    print("=" * 120)
    header = f"{'method':<20}"
    for family in args.families:
        for distance in args.distances:
            col = f"{FAMILY_ABBREV[family]}_{distance.split('_')[1]}"
            header += f" | {col:>10}"
    print(header)
    print("-" * 120)

    for method in args.methods:
        row = f"{method:<20}"
        for family in args.families:
            for distance in args.distances:
                files = list(results_root.glob(
                    f"{method}/{family}_{distance}_*/metrics.json"))
                if files:
                    wmfs = [json.load(open(f))["wmf"] for f in files]
                    row += f" | {np.mean(wmfs):>10.4f}"
                else:
                    row += f" | {'N/A':>10}"
        print(row)

    print("=" * 120)


if __name__ == "__main__":
    main()
