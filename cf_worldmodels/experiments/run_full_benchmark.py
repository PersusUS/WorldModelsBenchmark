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

Each family runs in a subprocess of its own. That is not tidiness: dm_control
cannot create an OpenGL context in a process where Gymnasium's MuJoCo already
has one, and the run dies at the boundary (F24).
"""
import argparse
import json
import subprocess
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
from src.benchmark.metrics import (
    compute_pf,
    compute_rd,
    compute_wmf,
    compute_task_A_fit_gain,
    compute_forward_transfer,
    compute_nll,
)
from src.benchmark.distances import compute_d_param, compute_d_trans
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

# Recorded in every result's protocol block, but NOT part of what two results
# must share to be comparable. `seeds` is the list one invocation was asked to
# run: provenance, not budget. A cell trained under `--seeds 5 6 7 8 9` and one
# trained under the config's `[0, 1, 2, 3, 4]` have identical hyperparameters,
# and pooling them is the entire point of adding seeds to a finished grid.
# Comparing whole protocol dicts made the runner refuse its own cache after any
# --seeds override, and made the summary refuse to aggregate a grid it had just
# extended.
PROTOCOL_IDENTITY_EXCLUDED = {"seeds"}


def protocol_identity(protocol) -> dict:
    """The part of a protocol block two results must share to be pooled."""
    return {key: value for key, value in (protocol or {}).items()
            if key not in PROTOCOL_IDENTITY_EXCLUDED}


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
        params_A = dict(seq_cfg.task_A.get("params", {}))
        params_B = dict(seq_cfg.task_B.get("params", {}))
        env_A = DMControlEnv(seq_cfg.task_A.domain_name,
                             seq_cfg.task_A.task_name, physics_params=params_A)
        env_B = DMControlEnv(seq_cfg.task_B.domain_name,
                             seq_cfg.task_B.task_name, physics_params=params_B)
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


def preflight_action_dims(family, cfg, distances):
    """
    Check every task pair of a family before training anything.

    One world model spans both tasks of a pair, so its GRU is built with one
    action width; a pair whose two tasks disagree cannot be run at all. This
    used to surface as a torch shape error 12 hours into the family, from
    inside the transition (`input has inconsistent input_size: got 34 expected
    38`), because only task A was ever checked against the declared value
    (F25).

    Every pair is reported at once, so a config with two broken levels takes
    one fix rather than two runs to find.
    """
    declared = int(cfg.model.action_dim)
    problems = []
    for distance in distances:
        seq_cfg = cfg.benchmark.sequences[distance]
        env_A, env_B = create_env_pair(family, seq_cfg)
        try:
            for task, env in (("task_A", env_A), ("task_B", env_B)):
                if env.action_dim != declared:
                    problems.append(
                        f"  {distance}/{task}: exposes action_dim="
                        f"{env.action_dim}, config declares {declared}"
                    )
        finally:
            env_A.close()
            env_B.close()

    if problems:
        raise SystemExit(
            f"{FAMILY_CONFIGS[family]} declares action_dim={declared} but its "
            f"tasks do not agree:\n" + "\n".join(problems) + "\n\n"
            "A pair whose two tasks have different action widths cannot be "
            "run by one world model. Fix the pair, or give the family an "
            "action space both tasks fit in."
        )


def collect_cell_buffers(env_A, env_B, seed, protocol):
    """
    The four buffers every method of one (family, distance, seed) shares.

    Collected once and handed to all five cells and to the reference pair.
    Before this was hoisted, each of the six re-seeded the environments and
    re-collected the same episodes, which is where most of the wall time of a
    Gymnasium or dm_control cell went: 60 episodes at ~2.7 s each, six times
    over, for data that is identical by construction.

    Identical because the two environments are seeded here, once per cell
    rather than per episode, so successive resets walk a deterministic stream
    of distinct initial states — which is also what keeps the held-out episodes
    different from the ones that get trained on. Rollout content depends only
    on the environments' own RNGs and on the order of these calls; all three
    families sample actions from an environment-owned generator, so nothing
    here consumes the global torch/numpy stream.
    """
    set_seed(seed)
    env_A.seed(seed)
    env_B.seed(seed + 1)

    buf_A = ReplayBuffer(max_episodes=protocol["n_collect"],
                         seq_len=protocol["seq_len"])
    buf_B = ReplayBuffer(max_episodes=protocol["n_collect"],
                         seq_len=protocol["seq_len"])
    buf_A_heldout = ReplayBuffer(max_episodes=protocol["n_eval_episodes"],
                                 seq_len=2)
    buf_B_heldout = ReplayBuffer(max_episodes=protocol["n_eval_episodes"],
                                 seq_len=2)
    collect_rollouts(env_A, buf_A, n_rollouts=protocol["n_collect"])
    collect_rollouts(env_B, buf_B, n_rollouts=protocol["n_collect"])
    collect_rollouts(env_A, buf_A_heldout, n_rollouts=protocol["n_eval_episodes"])
    collect_rollouts(env_B, buf_B_heldout, n_rollouts=protocol["n_eval_episodes"])
    return buf_A, buf_B, buf_A_heldout, buf_B_heldout


def reference_path(results_root, family, distance, seed):
    return results_root / "_reference" / f"{family}_{distance}_{seed}.json"


def run_reference_cell(buffers, action_dim, device, seed, family, distance,
                       tasks, protocol, results_root):
    """
    Train the two from-scratch models the cells of this (family, distance,
    seed) need, and store what they are needed for.

    Two paper claims rest on a model that has seen exactly one task:

      * `d_trans` (Eq. 9) is a KL between the transition models of two
        *environments*, each trained on its own. Comparing the pre- and
        post-switch snapshots instead would measure how far one model moved,
        which is what RD already measures (F15/P8).
      * Forward transfer needs the counterfactual "task B learned without
        having seen task A" (F20/P10).

    Both are properties of the task pair and the seed, not of the continual
    learning method, so this runs once per (family, distance, seed) and every
    method's cell reads the same numbers back. Plain RSSMs are used for the
    same reason, and they train on the very buffers the cells train on.
    """
    set_seed(seed)
    buf_A, buf_B, buf_A_heldout, buf_B_heldout = buffers

    model_A = create_model("finetuning", action_dim, protocol).to(device)
    stats_A = train_task(
        model_A, buf_A,
        torch.optim.Adam(model_A.parameters(), lr=protocol["learning_rate"]),
        device, protocol,
    )
    model_B = create_model("finetuning", action_dim, protocol).to(device)
    stats_B = train_task(
        model_B, buf_B,
        torch.optim.Adam(model_B.parameters(), lr=protocol["learning_rate"]),
        device, protocol,
    )

    # d_trans is evaluated on task-A transitions, in model_A's latent basis:
    # KL(P_A || P_B) is already asymmetric, and scoring it on the source task
    # keeps it on the same (z, a) support as PF and RD.
    shared_ds = build_latent_eval_dataset(
        model_A, buf_A_heldout, device,
        n_transitions=protocol["n_eval_transitions"], seed=seed,
    )
    d_trans = compute_d_trans(model_A, model_B, shared_ds, device)

    reference = {
        "d_trans": d_trans,
        # The "no pretraining" arm of forward transfer: task-B quality reached
        # from a random init under this exact budget and this exact data.
        "heldout_reconstruction_B_from_scratch": evaluate_reconstruction(
            model_B, buf_B_heldout, device,
            n_frames=protocol["n_recon_frames"], seed=seed,
        ),
        # Task A from scratch, for the same reason PF has a random-init
        # reference: it says whether the reference models trained at all.
        "heldout_reconstruction_A_from_scratch": evaluate_reconstruction(
            model_A, buf_A_heldout, device,
            n_frames=protocol["n_recon_frames"], seed=seed,
        ),
        "final_reconstruction_loss_A": stats_A["final_reconstruction_loss"],
        "final_reconstruction_loss_B": stats_B["final_reconstruction_loss"],
        "n_nan_steps_A": stats_A["n_nan_steps"],
        "n_nan_steps_B": stats_B["n_nan_steps"],
        "family": family,
        "distance": distance,
        "seed": seed,
        "tasks": tasks,
        "protocol": protocol,
    }
    save_metrics(reference, reference_path(results_root, family, distance, seed))
    return reference


def load_reference(results_root, family, distance, seed, protocol, tasks=None):
    """
    The stored reference for this cell, or None if it was never computed.

    A reference produced under a different protocol is refused for the same
    reason cached cells are: the from-scratch arm of forward transfer is only a
    counterfactual if it ran at the same budget.
    """
    path = reference_path(results_root, family, distance, seed)
    if not path.exists():
        return None
    stored = json.load(open(path))
    if protocol_identity(stored.get("protocol")) != protocol_identity(protocol):
        raise SystemExit(
            f"{path} was produced under a different protocol than the one "
            "requested. Archive or delete it before running."
        )
    if tasks is not None and stored.get("tasks") != tasks:
        raise SystemExit(
            f"{path} was produced from a different task pair than the one "
            "requested. Archive or delete it before running."
        )
    return stored


def run_cell(method, buffers, action_dim, device, seed, family, distance,
             tasks, protocol, results_root, reference=None):
    """Train one (method, family, distance, seed) cell and write its metrics."""
    # Seeding the global RNGs is not enough: the environments own RNGs that no
    # global seed reaches, and cuDNN needs to be put in deterministic mode. See
    # src/utils/seeding.py. `buffers` was collected under exactly this seed by
    # collect_cell_buffers, and rollout collection consumes nothing from the
    # global stream, so re-seeding here puts the model initialisation at the
    # same point it would reach if the collection had happened inline.
    set_seed(seed)

    buf_A, buf_B, buf_A_heldout, buf_B_heldout = buffers
    model = create_model(method, action_dim, protocol).to(device)

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
    # The 0.0 is not a measurement of PIS: it is the value the previous paper's
    # Eq. 6 was effectively evaluated at, and reproducing that equation is the
    # only reason WMF is still computed (D10, D18). What gets stored under
    # "pis" below is null, because nothing was measured.
    wmf = compute_wmf([pf], [rd], [0.0], alpha=weights["alpha"],
                      beta=weights["beta"], gamma=weights["gamma"])

    model_rand = create_model(method, action_dim, protocol).to(device)
    # model_i is the frozen post-task-A snapshot, so its NLL on D_A is the
    # task-A fit regardless of when it is evaluated.
    nll_A_after_task_A = compute_nll(model_i, eval_ds, device)
    nll_A_random_init = compute_nll(model_rand, eval_ds, device)
    task_A_fit_gain = compute_task_A_fit_gain(nll_A_after_task_A,
                                              nll_A_random_init)

    with preserve_rng_state():
        nll_A_after_task_B = compute_nll(model, eval_ds, device)
        recon_A_after_task_B = evaluate_reconstruction(
            model, buf_A_heldout, device,
            n_frames=protocol["n_recon_frames"], seed=seed,
        )
        # Task-B quality, on the same held-out episodes and with the same
        # estimator the reference used. This is the pretrained arm of forward
        # transfer; the from-scratch arm comes out of run_reference_cell.
        recon_B_after_task_B = evaluate_reconstruction(
            model, buf_B_heldout, device,
            n_frames=protocol["n_recon_frames"], seed=seed,
        )

    ft = None
    d_trans = None
    if reference is not None:
        ft = compute_forward_transfer(
            reference["heldout_reconstruction_B_from_scratch"],
            recon_B_after_task_B,
        )
        d_trans = reference["d_trans"]

    metrics = {
        # Forgetting metrics (Section 3.2). PF and RD are what the benchmark
        # reports; WMF is kept because it is the previous paper's Eq. 6, but it
        # is an aggregate of components on different scales, of which RD
        # supplies 78-97% (F14/P7).
        #
        # PIS is null, not 0.0 (D18/F6). It was never implemented — scoring a
        # policy trained in imagination needs a controller this repository does
        # not have — and a stored 0.0 reads as "measured, and it came out zero".
        # Null is the same convention ft and d_trans use when their reference
        # was skipped: never measured, never averaged in.
        "pf": pf,
        "rd": rd,
        "wmf": wmf,
        "pis": None,
        # Forward transfer, measured against a model trained on task B from
        # scratch (F20/P10). None when the reference for this cell was skipped.
        "ft": ft,
        "heldout_reconstruction_B_after_task_B": recon_B_after_task_B,
        "heldout_reconstruction_B_from_scratch": (
            None if reference is None
            else reference["heldout_reconstruction_B_from_scratch"]
        ),
        # Eq. 9, between one model per environment (F15/P8). A property of the
        # task pair and the seed, copied into every method's cell so that the
        # results directory stays self-contained.
        "d_trans": d_trans,
        # What used to be called FT. No task-B data enters it (F20).
        "task_A_fit_gain": task_A_fit_gain,
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
        # Which two tasks this cell actually ran. The protocol block says at
        # what budget; without this, a config change to a task pair leaves
        # stale cells that look current (F26).
        "tasks": tasks,
        "protocol": protocol,
    }
    results_dir = results_root / method / f"{family}_{distance}_{seed}"
    save_metrics(metrics, results_dir / "metrics.json")

    n_nan = task_A["n_nan_steps"] + task_B["n_nan_steps"]
    if n_nan:
        print(f"    WARNING: {n_nan} of {2 * protocol['n_train']} steps "
              f"produced a NaN loss and were dropped")

    return metrics


def task_spec(seq_cfg):
    """The two tasks of a distance level, as plain JSON."""
    return {
        "task_A": OmegaConf.to_container(seq_cfg.task_A, resolve=True),
        "task_B": OmegaConf.to_container(seq_cfg.task_B, resolve=True),
    }


def check_protocol_consistency(paths, protocol, tasks_by_distance=None):
    """
    Refuse to reuse results whose protocol or task pair differs from the one
    being asked for.

    Skipping on the mere existence of metrics.json is what makes the runner
    resumable, but it would also silently average two training budgets into one
    table cell. The task pair is checked for the same reason and learned the
    hard way: `metrics.json` recorded the protocol but not *which tasks ran*,
    so changing a broken pair in the config left 25 stale cells that the runner
    would have happily reused under the new pair's name (F26).

    Mismatches are reported all at once so the fix is a single move-or-delete.
    """
    mismatched = []
    for path in paths:
        stored = json.load(open(path))
        if protocol_identity(stored.get("protocol")) != protocol_identity(protocol):
            mismatched.append((path, "protocol"))
            continue
        if tasks_by_distance is not None:
            # results/<method>/<family>_<distance>_<seed>/metrics.json
            distance = stored.get("distance")
            expected = tasks_by_distance.get(distance)
            if expected is not None and stored.get("tasks") != expected:
                mismatched.append((path, "task pair"))

    if mismatched:
        listing = "\n".join(f"  {p}  ({why})" for p, why in mismatched[:10])
        more = "" if len(mismatched) <= 10 else \
            f"\n  ... and {len(mismatched) - 10} more"
        raise SystemExit(
            f"{len(mismatched)} cached result(s) do not match what was "
            f"requested:\n{listing}{more}\n\n"
            "Archive or delete them before rerunning — averaging two budgets, "
            "or two task pairs, into one table cell is exactly the kind of "
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
    parser.add_argument("--skip-reference", action="store_true",
                        help="do not train the from-scratch task-B reference; "
                             "cells then store ft and d_trans as null")
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


def family_subprocess_argv(args, family):
    """The command line that runs exactly one family of `args`.

    Every flag has to be forwarded. A flag built here and then dropped is how
    F9 happened: `run_ablations.py` assembled its overrides, printed them, and
    never passed them on, so five ablations silently ran the unmodified config.
    """
    argv = [sys.executable, str(Path(__file__).resolve()),
            "--families", family,
            "--methods", *args.methods,
            "--distances", *args.distances,
            "--results-dir", str(args.results_dir)]
    if args.skip_reference:
        argv.append("--skip-reference")
    for flag, value in (("--steps", args.n_train),
                        ("--batch-size", args.batch_size),
                        ("--seq-len", args.seq_len),
                        ("--n-collect", args.n_collect)):
        if value is not None:
            argv += [flag, str(value)]
    if args.seeds is not None:
        argv += ["--seeds", *[str(s) for s in args.seeds]]
    return argv


def run_families_in_subprocesses(args):
    """
    Run each family in a process of its own, then print the combined table.

    Not an optimisation — a requirement. MuJoCo and dm_control both want an
    OpenGL context, and dm_control cannot build one in a process where
    Gymnasium's MuJoCo already has: the second family to ask dies with
    `mujoco.FatalError: Default framebuffer is not complete`. It killed the
    first full run 30 hours in, at the gymnasium -> dmcontrol boundary (F24).
    Closing the environments does not release the context; only process exit
    does.

    A fresh interpreter per family also means that a crash in one family costs
    that family rather than the run, and the cells already on disk are skipped
    on the next attempt anyway.
    """
    for family in args.families:
        print(f"\n{'#' * 78}\n# {family} (subprocess)\n{'#' * 78}", flush=True)
        result = subprocess.run(family_subprocess_argv(args, family))
        if result.returncode != 0:
            raise SystemExit(
                f"the {family} subprocess exited with {result.returncode}. "
                f"The families that finished are on disk and will be skipped "
                f"when you rerun."
            )
    print_results_table(args, args.results_dir)


def main(argv=None):
    args = parse_args(argv)
    # One family per process; see run_families_in_subprocesses. --dry-run is
    # pure printing, so it stays in this one.
    if len(args.families) > 1 and not args.dry_run:
        return run_families_in_subprocesses(args)

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
        cfg = OmegaConf.load(FAMILY_CONFIGS[family])
        tasks_by_distance = {
            distance: task_spec(cfg.benchmark.sequences[distance])
            for distance in args.distances
        }
        check_protocol_consistency(family_cached, protocols[family],
                                   tasks_by_distance)

    print(f"\n{len(planned)} cell(s) planned, {len(existing)} already cached.")
    if args.skip_reference:
        print("--skip-reference: ft and d_trans will be stored as null.")
    else:
        refs = {(family, distance, seed)
                for family, distance, _method, seed in planned}
        cached_refs = sum(
            1 for family, distance, seed in refs
            if reference_path(results_root, family, distance, seed).exists())
        print(f"{len(refs)} reference model pair(s) planned "
              f"(one per family/distance/seed, two trainings each), "
              f"{cached_refs} already cached.")
    if args.dry_run:
        print("--dry-run: nothing was trained.")
        return

    for family in args.families:
        cfg = OmegaConf.load(FAMILY_CONFIGS[family])
        protocol = protocols[family]

        preflight_action_dims(family, cfg, args.distances)

        for distance in args.distances:
            seq_cfg = cfg.benchmark.sequences[distance]
            tasks = task_spec(seq_cfg)
            env_A, env_B = create_env_pair(family, seq_cfg)

            # The loop is nested by seed rather than by method so that the four
            # buffers are collected once and shared by the five cells and the
            # reference pair, all of which would otherwise re-collect the same
            # episodes.
            for seed in protocol["seeds"]:
                pending = [
                    method for method in args.methods
                    if not (results_root / method /
                            f"{family}_{distance}_{seed}" /
                            "metrics.json").exists()
                ]
                reference = load_reference(results_root, family, distance,
                                           seed, protocol, tasks)
                # Only train a missing reference if some cell still needs it: a
                # cached cell is never recomputed, so its ft would stay null
                # either way and the pair would cost two runs for nothing.
                need_reference = (reference is None and pending
                                  and not args.skip_reference)

                cached = len(args.methods) - len(pending)
                print(f"\n=== {family} | {distance} | seed={seed} ==="
                      + (f"  ({cached} method(s) cached)" if cached else ""))
                if not pending and not need_reference:
                    continue

                buffers = collect_cell_buffers(env_A, env_B, seed, protocol)

                if need_reference:
                    reference = run_reference_cell(
                        buffers, env_A.action_dim, device, seed, family,
                        distance, tasks, protocol, results_root)
                    print(f"  reference   | d_trans={reference['d_trans']:.4f} "
                          f"| task-B held-out recon from scratch="
                          f"{reference['heldout_reconstruction_B_from_scratch']:.2f}")

                for method in pending:
                    m = run_cell(method, buffers, env_A.action_dim, device,
                                 seed, family, distance, tasks, protocol,
                                 results_root, reference=reference)
                    ft = "  n/a" if m["ft"] is None else f"{m['ft']:.4f}"
                    print(f"  {method:<16} | PF={m['pf']:.4f} "
                          f"RD={m['rd']:.4f} FT={ft} (WMF={m['wmf']:.4f}) | "
                          f"task-A held-out recon="
                          f"{m['heldout_reconstruction_A_after_task_A']:.2f}"
                          f" -> {m['heldout_reconstruction_A_after_task_B']:.2f}")

            env_A.close()
            env_B.close()

    print_results_table(args, results_root)


def print_results_table(args, results_root):
    """
    Mean PF and RD per cell, read back from the metrics.json files.

    Two tables rather than one WMF table: the components live on different
    scales and RD supplies 78-97% of the aggregate (F14/P7), so a single column
    of WMF hides which metric is doing the work. `summarize_results.py` is the
    tool that produces the paper's tables; this is the end-of-run glance.
    """
    for key in ("pf", "rd"):
        print("\n" + "=" * 120)
        print(f"FULL RESULTS TABLE (mean {key.upper()} across seeds)")
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
                    values = [json.load(open(f))[key] for f in files]
                    if values:
                        row += f" | {np.mean(values):>10.4f}"
                    else:
                        row += f" | {'N/A':>10}"
            print(row)

        print("=" * 120)


if __name__ == "__main__":
    main()
