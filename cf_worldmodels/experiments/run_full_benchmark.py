"""
Run full benchmark: 5 methods × 3 families × 3 distances × 5 seeds.
Skips runs where results already exist.
"""
import sys
import json
import copy
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
from src.benchmark.protocol import build_latent_eval_dataset, collect_rollouts
from src.benchmark.metrics import compute_pf, compute_rd, compute_wmf, compute_ft, compute_nll
from src.benchmark.distances import compute_d_param
from src.utils.logging_utils import save_metrics

STEPS = 1000
SEQ_LEN = 5
BATCH_SIZE = 8
N_COLLECT = 20
SEEDS = 5

# Episodes collected from task A and held out of training. PF, RD and FT are
# all measured on these, so they must never enter the training buffer.
N_EVAL_EPISODES = 10
N_EVAL_TRANSITIONS = 100
# Transitions used to estimate EWC's Fisher diagonal (also real task-A data).
N_FISHER_TRANSITIONS = 50


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


def create_baseline_model(method, action_dim):
    """Create a baseline model."""
    if method == "finetuning":
        return FineTuningWorldModel(32, 512, action_dim)
    elif method == "replay_infinite":
        return InfiniteReplayWorldModel(32, 512, action_dim)
    elif method == "ewc":
        return EWCWorldModel(32, 512, action_dim, ewc_lambda=1000.0)
    elif method == "progressive_nets":
        return ProgressiveNetWorldModel(32, 512, action_dim)


def train_task(model, buffer, optimizer, device, steps):
    model.train()
    init_loss = None
    final_loss = None
    for step in range(1, steps + 1):
        batch = buffer.sample(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
        obs = batch["obs"].permute(0, 1, 4, 2, 3).to(device)
        actions = batch["actions"].to(device)
        loss, comps = model.compute_loss({"obs": obs, "actions": actions})
        if torch.isnan(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if init_loss is None:
            init_loss = comps["reconstruction"]
        final_loss = comps["reconstruction"]
    return init_loss, final_loss, comps


def run_baseline(method, env_A, env_B, device, seed, family, distance):
    torch.manual_seed(seed)
    np.random.seed(seed)

    action_dim = env_A.action_dim
    model = create_baseline_model(method, action_dim).to(device)

    buf_A = ReplayBuffer(max_episodes=N_COLLECT, seq_len=SEQ_LEN)
    buf_B = ReplayBuffer(max_episodes=N_COLLECT, seq_len=SEQ_LEN)
    buf_A_heldout = ReplayBuffer(max_episodes=N_EVAL_EPISODES, seq_len=2)
    collect_rollouts(env_A, buf_A, n_rollouts=N_COLLECT)
    collect_rollouts(env_B, buf_B, n_rollouts=N_COLLECT)
    collect_rollouts(env_A, buf_A_heldout, n_rollouts=N_EVAL_EPISODES)

    # Train Task A
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    init_A, final_A, _ = train_task(model, buf_A, opt, device, STEPS)
    model_i_state = {k: v.clone() for k, v in model.state_dict().items()}

    # D_i: held-out task-A transitions, encoded once by the post-task-A model
    # so that model_i and model_k are scored on identical inputs and targets.
    model_i = create_baseline_model(method, action_dim).to(device)
    model_i.load_state_dict(model_i_state)
    eval_ds = build_latent_eval_dataset(
        model_i, buf_A_heldout, device,
        n_transitions=N_EVAL_TRANSITIONS, seed=seed,
    )

    # Task switching
    if method == "ewc":
        ds = build_latent_eval_dataset(
            model_i, buf_A_heldout, device,
            n_transitions=N_FISHER_TRANSITIONS, seed=seed + 10_000,
        )
        model.consolidate(ds, device)
    elif method == "progressive_nets":
        model.add_column()
    elif method == "replay_infinite":
        model.add_task_data(task_id=0, episodes=buf_A.episodes)
        model.add_task_data(task_id=1, episodes=buf_B.episodes)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    # Train Task B
    if method == "replay_infinite":
        combined = ReplayBuffer(max_episodes=N_COLLECT * 2, seq_len=SEQ_LEN)
        for ep in buf_A.episodes:
            combined.add_episode(ep)
        for ep in buf_B.episodes:
            combined.add_episode(ep)
        init_B, final_B, comps = train_task(model, combined, opt, device, STEPS)
    else:
        init_B, final_B, comps = train_task(model, buf_B, opt, device, STEPS)

    # Compute metrics on the held-out task-A dataset built above
    pf = compute_pf(model_i, model, eval_ds, device)
    rd = compute_rd(model_i, model, eval_ds, horizon=15, n_samples=50)
    wmf = compute_wmf([pf], [rd], [0.0])

    model_rand = create_baseline_model(method, action_dim).to(device)
    nll_before = compute_nll(model_i, eval_ds, device)
    nll_random = compute_nll(model_rand, eval_ds, device)
    ft = compute_ft(nll_before, nll_random)

    results_dir = Path("results") / method / f"{family}_{distance}_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"wmf": wmf, "ft": ft, "pf": pf, "rd": rd, "pis": 0.0,
               "initial_reconstruction_loss": init_A, "final_reconstruction_loss": final_B}
    save_metrics(metrics, results_dir / "metrics.json")

    return wmf, pf, rd, ft


def run_ug_mtm(env_A, env_B, device, seed, family, distance):
    torch.manual_seed(seed)
    np.random.seed(seed)

    action_dim = env_A.action_dim
    ug_cfg = OmegaConf.load("configs/models/ug_mtm.yaml")
    ug_cfg.model.action_dim = int(action_dim)
    ug_cfg.model.mc_dropout_T = 3
    model = UG_MTM(ug_cfg).to(device)

    buf_A = ReplayBuffer(max_episodes=N_COLLECT, seq_len=SEQ_LEN)
    buf_B = ReplayBuffer(max_episodes=N_COLLECT, seq_len=SEQ_LEN)
    buf_A_heldout = ReplayBuffer(max_episodes=N_EVAL_EPISODES, seq_len=2)
    collect_rollouts(env_A, buf_A, n_rollouts=N_COLLECT)
    collect_rollouts(env_B, buf_B, n_rollouts=N_COLLECT)
    collect_rollouts(env_A, buf_A_heldout, n_rollouts=N_EVAL_EPISODES)

    # Train Task A
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    init_A, final_A, _ = train_task(model, buf_A, opt, device, STEPS)
    model_i_state = {k: v.clone() for k, v in model.state_dict().items()}

    model_i = UG_MTM(ug_cfg).to(device)
    model_i.load_state_dict(model_i_state)
    eval_ds = build_latent_eval_dataset(
        model_i, buf_A_heldout, device,
        n_transitions=N_EVAL_TRANSITIONS, seed=seed,
    )

    # Activate next expert, freeze shared components
    if model.K_active < model.K - 1:
        model.K_active += 1
    for name, param in model.named_parameters():
        if f"experts.{model.K_active}" in name:
            param.requires_grad_(True)
        elif "uncertainty_head" in name or "threshold_net" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    # Train Task B
    init_B, final_B, comps = train_task(model, buf_B, opt, device, STEPS)

    # Compute metrics on the held-out task-A dataset built above
    pf = compute_pf(model_i, model, eval_ds, device)
    rd = compute_rd(model_i, model, eval_ds, horizon=15, n_samples=50)
    wmf = compute_wmf([pf], [rd], [0.0])

    model_rand = UG_MTM(ug_cfg).to(device)
    nll_before = compute_nll(model_i, eval_ds, device)
    nll_random = compute_nll(model_rand, eval_ds, device)
    ft = compute_ft(nll_before, nll_random)

    results_dir = Path("results") / "ug_mtm" / f"{family}_{distance}_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"wmf": wmf, "ft": ft, "pf": pf, "rd": rd, "pis": 0.0,
               "initial_reconstruction_loss": init_A, "final_reconstruction_loss": final_B}
    save_metrics(metrics, results_dir / "metrics.json")

    return wmf, pf, rd, ft


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    families = {
        "minigrid": "configs/benchmark/minigrid.yaml",
        "gymnasium": "configs/benchmark/gymnasium.yaml",
        "dmcontrol": "configs/benchmark/dmcontrol.yaml",
    }
    methods = ["finetuning", "replay_infinite", "ewc", "progressive_nets", "ug_mtm"]
    distances = ["distance_min", "distance_med", "distance_max"]

    # Print d_param for gymnasium
    gym_cfg = OmegaConf.load("configs/benchmark/gymnasium.yaml")
    print("\n=== Gymnasium d_param values ===")
    for dist in distances:
        seq = gym_cfg.benchmark.sequences[dist]
        d = compute_d_param(dict(seq.task_A.params), dict(seq.task_B.params))
        print(f"  {dist}: d_param = {d:.4f}")

    for family, config_path in families.items():
        cfg = OmegaConf.load(config_path)

        for distance in distances:
            seq_cfg = cfg.benchmark.sequences[distance]
            env_A, env_B = create_env_pair(family, seq_cfg)

            for method in methods:
                # Check if all seeds already exist
                all_exist = all(
                    (Path("results") / method / f"{family}_{distance}_{s}" / "metrics.json").exists()
                    for s in range(SEEDS)
                )
                if all_exist:
                    print(f"  skip {method}/{family}/{distance} (all seeds exist)")
                    continue

                print(f"\n=== {method} | {family} | {distance} ===")
                for seed in range(SEEDS):
                    results_dir = Path("results") / method / f"{family}_{distance}_{seed}"
                    if (results_dir / "metrics.json").exists():
                        m = json.load(open(results_dir / "metrics.json"))
                        print(f"  seed={seed} | WMF={m['wmf']:.4f} (cached)")
                        continue

                    if method == "ug_mtm":
                        wmf, pf, rd, ft = run_ug_mtm(
                            env_A, env_B, device, seed, family, distance)
                    else:
                        wmf, pf, rd, ft = run_baseline(
                            method, env_A, env_B, device, seed, family, distance)

                    print(f"  seed={seed} | WMF={wmf:.4f} PF={pf:.4f} RD={rd:.4f} FT={ft:.4f}")

            env_A.close()
            env_B.close()

    # Print full results table
    print("\n" + "=" * 120)
    print("FULL RESULTS TABLE (mean WMF across 5 seeds)")
    print("=" * 120)
    header = f"{'method':<20}"
    for fam in families:
        for dist in distances:
            col = f"{fam[:3]}_{dist.split('_')[1]}"
            header += f" | {col:>10}"
    print(header)
    print("-" * 120)

    for method in methods:
        row = f"{method:<20}"
        for fam in families:
            for dist in distances:
                files = list(Path("results").glob(
                    f"{method}/{fam}_{dist}_*/metrics.json"))
                if files:
                    wmfs = [json.load(open(f))["wmf"] for f in files]
                    row += f" | {np.mean(wmfs):>10.4f}"
                else:
                    row += f" | {'N/A':>10}"
        print(row)

    print("=" * 120)


if __name__ == "__main__":
    main()
