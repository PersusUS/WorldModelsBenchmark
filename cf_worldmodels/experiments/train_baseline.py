"""
Train a baseline method on a sequential task pair (task_A -> task_B).
Computes PF, RD, and WMF after training.

Usage:
    python experiments/train_baseline.py \
        --method finetuning \
        --config configs/benchmark/minigrid.yaml \
        --distance distance_min \
        --seed 0 --steps 5000 --no_wandb
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
from src.models.rssm import RSSM
from src.baselines.finetuning import FineTuningWorldModel
from src.baselines.replay import InfiniteReplayWorldModel
from src.baselines.ewc import EWCWorldModel
from src.baselines.progressive_nets import ProgressiveNetWorldModel
from src.utils.buffer import ReplayBuffer
from src.benchmark.protocol import build_latent_eval_dataset, collect_rollouts
from src.benchmark.metrics import compute_pf, compute_rd, compute_wmf, compute_ft, compute_nll
from src.utils.checkpointing import save_checkpoint
from src.utils.logging_utils import save_metrics
from src.utils.seeding import set_seed


def make_env(env_cfg, family):
    """Create environment from config."""
    if family == "minigrid":
        return MiniGridEnv(env_cfg.env_id)
    elif family == "gymnasium":
        from src.envs.gymnasium_env import GymnasiumEnv
        params = OmegaConf.to_container(env_cfg.get("params", {}))
        return GymnasiumEnv(env_cfg.env_id, physics_params=params)
    elif family == "dmcontrol":
        from src.envs.dmcontrol_env import DMControlEnv
        return DMControlEnv(env_cfg.domain_name, env_cfg.task_name)


def create_model(method, latent_dim, hidden_dim, action_dim, beta_kl=1.0):
    """Create model based on method name."""
    if method == "finetuning":
        return FineTuningWorldModel(latent_dim, hidden_dim, action_dim, beta_kl)
    elif method == "replay_infinite":
        return InfiniteReplayWorldModel(latent_dim, hidden_dim, action_dim, beta_kl)
    elif method == "ewc":
        return EWCWorldModel(latent_dim, hidden_dim, action_dim, ewc_lambda=1000.0, beta_kl=beta_kl)
    elif method == "progressive_nets":
        return ProgressiveNetWorldModel(latent_dim, hidden_dim, action_dim, beta_kl)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_on_task(model, buffer, optimizer, device, steps, cfg, task_name=""):
    """Train model for given steps on buffer data."""
    seq_len = min(cfg.protocol.seq_len, 20)
    batch_size = min(cfg.protocol.batch_size, 8)
    initial_loss = None
    final_loss = None

    model.train()
    for step in range(1, steps + 1):
        batch = buffer.sample(batch_size=batch_size, seq_len=seq_len)
        obs = batch["obs"].permute(0, 1, 4, 2, 3).to(device)
        actions = batch["actions"].to(device)

        loss, comps = model.compute_loss({"obs": obs, "actions": actions})
        assert not torch.isnan(loss), f"NaN loss at step {step}"

        optimizer.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        optimizer.step()

        if initial_loss is None:
            initial_loss = comps["reconstruction"]
        final_loss = comps["reconstruction"]

        if step % 1000 == 0:
            print(f"  [{task_name}] Step {step}/{steps} | recon={comps['reconstruction']:.4f}")

    return initial_loss, final_loss, comps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--distance", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    family = cfg.benchmark.family
    seq_cfg = cfg.benchmark.sequences[args.distance]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Method: {args.method} | Family: {family} | Distance: {args.distance} | Seed: {args.seed}")

    # Create environments
    env_A = make_env(seq_cfg.task_A, family)
    env_B = make_env(seq_cfg.task_B, family)
    # Environments own RNGs that no global seed reaches — see src/utils/seeding.py
    env_A.seed(args.seed)
    env_B.seed(args.seed + 1)
    action_dim = env_A.action_dim

    # Create model
    model = create_model(
        args.method,
        cfg.model.latent_dim,
        cfg.model.hidden_dim,
        action_dim,
        cfg.model.beta_kl,
    ).to(device)

    # Collect data for both tasks
    n_collect = min(cfg.protocol.n_collect, 30)
    buf_A = ReplayBuffer(max_episodes=n_collect, seq_len=min(cfg.protocol.seq_len, 20))
    buf_B = ReplayBuffer(max_episodes=n_collect, seq_len=min(cfg.protocol.seq_len, 20))

    # Held out of training: PF, RD and FT are all measured on these episodes.
    buf_A_heldout = ReplayBuffer(max_episodes=10, seq_len=2)

    print("Collecting Task A rollouts...")
    collect_rollouts(env_A, buf_A, n_rollouts=n_collect)
    print("Collecting Task B rollouts...")
    collect_rollouts(env_B, buf_B, n_rollouts=n_collect)
    print("Collecting held-out Task A rollouts for evaluation...")
    collect_rollouts(env_A, buf_A_heldout, n_rollouts=10)

    # Save snapshot of model after task A (for reference model_i)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Phase 1: Train on Task A
    print(f"\n--- Training on Task A ({args.steps} steps) ---")
    init_A, final_A, _ = train_on_task(model, buf_A, optimizer, device, args.steps, cfg, "Task A")

    # Save model_i state (after task A)
    model_i_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Reference model (model_i = after task A), and the held-out task-A
    # dataset D_i encoded once by it, so every model is scored on the same
    # inputs and targets.
    model_i = create_model(
        args.method, cfg.model.latent_dim, cfg.model.hidden_dim, action_dim,
        cfg.model.beta_kl,
    ).to(device)
    model_i.load_state_dict(model_i_state)
    eval_dataset = build_latent_eval_dataset(
        model_i, buf_A_heldout, device, n_transitions=100, seed=args.seed,
    )

    # Handle method-specific task switching
    if args.method == "ewc":
        # Consolidate the Fisher diagonal on real task-A transitions
        dataset_A = build_latent_eval_dataset(
            model_i, buf_A_heldout, device, n_transitions=100,
            seed=args.seed + 10_000,
        )
        model.consolidate(dataset_A, device)
    elif args.method == "progressive_nets":
        model.add_column()
    elif args.method == "replay_infinite":
        model.add_task_data(task_id=0, episodes=buf_A.episodes)

    # Reset optimizer for new task (weights persist)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # Phase 2: Train on Task B
    print(f"\n--- Training on Task B ({args.steps} steps) ---")
    if args.method == "replay_infinite":
        model.add_task_data(task_id=1, episodes=buf_B.episodes)
        # Create combined buffer
        combined_buf = ReplayBuffer(max_episodes=n_collect * 2, seq_len=min(cfg.protocol.seq_len, 20))
        for ep in buf_A.episodes:
            combined_buf.add_episode(ep)
        for ep in buf_B.episodes:
            combined_buf.add_episode(ep)
        init_B, final_B, final_comps = train_on_task(
            model, combined_buf, optimizer, device, args.steps, cfg, "Task B (replay)"
        )
    else:
        init_B, final_B, final_comps = train_on_task(
            model, buf_B, optimizer, device, args.steps, cfg, "Task B"
        )

    # Compute metrics
    print("\n--- Computing metrics ---")

    # model_i and eval_dataset were built at the task switch, above.

    # PF: forgetting of task A
    pf = compute_pf(model_i, model, eval_dataset, device)
    print(f"PF = {pf:.4f}")

    # RD: rollout divergence
    rd = compute_rd(model_i, model, eval_dataset, horizon=15, n_samples=50)
    print(f"RD = {rd:.4f}")

    # WMF, the previous paper's Eq. 6. Its gamma term is evaluated at zero:
    # PIS is not part of the suite (D18) and was never implemented, so nothing
    # is measured to put there. Stored as null below, not as 0.0.
    wmf = compute_wmf([pf], [rd], [0.0])
    print(f"WMF = {wmf:.4f}")

    # FT
    nll_before = compute_nll(model_i, eval_dataset, device)
    model_random = create_model(
        args.method, cfg.model.latent_dim, cfg.model.hidden_dim, action_dim, cfg.model.beta_kl
    ).to(device)
    nll_random = compute_nll(model_random, eval_dataset, device)
    ft = compute_ft(nll_before, nll_random)
    print(f"FT = {ft:.4f}")

    # Save results
    results_dir = Path("results") / args.method / f"{family}_{args.distance}_{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        model, optimizer, args.steps * 2, 1,
        OmegaConf.to_container(cfg), final_comps,
        results_dir / "checkpoint_final.pt",
    )

    metrics = {
        "wmf": wmf,
        "ft": ft,
        "pf": pf,
        "rd": rd,
        "pis": None,
        "initial_reconstruction_loss": init_A,
        "final_reconstruction_loss": final_B,
    }
    save_metrics(metrics, results_dir / "metrics.json")
    print(f"\nResults saved to {results_dir}")

    env_A.close()
    env_B.close()


if __name__ == "__main__":
    main()
