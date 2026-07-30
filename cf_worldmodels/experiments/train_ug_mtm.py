"""
Train UG-MTM on sequential task pairs.

Usage:
    python experiments/train_ug_mtm.py \
        --config configs/benchmark/minigrid.yaml \
        --distances distance_min distance_med distance_max \
        --seeds 0 1 2 3 4 \
        --no_wandb
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from src.envs.minigrid_env import MiniGridEnv
from src.models.ug_mtm import UG_MTM
from src.utils.buffer import ReplayBuffer
from src.benchmark.protocol import build_latent_eval_dataset, collect_rollouts
from src.benchmark.metrics import compute_pf, compute_rd, compute_wmf, compute_ft, compute_nll
from src.utils.checkpointing import save_checkpoint
from src.utils.logging_utils import save_metrics
from src.utils.seeding import set_seed


def make_env(env_cfg, family):
    if family == "minigrid":
        return MiniGridEnv(env_cfg.env_id)
    elif family == "gymnasium":
        from src.envs.gymnasium_env import GymnasiumEnv
        params = OmegaConf.to_container(env_cfg.get("params", {}))
        return GymnasiumEnv(env_cfg.env_id, physics_params=params)
    elif family == "dmcontrol":
        from src.envs.dmcontrol_env import DMControlEnv
        return DMControlEnv(env_cfg.domain_name, env_cfg.task_name)


def train_on_task(model, buffer, optimizer, device, steps, cfg, task_name=""):
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
            print(f"  [{task_name}] Step {step}/{steps} | recon={comps['reconstruction']:.4f}"
                  f" | uncertainty={comps.get('uncertainty', 0):.4f}")

    return initial_loss, final_loss, comps


def run_single(cfg, distance, seed, steps, no_wandb, method="ug_mtm"):
    family = cfg.benchmark.family
    seq_cfg = cfg.benchmark.sequences[distance]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_A = make_env(seq_cfg.task_A, family)
    env_B = make_env(seq_cfg.task_B, family)
    # Environments own RNGs that no global seed reaches — see src/utils/seeding.py
    env_A.seed(seed)
    env_B.seed(seed + 1)
    action_dim = env_A.action_dim

    # Override action_dim in config
    ug_cfg = OmegaConf.load("configs/models/ug_mtm.yaml")
    ug_cfg.model.action_dim = action_dim
    model = UG_MTM(ug_cfg).to(device)

    n_collect = min(cfg.protocol.n_collect, 30)
    buf_A = ReplayBuffer(max_episodes=n_collect, seq_len=min(cfg.protocol.seq_len, 20))
    buf_B = ReplayBuffer(max_episodes=n_collect, seq_len=min(cfg.protocol.seq_len, 20))

    print("Collecting Task A rollouts...")
    collect_rollouts(env_A, buf_A, n_rollouts=n_collect)
    # Held out of training: PF, RD and FT are all measured on these episodes.
    buf_A_heldout = ReplayBuffer(max_episodes=10, seq_len=2)
    collect_rollouts(env_A, buf_A_heldout, n_rollouts=10)
    print("Collecting Task B rollouts...")
    collect_rollouts(env_B, buf_B, n_rollouts=n_collect)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n--- UG-MTM Training Task A ({steps} steps) ---")
    init_A, final_A, _ = train_on_task(model, buf_A, optimizer, device, steps, cfg, "Task A")

    model_i_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Activate next expert for new task
    if model.K_active < model.K - 1:
        model.K_active += 1
        print(f"Activated expert {model.K_active}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    print(f"\n--- UG-MTM Training Task B ({steps} steps) ---")
    init_B, final_B, final_comps = train_on_task(
        model, buf_B, optimizer, device, steps, cfg, "Task B"
    )

    # Compute metrics
    print("\n--- Computing metrics ---")

    # Reference model_i (after task A), and D_i encoded once by it so both
    # models are scored on identical inputs and targets.
    model_i = UG_MTM(ug_cfg).to(device)
    model_i.load_state_dict(model_i_state)
    eval_dataset = build_latent_eval_dataset(
        model_i, buf_A_heldout, device, n_transitions=100, seed=seed,
    )

    pf = compute_pf(model_i, model, eval_dataset, device)
    rd = compute_rd(model_i, model, eval_dataset, horizon=15, n_samples=50)
    pis = 0.0
    wmf = compute_wmf([pf], [rd], [pis])

    nll_before = compute_nll(model_i, eval_dataset, device)
    model_random = UG_MTM(ug_cfg).to(device)
    nll_random = compute_nll(model_random, eval_dataset, device)
    ft = compute_ft(nll_before, nll_random)

    print(f"PF={pf:.4f} | RD={rd:.4f} | WMF={wmf:.4f} | FT={ft:.4f}")

    results_dir = Path("results") / "ug_mtm" / f"{family}_{distance}_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        model, optimizer, steps * 2, 1,
        OmegaConf.to_container(cfg), final_comps,
        results_dir / "checkpoint_final.pt",
    )

    metrics = {
        "wmf": wmf, "ft": ft, "pf": pf, "rd": rd, "pis": pis,
        "initial_reconstruction_loss": init_A,
        "final_reconstruction_loss": final_B,
    }
    save_metrics(metrics, results_dir / "metrics.json")
    print(f"Results saved to {results_dir}")

    env_A.close()
    env_B.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="ug_mtm")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--distance", type=str, default=None)
    parser.add_argument("--distances", type=str, nargs="+",
                        default=["distance_min", "distance_med", "distance_max"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    distances = [args.distance] if args.distance else args.distances
    seeds = [args.seed] if args.seed is not None else args.seeds

    for dist in distances:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"UG-MTM | distance={dist} | seed={seed}")
            print(f"{'='*60}")
            run_single(cfg, dist, seed, args.steps, args.no_wandb)


if __name__ == "__main__":
    main()
