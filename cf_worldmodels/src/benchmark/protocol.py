"""
Benchmark protocol: data collection and sequential training procedures.
"""
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from src.envs.base_env import BaseEnv
from src.utils.buffer import ReplayBuffer


def collect_rollouts(env: BaseEnv, buffer: ReplayBuffer, n_rollouts: int,
                     policy: str = "random", max_steps: int = 500) -> None:
    """
    Collect n_rollouts complete episodes using the given policy.
    Stores each episode in the buffer.

    Args:
        env: environment wrapper (must implement BaseEnv)
        buffer: replay buffer to store episodes
        n_rollouts: number of episodes to collect
        policy: 'random' (only random supported for now)
        max_steps: maximum steps per episode to prevent infinite loops
    """
    for _ in tqdm(range(n_rollouts), desc="Collecting rollouts"):
        episode = []
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = env.sample_action()
            next_obs, reward, done, info = env.step(action)
            episode.append({
                "obs": obs,
                "action": action.astype("float32") if hasattr(action, "astype") else action,
                "reward": float(reward),
                "done": done,
            })
            obs = next_obs
            steps += 1

        # Add terminal observation
        if len(episode) > 0:
            episode[-1]["done"] = True

        buffer.add_episode(episode)


@torch.no_grad()
def build_latent_eval_dataset(model, buffer: ReplayBuffer, device,
                              n_transitions: int = 100,
                              seed: Optional[int] = None) -> dict:
    """
    Encode held-out rollouts into the latent transition triples that the
    benchmark metrics consume.

    Returns a dict with:
        obs:      (N, latent_dim)  encoded z_t
        actions:  (N, action_dim)  action taken at t
        next_obs: (N, latent_dim)  encoded z_{t+1}

    This is the dataset D_i that PF, RD and FT are defined over. Two
    properties matter for the metrics to mean what Section 3.2 of the paper
    says they mean:

    1. It must come from real task-i rollouts, not from sampled noise.
    2. The observations must be encoded ONCE, by a single model, so that every
       model scored on this dataset is evaluated on identical inputs and
       identical targets. Encoding separately per model would compare each
       model against its own moving latent space rather than against D_i.

    Pass the model whose latent space defines the task (i.e. the snapshot
    taken immediately after training on task i).
    """
    was_training = model.training
    model.eval()

    obs, actions, next_obs = [], [], []
    for ep in buffer.episodes:
        for t in range(len(ep) - 1):
            obs.append(ep[t]["obs"])
            actions.append(ep[t]["action"])
            next_obs.append(ep[t + 1]["obs"])

    if not obs:
        raise ValueError(
            "cannot build an evaluation dataset: the buffer holds no episode "
            "with at least two transitions"
        )

    rng = np.random.default_rng(seed)
    take = min(n_transitions, len(obs))
    idx = rng.choice(len(obs), size=take, replace=False)

    def encode(frames):
        # (N, 64, 64, 3) float32 [0,1] -> (N, 3, 64, 64) -> (N, latent_dim)
        x = torch.from_numpy(np.stack(frames)).to(device).permute(0, 3, 1, 2)
        return model.encode(x)

    dataset = {
        "obs": encode([obs[i] for i in idx]),
        "actions": torch.from_numpy(
            np.stack([actions[i] for i in idx])
        ).float().to(device),
        "next_obs": encode([next_obs[i] for i in idx]),
    }

    model.train(was_training)
    return dataset
