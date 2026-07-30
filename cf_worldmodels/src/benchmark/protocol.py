"""
Benchmark protocol: data collection and sequential training procedures.
"""
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from src.envs.base_env import BaseEnv
from src.models.vae import reconstruction_loss
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


@torch.no_grad()
def evaluate_reconstruction(model, buffer: ReplayBuffer, device,
                            n_frames: int = 200, seed: Optional[int] = None,
                            chunk_size: int = 64) -> float:
    """
    Reconstruction error on held-out frames, in pixel space.

    Returns squared error summed over pixels and averaged over frames — the
    same quantity `compute_loss` reports as its "reconstruction" component, so
    the two are directly comparable. The difference is that these frames never
    entered the training buffer, and the encoder runs in eval mode (z = mu, no
    posterior sampling), so the number is deterministic.

    Why this exists: it is the only quality signal in the benchmark that is
    comparable **across training budgets**. The latent NLL is measured against
    latents the model itself produces (see `build_latent_eval_dataset`), and
    that target moves as the encoder trains, so NLL at 1000 steps and NLL at
    10000 steps are not scored on the same data. Pixels do not move.
    """
    was_training = model.training
    model.eval()

    frames = [step["obs"] for ep in buffer.episodes for step in ep]
    if not frames:
        raise ValueError(
            "cannot evaluate reconstruction: the buffer holds no frames"
        )

    rng = np.random.default_rng(seed)
    take = min(n_frames, len(frames))
    idx = rng.choice(len(frames), size=take, replace=False)

    total = 0.0
    for start in range(0, take, chunk_size):
        batch = [frames[i] for i in idx[start:start + chunk_size]]
        # (N, 64, 64, 3) float32 [0,1] -> (N, 3, 64, 64)
        x = torch.from_numpy(np.stack(batch)).to(device).permute(0, 3, 1, 2)
        z = model.encode(x)
        h = torch.zeros(x.shape[0], model.hidden_dim, device=device)
        recon = model.decode(h, z)
        # reconstruction_loss averages over the chunk; undo that so chunks of
        # different sizes weigh by their frame count.
        total += float(reconstruction_loss(recon, x).item()) * x.shape[0]

    model.train(was_training)
    return total / take
