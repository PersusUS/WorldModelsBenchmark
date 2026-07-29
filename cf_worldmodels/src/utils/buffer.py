from typing import List, Dict
import numpy as np
import torch


class ReplayBuffer:
    """Episode-based replay buffer for world model training."""

    def __init__(self, max_episodes: int, seq_len: int):
        # max_episodes: maximum number of episodes to store
        # seq_len: minimum episode length to accept
        self.max_episodes = max_episodes
        self.seq_len = seq_len
        self.episodes: List[List[Dict]] = []

    def add_episode(self, episode: List[Dict]) -> None:
        """Add an episode if it has at least seq_len transitions."""
        if len(episode) < self.seq_len:
            return
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)  # drop oldest
        self.episodes.append(episode)

    def sample(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Sample random windows from stored episodes.
        Returns dict with torch tensors:
            obs: (batch_size, seq_len, 64, 64, 3)
            actions: (batch_size, seq_len, action_dim)
            rewards: (batch_size, seq_len)
            dones: (batch_size, seq_len)
        """
        indices = np.random.randint(0, len(self.episodes), size=batch_size)

        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        for idx in indices:
            ep = self.episodes[idx]
            max_start = len(ep) - seq_len
            start = np.random.randint(0, max_start + 1)
            window = ep[start : start + seq_len]

            obs_batch.append(np.stack([t["obs"] for t in window]))
            act_batch.append(np.stack([t["action"] for t in window]))
            rew_batch.append(np.array([t["reward"] for t in window], dtype=np.float32))
            done_batch.append(np.array([t["done"] for t in window], dtype=np.float32))

        return {
            "obs": torch.from_numpy(np.stack(obs_batch)),       # (B, T, 64, 64, 3)
            "actions": torch.from_numpy(np.stack(act_batch)),   # (B, T, A)
            "rewards": torch.from_numpy(np.stack(rew_batch)),   # (B, T)
            "dones": torch.from_numpy(np.stack(done_batch)),    # (B, T)
        }

    def __len__(self) -> int:
        return len(self.episodes)
