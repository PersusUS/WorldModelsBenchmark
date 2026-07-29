from typing import Tuple
import numpy as np
import gymnasium
import minigrid.envs  # noqa: F401 — registers MiniGrid envs

from src.envs.base_env import BaseEnv


def _resize_obs(obs: np.ndarray) -> np.ndarray:
    """Resize observation to (64, 64, 3) float32 in [0, 1]."""
    from PIL import Image
    img = Image.fromarray(obs)
    img = img.resize((64, 64), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class MiniGridEnv(BaseEnv):
    """MiniGrid environment wrapper with 64x64 RGB observations."""

    def __init__(self, env_id: str):
        self._env = gymnasium.make(env_id, render_mode="rgb_array")
        self._env_id = env_id

    def reset(self) -> np.ndarray:
        # obs: (H, W, 3) uint8 in [0, 255]
        _obs, _info = self._env.reset()
        rgb = self._env.render()  # (H, W, 3) uint8
        return _resize_obs(rgb)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # action: one-hot (action_dim,) or scalar int for discrete envs
        if np.isscalar(action) or action.ndim == 0:
            act = int(action)
        elif action.shape[0] == self._env.action_space.n:
            act = int(np.argmax(action))  # decode one-hot
        else:
            act = int(action[0])
        obs, reward, terminated, truncated, info = self._env.step(act)
        rgb = self._env.render()  # (H, W, 3) uint8
        done = terminated or truncated
        return _resize_obs(rgb), float(reward), done, info

    def sample_action(self) -> np.ndarray:
        # One-hot encode discrete action for consistent (action_dim,) shape
        action = self._env.action_space.sample()
        one_hot = np.zeros(self._env.action_space.n, dtype=np.float32)
        one_hot[action] = 1.0
        return one_hot

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        return (64, 64, 3)

    @property
    def action_dim(self) -> int:
        return self._env.action_space.n

    def close(self):
        self._env.close()
