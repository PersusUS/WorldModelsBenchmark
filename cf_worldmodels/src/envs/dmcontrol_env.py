"""
DMControl environment wrapper.
Reference: https://dm-control.readthedocs.io/en/latest/suite.html
"""
from typing import Tuple
import numpy as np
from dm_control import suite

from src.envs.base_env import BaseEnv


def _resize_obs(obs: np.ndarray) -> np.ndarray:
    """Resize RGB observation to (64, 64, 3) float32 in [0, 1]."""
    from PIL import Image
    img = Image.fromarray(obs)
    img = img.resize((64, 64), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class DMControlEnv(BaseEnv):
    """
    DMControl suite wrapper producing 64x64 RGB observations.
    Supported: ('cheetah','run'), ('walker','run'), ('reacher','easy')
    """

    def __init__(self, domain_name: str, task_name: str):
        self._domain = domain_name
        self._task = task_name
        self._env = suite.load(domain_name=domain_name, task_name=task_name)
        self._action_spec = self._env.action_spec()

    def reset(self) -> np.ndarray:
        """Reset environment. Returns obs: (64, 64, 3) float32 in [0,1]"""
        self._timestep = self._env.reset()
        rgb = self._env.physics.render(height=64, width=64, camera_id=0)
        return rgb.astype(np.float32) / 255.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Step. Returns (obs, reward, done, info). obs: (64,64,3) float32 [0,1]"""
        action = np.asarray(action, dtype=np.float64)
        # Clip action to spec bounds
        action = np.clip(action, self._action_spec.minimum, self._action_spec.maximum)
        self._timestep = self._env.step(action)
        rgb = self._env.physics.render(height=64, width=64, camera_id=0)
        obs = rgb.astype(np.float32) / 255.0
        reward = self._timestep.reward or 0.0
        done = self._timestep.last()
        return obs, float(reward), done, {}

    def sample_action(self) -> np.ndarray:
        """Sample random action from continuous action space."""
        spec = self._action_spec
        return np.random.uniform(
            low=spec.minimum, high=spec.maximum, size=spec.shape
        ).astype(np.float32)

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        return (64, 64, 3)

    @property
    def action_dim(self) -> int:
        return self._action_spec.shape[0]

    def close(self):
        pass  # dm_control doesn't require explicit cleanup
