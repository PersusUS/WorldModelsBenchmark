"""
Gymnasium environment wrapper with variable physics support.
Reference: https://gymnasium.farama.org/v0.29.1/environments/mujoco/half_cheetah/
MuJoCo physics: https://mujoco.readthedocs.io/en/3.1.1/python.html
"""
from typing import Tuple, Optional, Dict
import numpy as np
import gymnasium

from src.envs.base_env import BaseEnv


def _resize_obs(obs: np.ndarray) -> np.ndarray:
    """Resize RGB observation to (64, 64, 3) float32 in [0, 1]."""
    from PIL import Image
    img = Image.fromarray(obs)
    img = img.resize((64, 64), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


class GymnasiumEnv(BaseEnv):
    """
    Gymnasium MuJoCo environment with variable physics parameters.
    Supports gravity, mass_scale, friction_scale modifications.
    """

    def __init__(self, env_id: str, physics_params: Optional[Dict] = None):
        # physics_params keys: gravity (float), mass_scale (float), friction_scale (float)
        self._env = gymnasium.make(env_id, render_mode="rgb_array")
        self._env_id = env_id
        self._physics_params = physics_params or {}

        # Apply physics modifications AFTER make() BEFORE first reset()
        self._apply_physics()

    def _apply_physics(self):
        """Modify MuJoCo physics parameters."""
        model = self._env.unwrapped.model
        if "gravity" in self._physics_params:
            model.opt.gravity[2] = -abs(self._physics_params["gravity"])
        if "mass_scale" in self._physics_params:
            model.body_mass[:] *= self._physics_params["mass_scale"]
        if "friction_scale" in self._physics_params:
            model.geom_friction[:] *= self._physics_params["friction_scale"]

    def reset(self) -> np.ndarray:
        """Reset environment. Returns obs: (64, 64, 3) float32 in [0,1]"""
        _obs, _info = self._env.reset()
        rgb = self._env.render()  # (H, W, 3) uint8
        return _resize_obs(rgb)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Step. Returns (obs, reward, done, info). obs: (64,64,3) float32 [0,1]"""
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        rgb = self._env.render()
        done = terminated or truncated
        return _resize_obs(rgb), float(reward), done, info

    def sample_action(self) -> np.ndarray:
        """Sample random action from continuous action space."""
        return self._env.action_space.sample().astype(np.float32)

    def seed(self, seed: int) -> None:
        """Seed the env RNG and the action-space sampler; see BaseEnv.seed."""
        # Passing the seed to reset() is Gymnasium's own way of seeding the
        # episode RNG. Later seedless reset() calls continue that stream.
        self._env.reset(seed=seed)
        self._env.action_space.seed(seed)

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        return (64, 64, 3)

    @property
    def action_dim(self) -> int:
        return self._env.action_space.shape[0]

    def close(self):
        self._env.close()
