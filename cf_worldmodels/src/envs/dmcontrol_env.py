"""
DMControl environment wrapper.
Reference: https://dm-control.readthedocs.io/en/latest/suite.html
"""
from typing import Tuple, Optional, Dict
import numpy as np
from dm_control import suite

from src.envs.base_env import BaseEnv

PHYSICS_KEYS = ("gravity", "mass_scale", "friction_scale")


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

    def __init__(self, domain_name: str, task_name: str,
                 physics_params: Optional[Dict] = None):
        self._domain = domain_name
        self._task = task_name
        self._env = suite.load(domain_name=domain_name, task_name=task_name)
        self._action_spec = self._env.action_spec()
        # Own RNG for action sampling, so runs do not depend on whatever else
        # in the process has touched the global numpy RNG.
        self._rng = np.random.default_rng()

        self._physics_params = dict(physics_params or {})
        unknown = set(self._physics_params) - set(PHYSICS_KEYS)
        if unknown:
            # The `lateral_wind: true` that used to sit in the dmcontrol config
            # was read by nobody, so distance_min compared cheetah/run against
            # itself for the whole benchmark (F8). Unknown keys are now an
            # error: a perturbation that silently does nothing is worse than
            # one that never shipped.
            raise ValueError(
                f"unknown physics parameter(s): {', '.join(sorted(unknown))}. "
                f"Supported: {', '.join(PHYSICS_KEYS)}"
            )
        self._apply_physics()

    def _apply_physics(self):
        """
        Perturb the MuJoCo model in place, same knobs as the Gymnasium family.

        dm_control exposes the same MjModel, so these are the identical
        perturbations `GymnasiumEnv` applies, which is what makes d_param
        comparable across the two families. Note that MuJoCo's `opt.wind` is
        deliberately not offered: it only produces a force when `opt.density`
        or `opt.viscosity` is non-zero, and both default to zero here, so a
        wind setting alone would be another perturbation that does nothing.
        """
        model = self._env.physics.model
        if "gravity" in self._physics_params:
            model.opt.gravity[2] = -abs(self._physics_params["gravity"])
        if "mass_scale" in self._physics_params:
            model.body_mass[:] *= self._physics_params["mass_scale"]
        if "friction_scale" in self._physics_params:
            model.geom_friction[:] *= self._physics_params["friction_scale"]

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
        return self._rng.uniform(
            low=spec.minimum, high=spec.maximum, size=spec.shape
        ).astype(np.float32)

    def seed(self, seed: int) -> None:
        """Seed the task RNG and the action sampler; see BaseEnv.seed."""
        # dm_control randomizes the initial state through the task's own
        # RandomState, which `suite.load` seeds from OS entropy unless given
        # `task_kwargs={'random': ...}`. Reseeding it in place avoids rebuilding
        # the environment and its render context.
        self._env.task.random.seed(seed)
        self._rng = np.random.default_rng(seed)

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        return (64, 64, 3)

    @property
    def action_dim(self) -> int:
        return self._action_spec.shape[0]

    def close(self):
        pass  # dm_control doesn't require explicit cleanup
