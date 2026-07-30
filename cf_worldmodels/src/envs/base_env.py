from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseEnv(ABC):
    """
    Abstract base class for all environment wrappers.
    ALL observations must be returned as float32 ndarray in [0.0, 1.0]
    with shape (64, 64, 3). Resize and normalize INSIDE the wrapper.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment. Returns obs: (64, 64, 3) float32 in [0,1]"""

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Step. Returns (obs, reward, done, info). obs: (64,64,3) float32 [0,1]"""

    @abstractmethod
    def sample_action(self) -> np.ndarray:
        """Sample random action from action space."""

    @abstractmethod
    def seed(self, seed: int) -> None:
        """
        Seed this environment's own RNGs: both the one driving `reset` and the
        one driving `sample_action`.

        Required for reproducibility. A global `np.random.seed` does not reach
        either of them, so without this two runs with the same seed collect
        different rollouts and therefore train on different data (F16).

        Seeding is a one-shot operation: call it once at the start of a run, not
        per episode, so that successive `reset()` calls walk a deterministic
        stream of distinct initial states rather than repeating one.
        """

    @property
    @abstractmethod
    def obs_shape(self) -> Tuple[int, int, int]:
        """Always returns (64, 64, 3)"""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of action space."""

    def close(self):
        """Override to clean up resources."""
        pass
