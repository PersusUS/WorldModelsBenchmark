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
