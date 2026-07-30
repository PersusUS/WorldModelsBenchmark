"""
Reproducibility control for benchmark runs.

Seeding `torch` and `numpy` is not enough to make a run reproducible here. Two
independent sources of nondeterminism were measured (F16):

1. **The environments.** `collect_rollouts` calls `env.reset()` and
   `env.action_space.sample()`, and neither responds to `np.random.seed`:
   Gymnasium spaces and environments carry their own RNGs, seeded from OS
   entropy, and dm_control randomizes the initial state through the task's own
   `random` argument. Two runs with the same seed therefore trained on
   *different data*. Seed the environments through `BaseEnv.seed()`.

2. **The compute path.** cuDNN picks kernels by heuristic and its GRU backward
   is nondeterministic by default. With identical data and the same seed, the
   loss after 50 steps differed by 8.6e-02.

`set_seed` covers (2) and the global RNGs; the caller must also seed each
environment, which `set_seed` cannot reach.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed every global RNG and, by default, put the CUDA backend into
    deterministic mode.

    This does NOT seed the environments — they own RNGs that no global seed
    reaches. Call `BaseEnv.seed()` on each one as well; see the module
    docstring.

    Args:
        seed: the run's seed.
        deterministic: if True, force deterministic cuDNN kernels. Costs some
            throughput; required for a run to be reproducible on GPU.
            `cudnn.deterministic` plus `cudnn.benchmark = False` was verified
            sufficient to make training bit-identical on this setup, so
            `torch.use_deterministic_algorithms` is deliberately not enabled:
            it raises on ops that have no deterministic implementation without
            buying anything here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
