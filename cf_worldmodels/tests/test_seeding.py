"""Tests for src/utils/seeding.py and the reproducibility it is meant to buy.

Regression suite for F16: seeding torch and numpy was not enough to make a run
reproducible, because the environments own RNGs no global seed reaches and cuDNN
picks kernels nondeterministically.
"""
import hashlib
import random

import numpy as np
import pytest
import torch

from src.utils.seeding import preserve_rng_state, set_seed


# --- set_seed --------------------------------------------------------------

def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.randn(16)
    set_seed(123)
    assert torch.equal(a, torch.randn(16))


def test_set_seed_makes_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(16)
    set_seed(123)
    assert np.array_equal(a, np.random.rand(16))


def test_set_seed_makes_python_random_reproducible():
    import random

    set_seed(123)
    a = [random.random() for _ in range(8)]
    set_seed(123)
    assert a == [random.random() for _ in range(8)]


def test_set_seed_enables_deterministic_cudnn_by_default():
    """Without this, cuDNN's GRU backward is nondeterministic and two runs with
    the same seed diverge even on identical data."""
    set_seed(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_can_opt_out_of_determinism():
    """Deterministic kernels cost throughput; the flag must be escapable."""
    set_seed(0, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    set_seed(0)  # restore for the rest of the session


def test_different_seeds_give_different_draws():
    set_seed(1)
    a = torch.randn(16)
    set_seed(2)
    assert not torch.equal(a, torch.randn(16))


# --- training is reproducible end to end -----------------------------------

@pytest.mark.slow
def test_training_twice_with_the_same_seed_gives_identical_weights():
    """The actual promise of F16, on the compute path: same seed, same data,
    bit-identical result. Measured at 8.6e-02 divergence in loss before the fix."""
    from src.baselines.finetuning import FineTuningWorldModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fixed data, so this isolates the compute path from environment sampling.
    obs = torch.rand(4, 3, 3, 64, 64,
                     generator=torch.Generator().manual_seed(7))
    actions = torch.rand(4, 3, 5,
                         generator=torch.Generator().manual_seed(8))

    def train_once():
        set_seed(999)
        model = FineTuningWorldModel(16, 32, 5).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(10):
            loss, _ = model.compute_loss({"obs": obs, "actions": actions})
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.item()), [p.detach().cpu().clone()
                                    for p in model.parameters()]

    loss_a, params_a = train_once()
    loss_b, params_b = train_once()

    assert loss_a == loss_b, f"loss diverged: {loss_a!r} vs {loss_b!r}"
    for pa, pb in zip(params_a, params_b):
        assert torch.equal(pa, pb)


# --- environments are seedable ---------------------------------------------

def test_base_env_requires_a_seed_method():
    """Every wrapper must be seedable: a global numpy seed does not reach the
    RNGs behind `reset` and `sample_action`."""
    from src.envs.base_env import BaseEnv

    assert "seed" in BaseEnv.__abstractmethods__


def _rollout_fingerprint(env, seed, n_rollouts=2, max_steps=6):
    """Reseed `env` and hash every observation and action a training loop would
    then see."""
    from src.benchmark.protocol import collect_rollouts
    from src.utils.buffer import ReplayBuffer

    set_seed(seed)
    env.seed(seed)
    buf = ReplayBuffer(max_episodes=n_rollouts, seq_len=2)
    collect_rollouts(env, buf, n_rollouts=n_rollouts, max_steps=max_steps)

    h = hashlib.sha256()
    for ep in buf.episodes:
        for s in ep:
            h.update(np.ascontiguousarray(s["obs"]).tobytes())
            h.update(np.ascontiguousarray(
                np.asarray(s["action"], dtype=np.float32)).tobytes())
    return h.hexdigest()


@pytest.mark.integration
@pytest.mark.parametrize("family", ["minigrid", "gymnasium", "dmcontrol"])
def test_seeded_rollouts_are_reproducible(family):
    """The dominant source of F16: unseeded environments meant two runs with the
    same seed trained on different data.

    Reseeds one environment rather than building two. For all three wrappers
    `seed()` fully reseeds the stream, so this is equivalent — and it avoids
    creating a second simulator, which is not safe here: see I20.
    """
    if family == "minigrid":
        from src.envs.minigrid_env import MiniGridEnv
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0")
    elif family == "gymnasium":
        from src.envs.gymnasium_env import GymnasiumEnv
        env = GymnasiumEnv("HalfCheetah-v4")
    else:
        import mujoco
        from src.envs.dmcontrol_env import DMControlEnv
        try:
            env = DMControlEnv("cheetah", "run")
            env.reset()
        except mujoco.FatalError as exc:
            # I20: once dm_control has rendered in a process, opening and
            # closing a MuJoCo/Gymnasium environment invalidates the shared GLFW
            # context, and any dm_control environment built afterwards cannot
            # render. Verified not to affect run_full_benchmark.py, whose family
            # order (minigrid, gymnasium, dmcontrol) never hits the sequence.
            # This test does cover dm_control when run on its own:
            #   pytest tests/test_seeding.py -k dmcontrol
            pytest.skip(f"dm_control render context already invalidated (I20): {exc}")

    try:
        assert _rollout_fingerprint(env, 999) == _rollout_fingerprint(env, 999)
    finally:
        env.close()


@pytest.mark.integration
def test_different_seeds_give_different_rollouts():
    """Seeding must not flatten the environment into a single fixed trajectory."""
    from src.envs.minigrid_env import MiniGridEnv

    env = MiniGridEnv("MiniGrid-FourRooms-v0")
    try:
        assert _rollout_fingerprint(env, 1) != _rollout_fingerprint(env, 2)
    finally:
        env.close()


@pytest.mark.integration
def test_seeding_does_not_make_every_episode_identical():
    """Seeding must fix the *stream*, not collapse it: successive resets have to
    keep producing distinct episodes, otherwise the held-out task-A set would be
    a copy of the training set and PF would be measured on training data."""
    from src.benchmark.protocol import collect_rollouts
    from src.envs.minigrid_env import MiniGridEnv
    from src.utils.buffer import ReplayBuffer

    set_seed(7)
    env = MiniGridEnv("MiniGrid-FourRooms-v0")
    env.seed(7)
    buf = ReplayBuffer(max_episodes=4, seq_len=2)
    collect_rollouts(env, buf, n_rollouts=4, max_steps=8)
    env.close()

    digests = {
        hashlib.sha256(
            np.ascontiguousarray(ep[0]["obs"]).tobytes()
        ).hexdigest()
        for ep in buf.episodes
    }
    assert len(digests) > 1, "every episode started from the same state"


# --- preserve_rng_state ----------------------------------------------------

class TestPreserveRngState:
    """
    Instrumentation added mid-run must not shift the random stream the rest of
    the run draws from (F17). Without that guarantee, a run that measures its
    own task-A convergence is no longer the same run as one that does not.
    """

    def test_torch_stream_continues_unchanged(self):
        set_seed(5)
        expected = torch.randn(8)

        set_seed(5)
        with preserve_rng_state():
            torch.randn(64)
        assert torch.equal(expected, torch.randn(8))

    def test_numpy_stream_continues_unchanged(self):
        set_seed(5)
        expected = np.random.rand(8)

        set_seed(5)
        with preserve_rng_state():
            np.random.rand(64)
        assert np.array_equal(expected, np.random.rand(8))

    def test_python_random_stream_continues_unchanged(self):
        set_seed(5)
        expected = [random.random() for _ in range(4)]

        set_seed(5)
        with preserve_rng_state():
            [random.random() for _ in range(64)]
        assert expected == [random.random() for _ in range(4)]

    def test_restores_even_when_the_body_raises(self):
        set_seed(5)
        expected = torch.randn(8)

        set_seed(5)
        with pytest.raises(RuntimeError):
            with preserve_rng_state():
                torch.randn(64)
                raise RuntimeError("measurement blew up")
        assert torch.equal(expected, torch.randn(8))

    def test_dropout_in_train_mode_is_a_stream_consumer(self):
        """
        The guard is not theoretical: UG-MTM evaluates its gate with dropout
        active, and each of those passes draws from the torch stream. If this
        ever stops being true the guard is still correct, but the reason it
        exists changes.
        """
        layer = torch.nn.Dropout(p=0.5)
        layer.train()
        x = torch.ones(64)

        set_seed(5)
        expected = torch.randn(8)
        set_seed(5)
        layer(x)
        assert not torch.equal(expected, torch.randn(8))
