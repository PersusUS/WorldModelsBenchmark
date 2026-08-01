"""Integration tests for the environment wrappers and the rollout protocol.

These construct real simulators (MiniGrid, MuJoCo, dm_control) and are slower
than the unit tests. Skip them with `pytest -m "not integration"`.
"""
import numpy as np
import pytest

from src.envs.base_env import BaseEnv
from src.utils.buffer import ReplayBuffer

pytestmark = pytest.mark.integration

MINIGRID_ID = "MiniGrid-Empty-5x5-v0"
MUJOCO_ID = "HalfCheetah-v4"


def assert_valid_observation(obs):
    """Every wrapper must honour the (64, 64, 3) float32 [0, 1] contract."""
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


# --- MiniGrid --------------------------------------------------------------

@pytest.fixture(scope="module")
def minigrid_env():
    from src.envs.minigrid_env import MiniGridEnv

    env = MiniGridEnv(MINIGRID_ID)
    yield env
    env.close()


def test_minigrid_is_a_base_env(minigrid_env):
    assert isinstance(minigrid_env, BaseEnv)


def test_minigrid_reset_observation(minigrid_env):
    assert_valid_observation(minigrid_env.reset())


def test_minigrid_step_observation(minigrid_env):
    minigrid_env.reset()
    obs, reward, done, info = minigrid_env.step(minigrid_env.sample_action())
    assert_valid_observation(obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_minigrid_obs_shape_property(minigrid_env):
    assert minigrid_env.obs_shape == (64, 64, 3)


def test_minigrid_action_is_one_hot(minigrid_env):
    action = minigrid_env.sample_action()
    assert action.shape == (minigrid_env.action_dim,)
    assert action.dtype == np.float32
    assert action.sum() == pytest.approx(1.0)
    assert set(np.unique(action)) <= {0.0, 1.0}


def test_minigrid_action_dim_matches_config():
    """configs/benchmark/minigrid.yaml hardcodes action_dim: 7."""
    from omegaconf import OmegaConf

    from src.envs.minigrid_env import MiniGridEnv

    cfg = OmegaConf.load("configs/benchmark/minigrid.yaml")
    for level in ["distance_min", "distance_med", "distance_max"]:
        seq = cfg.benchmark.sequences[level]
        for task in [seq.task_A, seq.task_B]:
            env = MiniGridEnv(task.env_id)
            try:
                assert env.action_dim == cfg.model.action_dim
            finally:
                env.close()


def test_minigrid_accepts_one_hot_and_scalar_actions(minigrid_env):
    minigrid_env.reset()
    minigrid_env.step(minigrid_env.sample_action())
    minigrid_env.step(np.int64(0))


# --- Gymnasium / MuJoCo ----------------------------------------------------

@pytest.fixture(scope="module")
def gym_env():
    from src.envs.gymnasium_env import GymnasiumEnv

    env = GymnasiumEnv(MUJOCO_ID)
    yield env
    env.close()


def test_gymnasium_reset_and_step_observations(gym_env):
    assert_valid_observation(gym_env.reset())
    obs, reward, done, _ = gym_env.step(gym_env.sample_action())
    assert_valid_observation(obs)
    assert isinstance(reward, float)


def test_gymnasium_action_is_continuous(gym_env):
    action = gym_env.sample_action()
    assert action.shape == (gym_env.action_dim,)
    assert action.dtype == np.float32


def test_gymnasium_action_dim_matches_config(gym_env):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/benchmark/gymnasium.yaml")
    assert gym_env.action_dim == cfg.model.action_dim


def test_gymnasium_applies_gravity_override():
    """The whole Gymnasium family varies only physics, so the override has to
    reach the MuJoCo model."""
    from src.envs.gymnasium_env import GymnasiumEnv

    env = GymnasiumEnv(MUJOCO_ID, physics_params={"gravity": 4.0})
    try:
        assert env._env.unwrapped.model.opt.gravity[2] == pytest.approx(-4.0)
    finally:
        env.close()


def test_gymnasium_applies_mass_and_friction_scaling():
    from src.envs.gymnasium_env import GymnasiumEnv

    reference = GymnasiumEnv(MUJOCO_ID)
    scaled = GymnasiumEnv(
        MUJOCO_ID, physics_params={"mass_scale": 3.0, "friction_scale": 0.5}
    )
    try:
        assert np.allclose(
            scaled._env.unwrapped.model.body_mass,
            reference._env.unwrapped.model.body_mass * 3.0,
        )
        assert np.allclose(
            scaled._env.unwrapped.model.geom_friction,
            reference._env.unwrapped.model.geom_friction * 0.5,
        )
    finally:
        reference.close()
        scaled.close()


def test_gymnasium_default_physics_are_untouched():
    from src.envs.gymnasium_env import GymnasiumEnv

    reference = GymnasiumEnv(MUJOCO_ID)
    explicit = GymnasiumEnv(
        MUJOCO_ID,
        physics_params={"gravity": 9.8, "mass_scale": 1.0,
                        "friction_scale": 1.0},
    )
    try:
        assert np.allclose(
            reference._env.unwrapped.model.body_mass,
            explicit._env.unwrapped.model.body_mass,
        )
    finally:
        reference.close()
        explicit.close()


# --- dm_control ------------------------------------------------------------

@pytest.fixture(scope="module")
def dmc_env():
    from src.envs.dmcontrol_env import DMControlEnv

    env = DMControlEnv("cheetah", "run")
    yield env
    env.close()


def test_dmcontrol_reset_and_step_observations(dmc_env):
    assert_valid_observation(dmc_env.reset())
    obs, reward, done, _ = dmc_env.step(dmc_env.sample_action())
    assert_valid_observation(obs)
    assert isinstance(reward, float)


def test_dmcontrol_clips_out_of_range_actions(dmc_env):
    """dm_control raises on out-of-spec actions, so the wrapper must clip."""
    dmc_env.reset()
    obs, _, _, _ = dmc_env.step(np.full(dmc_env.action_dim, 1e3))
    assert_valid_observation(obs)


def test_dmcontrol_sampled_actions_are_within_spec(dmc_env):
    action = dmc_env.sample_action()
    spec = dmc_env._action_spec
    assert np.all(action >= spec.minimum)
    assert np.all(action <= spec.maximum)


def test_dmcontrol_applies_physics_perturbations():
    """F8: the physics block used to be dropped on the floor for this family."""
    from src.envs.dmcontrol_env import DMControlEnv

    env = DMControlEnv("cheetah", "run",
                       physics_params={"gravity": 4.0, "mass_scale": 2.0})
    try:
        model = env._env.physics.model
        assert model.opt.gravity[2] == pytest.approx(-4.0)
        baseline = DMControlEnv("cheetah", "run")
        try:
            assert np.allclose(model.body_mass,
                               2.0 * baseline._env.physics.model.body_mass)
        finally:
            baseline.close()
    finally:
        env.close()


def test_dmcontrol_rejects_a_perturbation_it_cannot_apply():
    """`lateral_wind: true` sat in the config for the whole project doing
    nothing. MuJoCo's opt.wind needs a non-zero fluid density to have any
    effect, so the wrapper refuses keys it does not implement."""
    from src.envs.dmcontrol_env import DMControlEnv

    with pytest.raises(ValueError, match="lateral_wind"):
        DMControlEnv("cheetah", "run", physics_params={"lateral_wind": True})


def test_dmcontrol_first_reward_is_a_float_not_none(dmc_env):
    """timestep.reward is None on the first step; the wrapper must map it to 0."""
    dmc_env.reset()
    _, reward, _, _ = dmc_env.step(dmc_env.sample_action())
    assert reward is not None
    assert isinstance(reward, float)


# --- rollout protocol ------------------------------------------------------

def test_collect_rollouts_fills_the_buffer(minigrid_env):
    from src.benchmark.protocol import collect_rollouts

    buffer = ReplayBuffer(max_episodes=4, seq_len=2)
    collect_rollouts(minigrid_env, buffer, n_rollouts=2, max_steps=10)

    assert len(buffer) > 0
    batch = buffer.sample(batch_size=2, seq_len=2)
    assert batch["obs"].shape == (2, 2, 64, 64, 3)
    assert batch["actions"].shape == (2, 2, minigrid_env.action_dim)


def test_collect_rollouts_respects_max_steps(minigrid_env):
    from src.benchmark.protocol import collect_rollouts

    buffer = ReplayBuffer(max_episodes=2, seq_len=1)
    collect_rollouts(minigrid_env, buffer, n_rollouts=1, max_steps=3)
    assert all(len(ep) <= 3 for ep in buffer.episodes)


def test_collect_rollouts_marks_the_final_transition_done(minigrid_env):
    from src.benchmark.protocol import collect_rollouts

    buffer = ReplayBuffer(max_episodes=2, seq_len=1)
    collect_rollouts(minigrid_env, buffer, n_rollouts=1, max_steps=5)
    assert buffer.episodes[0][-1]["done"] is True


# --- the two tasks of a pair must actually differ (F26) ---------------------

@pytest.mark.integration
@pytest.mark.parametrize("family", ["minigrid", "gymnasium", "dmcontrol"])
def test_each_distance_pair_produces_different_data(family):
    """Comparing the config dicts is not enough, and F26 is why.

    dmcontrol's distance_max was set to walker/stand against distance_med's
    walker/run. Different task names, different dicts -- and bit-identical
    data, because dm_control's tasks share a domain's physics and initial-state
    distribution and differ only in the reward function, which a reward-free
    world model trained on random rollouts never sees.

    So the check has to be on the observations: same seed, same actions, and
    the two tasks must still diverge.
    """
    from omegaconf import OmegaConf

    from experiments.run_full_benchmark import FAMILY_CONFIGS, create_env_pair

    import mujoco

    cfg = OmegaConf.load(FAMILY_CONFIGS[family])
    for level in ["distance_min", "distance_med", "distance_max"]:
        env_A, env_B = create_env_pair(family, cfg.benchmark.sequences[level])
        try:
            frames = []
            for env in (env_A, env_B):
                env.seed(0)
                obs = env.reset()
                action = np.zeros(env.action_dim, dtype=np.float32)
                for _ in range(5):
                    obs, _r, done, _i = env.step(action)
                    if done:
                        break
                frames.append(obs)
            assert not np.array_equal(frames[0], frames[1]), (
                f"{family}/{level}: both tasks produce identical observations"
            )
        except mujoco.FatalError as exc:
            # F24/I20: dm_control cannot build a render context in a process
            # where MuJoCo already has one, which is why the runner gives each
            # family its own process. Covered on its own with:
            #   pytest tests/test_envs.py -k "different_data and dmcontrol"
            pytest.skip(f"render context already taken (F24): {exc}")
        finally:
            env_A.close()
            env_B.close()
