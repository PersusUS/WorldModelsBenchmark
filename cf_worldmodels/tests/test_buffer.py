"""Tests for src/utils/buffer.py (ReplayBuffer)."""
import numpy as np
import pytest
import torch

from src.utils.buffer import ReplayBuffer

from conftest import ACTION_DIM


def make_episode(length: int):
    return [
        {
            "obs": np.random.rand(64, 64, 3).astype(np.float32),
            "action": np.random.rand(ACTION_DIM).astype(np.float32),
            "reward": float(i),
            "done": i == length - 1,
        }
        for i in range(length)
    ]


def test_starts_empty():
    assert len(ReplayBuffer(max_episodes=4, seq_len=5)) == 0


def test_accepts_episode_at_least_seq_len_long():
    buf = ReplayBuffer(max_episodes=4, seq_len=5)
    buf.add_episode(make_episode(5))
    assert len(buf) == 1


def test_rejects_episode_shorter_than_seq_len():
    buf = ReplayBuffer(max_episodes=4, seq_len=5)
    buf.add_episode(make_episode(4))
    assert len(buf) == 0


def test_evicts_oldest_episode_when_full():
    buf = ReplayBuffer(max_episodes=2, seq_len=2)
    first, second, third = make_episode(3), make_episode(3), make_episode(3)
    buf.add_episode(first)
    buf.add_episode(second)
    buf.add_episode(third)

    assert len(buf) == 2
    assert buf.episodes[0] is second
    assert buf.episodes[1] is third


def test_sample_returns_expected_shapes():
    buf = ReplayBuffer(max_episodes=4, seq_len=3)
    for _ in range(4):
        buf.add_episode(make_episode(10))

    batch = buf.sample(batch_size=5, seq_len=3)
    assert batch["obs"].shape == (5, 3, 64, 64, 3)
    assert batch["actions"].shape == (5, 3, ACTION_DIM)
    assert batch["rewards"].shape == (5, 3)
    assert batch["dones"].shape == (5, 3)


def test_sample_returns_float32_tensors():
    buf = ReplayBuffer(max_episodes=2, seq_len=3)
    buf.add_episode(make_episode(10))

    batch = buf.sample(batch_size=2, seq_len=3)
    for key in ["obs", "actions", "rewards", "dones"]:
        assert isinstance(batch[key], torch.Tensor)
        assert batch[key].dtype == torch.float32


def test_sample_observations_stay_in_unit_interval():
    buf = ReplayBuffer(max_episodes=2, seq_len=3)
    buf.add_episode(make_episode(10))

    obs = buf.sample(batch_size=2, seq_len=3)["obs"]
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_sample_window_equal_to_episode_length_is_valid():
    """max_start == 0 must still produce a valid window, not an empty range."""
    buf = ReplayBuffer(max_episodes=2, seq_len=5)
    buf.add_episode(make_episode(5))
    batch = buf.sample(batch_size=3, seq_len=5)
    assert batch["obs"].shape == (3, 5, 64, 64, 3)


def test_sample_can_oversample_a_small_buffer():
    """batch_size may exceed the number of stored episodes (sampling is with
    replacement)."""
    buf = ReplayBuffer(max_episodes=4, seq_len=3)
    buf.add_episode(make_episode(6))
    batch = buf.sample(batch_size=8, seq_len=3)
    assert batch["obs"].shape[0] == 8


def test_sample_from_empty_buffer_raises():
    buf = ReplayBuffer(max_episodes=4, seq_len=3)
    with pytest.raises(ValueError):
        buf.sample(batch_size=2, seq_len=3)
