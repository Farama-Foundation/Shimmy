"""Tests AndroidEnvCompatibilityV0 against a fake dm_env that mimics the AndroidEnv interface.

A real AndroidEnv requires the Android Emulator and a task config, which is impractical
to spin up in CI. These tests instead exercise the spec/observation conversion path
against a stand-in environment with the same dict-based shape.
"""

from __future__ import annotations

from collections import OrderedDict

import dm_env
import gymnasium
import numpy as np
import pytest
from dm_env import specs

pytest.importorskip("android_env")

from shimmy.android_env_compatibility import AndroidEnvCompatibilityV0  # noqa: E402


class _FakeAndroidEnv(dm_env.Environment):
    """A minimal `dm_env.Environment` shaped like AndroidEnv: dict obs, dict action, pixels."""

    def __init__(self, episode_length: int = 3):
        self._episode_length = episode_length
        self._step = 0

    def action_spec(self):
        return OrderedDict(
            [
                (
                    "action_type",
                    specs.DiscreteArray(
                        num_values=3, dtype=np.int32, name="action_type"
                    ),
                ),
                (
                    "touch_position",
                    specs.BoundedArray(
                        shape=(2,),
                        dtype=np.float32,
                        minimum=0.0,
                        maximum=1.0,
                        name="touch_position",
                    ),
                ),
            ]
        )

    def observation_spec(self):
        return OrderedDict(
            [
                ("pixels", specs.Array(shape=(4, 4, 3), dtype=np.uint8, name="pixels")),
                ("timedelta", specs.Array(shape=(), dtype=np.int64, name="timedelta")),
                (
                    "orientation",
                    specs.Array(shape=(4,), dtype=np.uint8, name="orientation"),
                ),
            ]
        )

    def reset(self) -> dm_env.TimeStep:
        self._step = 0
        return dm_env.restart(self._make_obs())

    def step(self, action) -> dm_env.TimeStep:
        self._step += 1
        if self._step >= self._episode_length:
            return dm_env.termination(reward=1.0, observation=self._make_obs())
        return dm_env.transition(reward=0.0, observation=self._make_obs())

    def close(self) -> None:
        pass

    def _make_obs(self) -> OrderedDict:
        return OrderedDict(
            [
                ("pixels", np.zeros((4, 4, 3), dtype=np.uint8)),
                ("timedelta", np.int64(0)),
                ("orientation", np.array([1, 0, 0, 0], dtype=np.uint8)),
            ]
        )


def test_spec_conversion():
    """Dict specs are converted to Gymnasium Dict spaces with the right sub-spaces."""
    env = AndroidEnvCompatibilityV0(_FakeAndroidEnv())

    assert isinstance(env.observation_space, gymnasium.spaces.Dict)
    assert set(env.observation_space.spaces) == {"pixels", "timedelta", "orientation"}
    assert env.observation_space["pixels"].shape == (4, 4, 3)

    assert isinstance(env.action_space, gymnasium.spaces.Dict)
    assert isinstance(env.action_space["action_type"], gymnasium.spaces.Discrete)
    assert isinstance(env.action_space["touch_position"], gymnasium.spaces.Box)

    env.close()


def test_reset_returns_dict_obs():
    """reset() yields a dict observation containing the expected keys."""
    env = AndroidEnvCompatibilityV0(_FakeAndroidEnv())
    obs, info = env.reset(seed=0)

    assert isinstance(obs, dict)
    assert set(obs) == {"pixels", "timedelta", "orientation"}
    assert "timestep.step_type" in info

    env.close()


def test_step_and_termination():
    """The episode runs through to termination after `episode_length` steps."""
    env = AndroidEnvCompatibilityV0(_FakeAndroidEnv(episode_length=3))
    env.reset(seed=0)

    action = {
        "action_type": np.int32(1),
        "touch_position": np.array([0.5, 0.5], dtype=np.float32),
    }

    _, reward, terminated, truncated, _ = env.step(action)
    assert reward == 0.0 and not terminated and not truncated

    _, reward, terminated, truncated, _ = env.step(action)
    assert reward == 0.0 and not terminated and not truncated

    _, reward, terminated, truncated, _ = env.step(action)
    assert reward == 1.0 and terminated and not truncated

    env.close()


def test_render_returns_pixels():
    """rgb_array render mode returns the latest pixels observation."""
    env = AndroidEnvCompatibilityV0(_FakeAndroidEnv(), render_mode="rgb_array")
    env.reset(seed=0)

    frame = env.render()
    assert frame is not None and isinstance(frame, np.ndarray)
    assert frame.shape == (4, 4, 3)
    assert frame.dtype == np.uint8

    env.close()


def test_compatibility_env_registered():
    """The generic compatibility env id is registered on `import shimmy`."""
    from gymnasium.envs.registration import registry

    import shimmy  # noqa: F401

    assert "android_env/compatibility-env-v0" in registry
