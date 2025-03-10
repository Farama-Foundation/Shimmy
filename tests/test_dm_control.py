"""Tests the functionality of the DmControlCompatibility Wrapper on dm_control envs."""

import pickle
import warnings
from typing import Callable

import dm_control.suite
import gymnasium as gym
import numpy as np
import pytest
from dm_control import composer
from dm_control.suite.wrappers import (
    action_noise,
    action_scale,
    mujoco_profiling,
    pixels,
)
from gymnasium.envs.registration import registry
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env, data_equivalence

import shimmy
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from shimmy.registration import DM_CONTROL_SUITE_ENVS

gym.register_envs(shimmy)


DM_CONTROL_ENV_IDS = [
    env_id
    for env_id in registry
    if env_id.startswith("dm_control") and env_id != "dm_control/compatibility-env-v0"
]


def test_dm_control_suite_envs():
    """Tests that all DM_CONTROL_ENVS are equal to the known dm-control.suite tasks."""
    assert dm_control.suite.ALL_TASKS == DM_CONTROL_SUITE_ENVS


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "Calling `env.close()` on the closed environment should be allowed, but it raised an exception: _data",
        "Calling `env.close()` on the closed environment should be allowed, but it raised an exception: 'Physics' object has no attribute '_data'",
    ]
]
CHECK_ENV_IGNORE_WARNINGS.append("`in1d` is deprecated. Use `np.isin` instead.")


@pytest.mark.parametrize("env_id", DM_CONTROL_ENV_IDS)
def test_check_env(env_id):
    """Check that environment pass the gymnasium check_env."""
    env = gym.make(env_id, disable_env_checker=True)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_id", DM_CONTROL_ENV_IDS)
def test_seeding(env_id):
    """Test that dm-control seeding works."""
    env_1 = gym.make(env_id)
    env_2 = gym.make(env_id)

    if "lqr" in env_id or (env_1.spec is not None and env_1.spec.nondeterministic):
        # LQR fails this test currently.
        return

    obs_1, info_1 = env_1.reset(seed=42)
    obs_2, info_2 = env_2.reset(seed=42)
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(10):
        actions = env_1.action_space.sample()
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()


@pytest.mark.skip(
    reason="Fatal Python error: Segmentation fault (with or without EzPickle)"
)
@pytest.mark.parametrize("env_id", DM_CONTROL_ENV_IDS[0])
def test_pickle(env_id):
    """Test that dm-control seeding works."""
    env_1 = gym.make(env_id)
    env_2 = pickle.loads(pickle.dumps(env_1))

    if "lqr" in env_id or (env_1.spec is not None and env_1.spec.nondeterministic):
        # LQR fails this test currently.
        return

    obs_1, info_1 = env_1.reset(seed=42)
    obs_2, info_2 = env_2.reset(seed=42)
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(100):
        actions = env_1.action_space.sample()
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()


@pytest.mark.parametrize("camera_id", [-1, 0, 1])
def test_rendering_camera_id(camera_id):
    """Test that dm-control rendering works."""
    env = gym.make(
        DM_CONTROL_ENV_IDS[0],
        render_mode="rgb_array",
        render_kwargs=dict(camera_id=camera_id),
    )
    env.reset()
    frames = []
    for _ in range(10):
        frames.append(env.render())
        env.step(env.action_space.sample())

    env.close()


@pytest.mark.parametrize("height,width", [(84, 84), (48, 48), (128, 128), (100, 200)])
def test_rendering_multiple_cameras(height, width):
    """Test that multi_camera rendering mode works for dm-control environments."""
    env = gym.make(
        DM_CONTROL_ENV_IDS[0],
        render_mode="multi_camera",
        render_kwargs=dict(
            height=height,
            width=width,
        ),
    )
    env.reset()
    frames = []
    for _ in range(10):
        frames.append(env.render())
        env.step(env.action_space.sample())

    env.close()


@pytest.mark.parametrize("height,width", [(84, 84), (48, 48), (128, 128), (100, 200)])
def test_rendering_depth(height, width):
    """Test that depth rendering mode works for dm-control environments."""
    env = gym.make(
        DM_CONTROL_ENV_IDS[0],
        render_mode="depth_array",
        render_kwargs=dict(
            height=height,
            width=width,
        ),
    )
    env.reset()
    frames = []
    for _ in range(10):
        frames.append(env.render())
        env.step(env.action_space.sample())

    env.close()


@pytest.mark.parametrize("height,width", [(84, 84), (48, 48), (128, 128), (100, 200)])
def test_render_height_widths(height, width):
    """Tests that dm-control rendering heights and widths works."""
    env = gym.make(
        DM_CONTROL_ENV_IDS[0],
        render_mode="rgb_array",
        render_kwargs=dict(
            height=height,
            width=width,
        ),
    )
    env.reset()
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (height, width, 3), frame.shape

    env.close()


@pytest.mark.skip(
    reason="This test is currently broken due to an issue with DM control and Gymnasium v29."
)
@pytest.mark.parametrize(
    "wrapper_fn",
    (
        action_noise.Wrapper,
        lambda x: action_scale.Wrapper(x, minimum=0, maximum=1),
        mujoco_profiling.Wrapper,
        pixels.Wrapper,
    ),
    ids=["action noise", "action scale", "mujoco profiling", "pixels"],
)
def test_dm_control_wrappers(
    wrapper_fn: Callable[[composer.Environment], composer.Environment],
):
    """Test the built-in dm-control wrappers."""
    dm_control_env = dm_control.suite.load(*DM_CONTROL_SUITE_ENVS[0])

    if wrapper_fn is action_noise.Wrapper and isinstance(
        dm_control_env, composer.Environment
    ):
        return
    wrapped_env = wrapper_fn(dm_control_env)
    env = DmControlCompatibilityV0(wrapped_env)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env, skip_render_check=True)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")

    env = gym.make(
        "dm_control/compatibility-env-v0", env=wrapped_env, disable_env_checker=True
    )
    check_env(env.unwrapped, skip_render_check=True)
    env.close()
