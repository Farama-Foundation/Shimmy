"""Tests the functionality of the DMEnvWrapper on dm_control envs."""
import warnings
from typing import Callable

import dm_control.suite
import dm_env
import gymnasium as gym
import numpy as np
import pytest
from dm_control.suite.wrappers import (
    action_noise,
    action_scale,
    mujoco_profiling,
    pixels,
)
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env, data_equivalence

from shimmy.dm_control_compatibility import DmControlCompatibility
from shimmy.registration import DM_CONTROL_ENVS


def test_dm_control_envs():
    """Tests that all DM_CONTROL_ENVS are equal to the known dm-control.suite tasks."""
    assert dm_control.suite.ALL_TASKS == DM_CONTROL_ENVS


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: ()",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (8, 2)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (2, 4)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (4, 4)",
    ]
]


@pytest.mark.parametrize("domain_name, task_name", DM_CONTROL_ENVS)
def test_check_env(domain_name, task_name):
    """Check that environment pass the gymnasium check_env."""
    env = gym.make(f"dm_control/{domain_name}-{task_name}-v0")

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")

    env.close()


# @pytest.mark.parametrize("domain_name, task_name", DM_CONTROL_ENVS)
# def test_seeding(domain_name, task_name):
#     """Test that dm-control seeding works."""
#     env_1 = gym.make(f"dm_control/{domain_name}-{task_name}-v0")
#     env_2 = gym.make(f"dm_control/{domain_name}-{task_name}-v0")
#
#     if domain_name == "lqr":
#         # LQR fails this test currently.
#         return
#
#     obs_1, info_1 = env_1.reset(seed=42)
#     obs_2, info_2 = env_2.reset(seed=42)
#     assert data_equivalence(obs_1, obs_2)
#     assert data_equivalence(info_1, info_2)
#     for _ in range(100):
#         actions = env_1.action_space.sample()
#         obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
#         obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
#         assert data_equivalence(obs_1, obs_2)
#         assert reward_1 == reward_2
#         assert term_1 == term_2 and trunc_1 == trunc_2
#         assert data_equivalence(info_1, info_2)
#
#     env_1.close()
#     env_2.close()


# @pytest.mark.parametrize("camera_id", [0, 1])
# def test_rendering_camera_id(camera_id):
#     """Test that dm-control rendering works."""
#     domain_name, task_name = DM_CONTROL_ENVS[0]
#     env = gym.make(
#         f"dm_control/{domain_name}-{task_name}-v0",
#         render_mode="rgb_array",
#         camera_id=camera_id,
#     )
#     env.reset()
#     frames = []
#     for _ in range(10):
#         frames.append(env.render())
#         env.step(env.action_space.sample())
#
#     env.close()


@pytest.mark.parametrize("height,width", [(84, 84), (48, 48), (128, 128), (100, 200)])
def test_render_height_widths(height, width):
    """Tests that dm-control rendering heights and widths works."""
    domain_name, task_name = DM_CONTROL_ENVS[0]
    env = gym.make(
        f"dm_control/{domain_name}-{task_name}-v0",
        render_mode="rgb_array",
        render_height=height,
        render_width=width,
    )
    env.reset()
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (height, width, 3), frame.shape


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
    wrapper_fn: Callable[[dm_env.Environment], dm_env.Environment]
):
    """Test the built-in dm-control wrappers."""
    domain_name, task_name = DM_CONTROL_ENVS[0]

    dm_env = dm_control.suite.load(domain_name, task_name)
    wrapped_env = wrapper_fn(dm_env)

    env = DmControlCompatibility(wrapped_env)
    check_env(env)
