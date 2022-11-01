"""Tests the functionality of the DMEnvWrapper on dm_control envs."""
import warnings

import dm_control.suite
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence

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
def test_dm_control_check_env(domain_name, task_name):
    """Check that environment pass the gymnasium check_env."""
    env = gym.make(f"dm_control/{domain_name}-{task_name}-v0")

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning in caught_warnings:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")

    env.close()


@pytest.mark.parametrize("domain_name, task_name", DM_CONTROL_ENVS)
def test_dm_control_seeding(domain_name, task_name):
    """Test that dm-control seeding works."""
    env_1 = gym.make(f"dm_control/{domain_name}-{task_name}-v0")
    env_2 = gym.make(f"dm_control/{domain_name}-{task_name}-v0")

    obs_1, info_1 = env_1.reset(seed=42)
    obs_2, info_2 = env_2.reset(seed=42)
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(100):
        actions = env_1.action_space.sample()
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_1.step(actions)
        assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()


@pytest.mark.parametrize("domain_name, task_name", DM_CONTROL_ENVS)
@pytest.mark.parametrize("camera_id", [0, 1])
def test_dm_control_rendering(domain_name, task_name, camera_id):
    """Test that dm-control rendering works."""
    env = gym.make(
        f"dm_control/{domain_name}-{task_name}-v0",
        render_mode="rgb_array",
        camera_id=camera_id,
    )
    env.reset()
    frames = []
    for _ in range(10):
        frames.append(env.render())
        env.step(env.action_space.sample())

    env.close()
