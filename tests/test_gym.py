"""Tests that gym compatibility environment work as expected."""

import warnings

import gym
import gymnasium
import pytest
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env

CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "This version of the mujoco environments depends on the mujoco-py bindings, which are no longer maintained and may stop working. Please upgrade to the v4 versions of the environments (which depend on the mujoco python bindings instead), unless you are trying to precisely replicate previous works).",
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]

# We do not test Atari environment's here because we check all variants of Pong in test_envs.py (There are too many Atari environments)
CLASSIC_CONTROL_ENVS = [
    env_id
    for env_id, spec in gym.envs.registry.items()  # pyright: ignore[reportGeneralTypeIssues]
    if ("classic_control" in spec.entry_point)
]


@pytest.mark.parametrize(
    "env_id", CLASSIC_CONTROL_ENVS, ids=[env_id for env_id in CLASSIC_CONTROL_ENVS]
)
def test_gym_conversion_by_id(env_id):
    """Tests that the gym conversion works through specifying the env_id."""
    env = gymnasium.make("GymV26Environment-v0", env_id=env_id).unwrapped

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env)

    for warning in caught_warnings:
        if (
            isinstance(warning.message, Warning)
            and warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS
        ):
            raise Error(f"Unexpected warning: {warning.message}")

    env.close()


@pytest.mark.parametrize(
    "env_id", CLASSIC_CONTROL_ENVS, ids=[env_id for env_id in CLASSIC_CONTROL_ENVS]
)
def test_gym_conversion_instantiated(env_id):
    """Tests that the gym conversion works with an instantiated gym environment."""
    env = gym.make(env_id)
    env = gymnasium.make("GymV26Environment-v0", env=env).unwrapped

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env)

    for warning in caught_warnings:
        if (
            isinstance(warning.message, Warning)
            and warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS
        ):
            raise Error(f"Unexpected warning: {warning.message}")

    env.close()
