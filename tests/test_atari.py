"""Tests the ale-py environments are correctly registered."""
import warnings

import gymnasium as gym
import pytest
from ale_py import roms
from ale_py.roms import utils as rom_utils
from gymnasium.envs.registration import registry
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env

from shimmy.utils.envs_configs import ALL_ATARI_GAMES


def test_all_atari_roms():
    """Tests that the static variable ALL_ATARI_GAME is equal to all actual roms."""
    assert ALL_ATARI_GAMES == tuple(map(rom_utils.rom_name_to_id, dir(roms)))


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "Official support for the `seed` function is dropped. Standard practice is to reset gymnasium environments using `env.reset(seed=<desired seed>)`",
        "No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps.",
    ]
]


@pytest.mark.parametrize(
    "env_id",
    [
        env_id
        for env_id, env_spec in registry.items()
        if "Pong" in env_id and env_spec.entry_point == "shimmy.atari_env:AtariEnv"
    ],
)
def test_atari_envs(env_id):
    """Tests the atari envs, as there are 1000 possible environment, we only test the Pong variants."""
    env = gym.make(env_id)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    env.close()

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")
