import warnings

import pytest
from ale_py import roms
from gymnasium.envs import registry
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env

from shimmy.utils.envs_configs import ALL_ATARI_GAMES
from ale_py.roms import utils as rom_utils
import gymnasium as gym


def test_all_atari_roms():
    assert ALL_ATARI_GAMES == tuple(map(rom_utils.rom_name_to_id, dir(roms)))


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "Official support for the `seed` function is dropped. Standard practice is to reset gymnasium environments using `env.reset(seed=<desired seed>)`"
    ]
]


@pytest.mark.parametrize("env_id", filter(lambda env_id: "Pong" in env_id and gym.spec(env_id).entry_point == "shimmy.ale_py_env:AtariEnv", registry.keys()))
def test_atari_envs(env_id):
    """Tests the atari envs, as there are 1000 possible environment, we only test the Pong variants.

    Known environments that fail this test - ALE/TicTacToe3D-v5, ALE/VideoChess-v5, ALE/Videocube-v5 + ram variants.
    """
    env = gym.make(env_id)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")
