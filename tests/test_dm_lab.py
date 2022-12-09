"""Tests the multi-agent dm-control soccer environment."""

import gymnasium
from gymnasium.utils.env_checker import check_env

from shimmy.dm_lab_compatibility import DmLabCompatibilityV0

import deepmind_lab

def test_check_env():
    """Check that environment pass the gym check_env."""
    observations = ["RGBD"]
    config = {"width": "640", "height": "480", "botCount": "2"}
    renderer = "hardware"

    env = deepmind_lab.Lab('lt_chasm', observations, config=config, renderer=renderer)

    env = DmLabCompatibilityV0(env)

    check_env(env)

    env.close()
