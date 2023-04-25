"""Tests the multi-agent dm-control soccer environment."""
# pyright: reportUndefinedVariable=false
# flake8: noqa F821
import pickle

import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence

from shimmy.dm_lab_compatibility import DmLabCompatibilityV0

pytest.importorskip("deepmind_lab")
import deepmind_lab

LEVEL_NAMES = [
    "lt_chasm",
    "lt_hallway_slope",
    "lt_horseshoe_color",
    "lt_space_bounce_hard",
    "nav_maze_random_goal_01",
    "nav_maze_random_goal_02",
    "nav_maze_random_goal_03",
    "nav_maze_static_01",
    "nav_maze_static_02",
    "nav_maze_static_03",
    "seekavoid_arena_01",
    "stairway_to_melon",
]

PASSING_LEVEL_NAMES = ["lt_chasm"]


@pytest.mark.parametrize("level_name", PASSING_LEVEL_NAMES)
def test_check_env(level_name):
    """Check that environment pass the gym check_env."""
    observations = ["RGBD"]
    config = {"width": "640", "height": "480", "botCount": "2"}
    renderer = "hardware"

    env = deepmind_lab.Lab(level_name, observations, config=config, renderer=renderer)
    env = DmLabCompatibilityV0(env)

    check_env(env)

    env.close()


@pytest.mark.skip(reason="Seeding tests are not currently possible for DM Lab.")
@pytest.mark.parametrize("level_name", LEVEL_NAMES)
def test_seeding(level_name):
    """Checks that the environment can be properly seeded."""
    observations = ["RGBD"]
    config = {"width": "640", "height": "480", "botCount": "2", "random_seed": "42"}
    renderer = "hardware"

    env_1 = deepmind_lab.Lab(level_name, observations, config=config, renderer=renderer)
    env_1 = DmLabCompatibilityV0(env_1)

    env_2 = deepmind_lab.Lab(level_name, observations, config=config, renderer=renderer)
    env_2 = DmLabCompatibilityV0(env_2)

    obs_1, info_1 = env_1.reset()
    obs_2, info_2 = env_2.reset()
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(100):
        actions = env_1.action_space.sample()
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        # assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()


@pytest.mark.skip(reason="Seeding tests are not currently possible for DM Lab.")
@pytest.mark.parametrize("level_name", LEVEL_NAMES)
def test_pickle(level_name):
    """Checks that the environment can be saved and loaded by pickling."""
    observations = ["RGBD"]
    config = {"width": "640", "height": "480", "botCount": "2", "random_seed": "42"}
    renderer = "hardware"

    env_1 = deepmind_lab.Lab(level_name, observations, config=config, renderer=renderer)
    env_1 = DmLabCompatibilityV0(env_1)

    env_2 = pickle.loads(pickle.dumps(env_1))

    obs_1, info_1 = env_1.reset()
    obs_2, info_2 = env_2.reset()
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(100):
        actions = env_1.action_space.sample()
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        # assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()
