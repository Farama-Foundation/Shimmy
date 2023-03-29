"""Tests the multi-agent dm-control soccer environment."""

import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence

from shimmy.dm_lab_compatibility import DmLabCompatibilityV0


# TODO: check for warnings
# @pytest.mark.skip(reason="no way of currently testing this")
def test_check_env():
    """Check that environment pass the gym check_env."""
    import deepmind_lab  # noqa: E402

    observations = ["RGBD"]
    config = {"width": "640", "height": "480", "botCount": "2"}
    renderer = "hardware"

    env = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
    env = DmLabCompatibilityV0(env)

    check_env(env)

    env.close()


#
# def test_seeding():
#     """Checks that the environment can be properly seeded."""
#     observations = ["RGBD"]
#     config = {"width": "640", "height": "480", "botCount": "2"}
#     renderer = "hardware"
#
#     env_1 = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
#     env_1 = DmLabCompatibilityV0(env_1)
#
#     env_2 = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
#     env_2 = DmLabCompatibilityV0(env_2)
#
#     obs_1, info_1 = env_1.reset(seed=42)
#     obs_2, info_2 = env_2.reset(seed=42)
#     assert data_equivalence(obs_1, obs_2)
#     assert data_equivalence(info_1, info_2)
#     for _ in range(100):
#         actions = int(env_1.action_space.sample())
#         obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
#         obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
#         assert data_equivalence(obs_1, obs_2)
#         assert reward_1 == reward_2
#         assert term_1 == term_2 and trunc_1 == trunc_2
#         assert data_equivalence(info_1, info_2)
#
#     env_1.close()
#     env_2.close()
#
#
# def test_pickle():
#     """Checks that the environment can be saved and loaded by pickling."""
#     observations = ["RGBD"]
#     config = {"width": "640", "height": "480", "botCount": "2"}
#     renderer = "hardware"
#
#     env_1 = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
#     env_1 = DmLabCompatibilityV0(env_1)
#
#     env_2 = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
#     env_2 = DmLabCompatibilityV0(env_2)
#
#     obs_1, info_1 = env_1.reset(seed=42)
#     obs_2, info_2 = env_2.reset(seed=42)
#     assert data_equivalence(obs_1, obs_2)
#     assert data_equivalence(info_1, info_2)
#     for _ in range(100):
#         actions = int(env_1.action_space.sample())
#         obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
#         obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
#         assert data_equivalence(obs_1, obs_2)
#         assert reward_1 == reward_2
#         assert term_1 == term_2 and trunc_1 == trunc_2
#         assert data_equivalence(info_1, info_2)
#
#     env_1.close()
#     env_2.close()
