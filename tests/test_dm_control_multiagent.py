"""Tests the multi-agent dm-control soccer environment."""

import gymnasium
import pytest
from dm_control.locomotion import soccer as dm_soccer
from gymnasium.utils.env_checker import data_equivalence
from pettingzoo.test import parallel_api_test

from shimmy import DmControlMultiAgentCompatibilityV0

DM_CONTROL_MULTIAGENT_ENVS = [1]


@pytest.mark.parametrize("env", DM_CONTROL_MULTIAGENT_ENVS)
def test_check_env(env):
    """Check that environment pass the pettingzoo check_env."""
    env = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.BOXHEAD,
    )

    env = DmControlMultiAgentCompatibilityV0(env)

    parallel_api_test(env)

    env.close()


@pytest.mark.parametrize("env", DM_CONTROL_MULTIAGENT_ENVS)
def test_seeding(env):
    """Test that dm-control seeding works. This fails because for some reason setting random state doesn't do anything."""
    env_1 = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.BOXHEAD,
    )

    env_2 = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        random_state=42,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.BOXHEAD,
    )

    env_1 = DmControlMultiAgentCompatibilityV0(env_1)
    env_2 = DmControlMultiAgentCompatibilityV0(env_2)

    obs_1 = env_1.reset()
    obs_2 = env_2.reset()

    assert data_equivalence(obs_1, obs_2)
    for _ in range(100):
        actions = dict([(a, env_1.action_space(a).sample()) for a in env_1.agents])
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()
