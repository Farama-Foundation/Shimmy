"""Tests the multi-agent dm-control soccer environment."""
import pickle

import pytest
from dm_control.locomotion import soccer as dm_soccer
from gymnasium.utils.env_checker import data_equivalence
from pettingzoo.test import parallel_api_test

from shimmy.dm_control_multiagent_compatibility import (
    DmControlMultiAgentCompatibilityV0,
)

WALKER_TYPES = [
    dm_soccer.WalkerType.BOXHEAD,
    dm_soccer.WalkerType.ANT,
    dm_soccer.WalkerType.HUMANOID,
]


@pytest.mark.parametrize("walker_type", WALKER_TYPES)
def test_check_env(walker_type):
    """Check that environment pass the pettingzoo check_env."""
    env = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=walker_type,
    )

    env = DmControlMultiAgentCompatibilityV0(env)

    parallel_api_test(env)

    env.close()


@pytest.mark.parametrize("walker_type", WALKER_TYPES)
def test_seeding(walker_type):
    """Tests the seeding of the openspiel conversion wrapper."""
    # load envs
    env1 = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=walker_type,
    )
    env2 = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=walker_type,
    )

    # convert the environment
    env1 = DmControlMultiAgentCompatibilityV0(env1, render_mode=None)
    env2 = DmControlMultiAgentCompatibilityV0(env2, render_mode=None)

    env1.reset(seed=42)
    env2.reset(seed=42)

    for agent in env1.possible_agents:
        env1.action_space(agent).seed(42)
        env2.action_space(agent).seed(42)

    while env1.agents:
        actions1 = {agent: env1.action_space(agent).sample() for agent in env1.agents}
        actions2 = {agent: env2.action_space(agent).sample() for agent in env2.agents}

        assert data_equivalence(actions1, actions2), "Incorrect action seeding"

        obs1, rewards1, terminations1, truncations1, infos1 = env1.step(actions1)
        obs2, rewards2, terminations2, truncations2, infos2 = env2.step(actions2)

        assert not data_equivalence(
            obs1, obs2
        ), "Observations are expected to be slightly different (ball position/velocity)"
        assert data_equivalence(rewards1, rewards2), "Incorrect values for rewards"
        assert data_equivalence(terminations1, terminations2), "Incorrect terminations."
        assert data_equivalence(truncations1, truncations2), "Incorrect truncations"
        assert data_equivalence(infos1, infos2), "Incorrect infos"
    env1.close()
    env2.close()


@pytest.mark.skip(reason="Cannot pickle weakdef objects used in dm_soccer envs.")
@pytest.mark.parametrize("walker_type", WALKER_TYPES)
def test_pickle(walker_type):
    """Tests the seeding of the openspiel conversion wrapper."""
    env1 = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=walker_type,
    )
    env1 = DmControlMultiAgentCompatibilityV0(env1, render_mode=None)
    env2 = pickle.loads(pickle.dumps(env1))

    env1.reset(seed=42)
    env2.reset(seed=42)

    for agent in env1.possible_agents:
        env1.action_space(agent).seed(42)
        env2.action_space(agent).seed(42)

    while env1.agents:
        actions1 = {agent: env1.action_space(agent).sample() for agent in env1.agents}
        actions2 = {agent: env2.action_space(agent).sample() for agent in env2.agents}

        assert data_equivalence(actions1, actions2), "Incorrect action seeding"

        obs1, rewards1, terminations1, truncations1, infos1 = env1.step(actions1)
        obs2, rewards2, terminations2, truncations2, infos2 = env2.step(actions2)

        assert data_equivalence(obs1, obs2), "Incorrect observations"
        assert data_equivalence(rewards1, rewards2), "Incorrect values for rewards"
        assert data_equivalence(terminations1, terminations2), "Incorrect terminations."
        assert data_equivalence(truncations1, truncations2), "Incorrect truncations"
        assert data_equivalence(infos1, infos2), "Incorrect infos"
    env1.close()
    env2.close()
