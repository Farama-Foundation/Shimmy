"""Tests the multi-agent dm-control soccer environment."""
import pickle

import pytest
from dm_control.locomotion import soccer as dm_soccer
from gymnasium.utils.env_checker import data_equivalence
from pettingzoo.test import parallel_api_test

from shimmy.dm_control_multiagent_compatibility import (
    DmControlMultiAgentCompatibilityV0,
)
from shimmy.utils.dm_control_multiagent import load_dm_control_soccer

TEAM_SIZE = [2, 5, 7]
TIME_LIMITS = [0.0, 10.0]
DISABLE_WALKER_CONTACTS = [True, False]
ENABLE_FIELD_BOX = [True, False]
TERMINATE_ON_GOAL = [True, False]
WALKER_TYPE = [
    dm_soccer.WalkerType.BOXHEAD,
    dm_soccer.WalkerType.ANT,
    dm_soccer.WalkerType.HUMANOID,
]

CONFIGS = []

for val in TEAM_SIZE:
    CONFIGS.append((val, None, None, None, None, None))

for val in TIME_LIMITS:
    CONFIGS.append((None, val, None, None, None, None))

for val in DISABLE_WALKER_CONTACTS:
    CONFIGS.append((None, None, val, None, None, None))

for val in ENABLE_FIELD_BOX:
    CONFIGS.append((None, None, None, val, None, None))

for val in TERMINATE_ON_GOAL:
    CONFIGS.append((None, None, None, None, val, None))

for val in WALKER_TYPE:
    CONFIGS.append((None, None, None, None, None, val))


@pytest.mark.parametrize("config", CONFIGS)
def test_loading_env(config):
    """Tests the loading of all DM Control Soccer environments using the DmControlMultiAgentCompatibility wrapper."""
    team_size, time_limit, disable_walker_contacts, enable_field_box, terminate_on_goal, walker_type = config  # fmt: skip

    env = DmControlMultiAgentCompatibilityV0(
        team_size=team_size,
        time_limit=time_limit,
        disable_walker_contacts=disable_walker_contacts,
        enable_field_box=enable_field_box,
        terminate_on_goal=terminate_on_goal,
        walker_type=walker_type,
        render_mode=None,
    )

    parallel_api_test(env)

    # run through the environment
    env.reset()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs1, rewards1, terminations1, truncations1, infos1 = env.step(actions)
    env.close()


@pytest.mark.parametrize("config", CONFIGS)
def test_existing_env(config):
    """Tests wrapping existing DM Control Soccer environments with the DmControlMultiAgentCompatibility wrapper."""
    team_size, time_limit, disable_walker_contacts, enable_field_box, terminate_on_goal, walker_type = config  # fmt: skip
    env = load_dm_control_soccer(
        team_size,
        time_limit,
        disable_walker_contacts,
        enable_field_box,
        terminate_on_goal,
        walker_type,
    )

    env = DmControlMultiAgentCompatibilityV0(env)

    parallel_api_test(env)

    env.close()


@pytest.mark.parametrize("config", CONFIGS)
def test_seeding(config):
    """Tests the seeding of the DmControlMultiAgentCompatibility wrapper."""
    team_size, time_limit, disable_walker_contacts, enable_field_box, terminate_on_goal, walker_type = config  # fmt: skip

    env1 = load_dm_control_soccer(
        team_size,
        time_limit,
        disable_walker_contacts,
        enable_field_box,
        terminate_on_goal,
        walker_type,
    )
    env2 = load_dm_control_soccer(
        team_size,
        time_limit,
        disable_walker_contacts,
        enable_field_box,
        terminate_on_goal,
        walker_type,
    )

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

        assert data_equivalence(obs1, obs2)
        assert data_equivalence(rewards1, rewards2), "Incorrect values for rewards"
        assert data_equivalence(terminations1, terminations2), "Incorrect terminations."
        assert data_equivalence(truncations1, truncations2), "Incorrect truncations"
        assert data_equivalence(infos1, infos2), "Incorrect infos"
    env1.close()
    env2.close()


@pytest.mark.skip(reason="Cannot pickle weakref objects used in dm_soccer envs.")
@pytest.mark.parametrize("config", CONFIGS)
def test_pickle(config):
    """Tests that environments using the DmControlMultiAgentCompatibility wrapper can be serialized and deserialized via pickle."""
    team_size, time_limit, disable_walker_contacts, enable_field_box, terminate_on_goal, walker_type = config  # fmt: skip
    env1 = load_dm_control_soccer(
        team_size,
        time_limit,
        disable_walker_contacts,
        enable_field_box,
        terminate_on_goal,
        walker_type,
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
