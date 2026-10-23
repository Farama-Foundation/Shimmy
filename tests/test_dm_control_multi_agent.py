"""Tests the multi-agent dm-control soccer environment."""

import pickle

import dm_env
import numpy as np
import pytest
from dm_control.locomotion import soccer as dm_soccer
from gymnasium.utils.env_checker import data_equivalence
from pettingzoo.test import parallel_api_test

from shimmy.dm_control_multiagent_compatibility import (
    DmControlMultiAgentCompatibilityV0,
    _unravel_ma_timestep,
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
        env1.observation_space(agent).seed(42)
        env2.observation_space(agent).seed(42)

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


def test_action_order_independent():
    """Tests that actions are applied to the matching agents regardless of dict order."""
    config = CONFIGS[0]
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
    agents = env1.agents
    actions = {agent: env1.action_space(agent).sample() for agent in agents}
    actions[agents[1]] = actions[agents[1]] * 0.5

    obs1, rewards1, _, _, _ = env1.step(actions)
    reversed_actions = dict(reversed(list(actions.items())))
    obs2, rewards2, _, _, _ = env2.step(reversed_actions)

    assert data_equivalence(obs1, obs2), "Incorrect observations"
    assert data_equivalence(rewards1, rewards2), "Incorrect values for rewards"
    env1.close()
    env2.close()


@pytest.mark.parametrize(
    "make_timestep, terminated, truncated",
    [(dm_env.termination, True, False), (dm_env.truncation, False, True)],
)
def test_last_timestep_terminates_or_truncates(make_timestep, terminated, truncated):
    """Tests that a final discount of 0 terminates the episode and a positive one truncates it, as in dm_env."""
    agents = ["player_0", "player_1"]
    timestep = make_timestep(reward=[0.0, 0.0], observation=[{}, {}])

    _, _, terminations, truncations, _ = _unravel_ma_timestep(timestep, agents)

    assert terminations == {agent: terminated for agent in agents}
    assert truncations == {agent: truncated for agent in agents}


def test_time_limit_truncates():
    """Tests that running out of time truncates the episode (dm_control ends it with discount 1)."""
    env = DmControlMultiAgentCompatibilityV0(
        team_size=2, time_limit=1.0, terminate_on_goal=False, render_mode=None
    )
    env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)

    assert all(truncations.values())
    assert not any(terminations.values())
    env.close()


def test_goal_terminates():
    """Tests that a goal with `terminate_on_goal` terminates the episode (dm_control ends it with discount 0)."""
    soccer_env = load_dm_control_soccer(2, 10.0, None, None, True, None)
    env = DmControlMultiAgentCompatibilityV0(soccer_env, render_mode=None)
    env.reset(seed=42)

    # Keep the ball just behind a goal line until the arena detects a goal.
    behind_goal_line = [-(soccer_env.task.arena.size[0] + 0.5), 0.0, 0.5]
    for _ in range(100):
        soccer_env.task.ball.set_pose(soccer_env.physics, position=behind_goal_line)
        actions = {
            agent: np.zeros_like(env.action_space(agent).low) for agent in env.agents
        }
        _, _, terminations, truncations, _ = env.step(actions)
        if not env.agents:
            break

    assert soccer_env.task.arena.detected_goal() is not None
    assert all(terminations.values())
    assert not any(truncations.values())
    env.close()


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
