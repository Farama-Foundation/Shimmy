import random

import numpy as np
import pyspiel
import pytest

from shimmy import OpenspielWrapper


@pytest.mark.parametrize("game", pyspiel.registered_names())
def test_all_games(game):
    """Tests the conversion of all openspiel envs."""
    env = OpenspielWrapper(game=game, render_mode=None)

    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample(mask=info["action_mask"])
        env.step(action)


def test_seeding():
    """Tests the seeding of the openspiel conversion wrapper."""
    # load envs
    env1 = pyspiel.load_game("2048")
    env2 = pyspiel.load_game("2048")

    # convert the environment
    env1 = OpenspielWrapper(env1, render_mode=None)
    env2 = OpenspielWrapper(env2, render_mode=None)
    env1.reset(seed=42)
    env2.reset(seed=42)

    agent1 = env1.agent_iter()
    agent2 = env2.agent_iter()

    for agent1, agent2 in zip(env1.agent_iter(), env2.agent_iter()):
        obs1, rew1, term1, trunc1, info1 = env1.last()
        obs2, rew2, term2, trunc2, info2 = env2.last()

        returns1 = (obs1, rew1, term1, trunc1)
        returns2 = (obs2, rew2, term2, trunc2)

        act1 = env1.action_space(agent1).sample(mask=info1["action_mask"])
        act2 = env1.action_space(agent2).sample(mask=info2["action_mask"])

        env1.step(act1)
        env2.step(act2)

        for stuff1, stuff2 in zip(returns1, returns2):
            if isinstance(stuff1, bool):
                assert stuff1 == stuff2, "Incorrect returns on iteration."
            elif isinstance(stuff1, np.ndarray):
                assert (stuff1 == stuff2).all(), "Incorrect returns on iteration."
