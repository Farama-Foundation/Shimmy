"""Tests the functionality of the MeltingPotCompatibility wrapper on meltingpot substrates."""

import numpy as np
import pytest

from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0
from meltingpot.python.configs.substrates import SUBSTRATES
from pettingzoo.test import parallel_api_test


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_passing_substrates(substrate_name):
    """Tests the conversion of all openspiel envs."""
    for _ in range(5):
        env = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode=None)

        # api test the env
        parallel_api_test(env)

        env.reset()
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_seeding(substrate_name):
    """Tests the seeding of the openspiel conversion wrapper."""
    # load and convert the envs
    env1 = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode=None)
    env2 = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode=None)

    env1.reset(seed=42)
    env2.reset(seed=42)

    a_space1 = env1.action_space(env1.agents[0])
    a_space1.seed(42)
    a_space2 = env2.action_space(env1.agents[0])
    a_space2.seed(42)

    while env1.agents:
        actions1 = {agent: env1.action_space(agent).sample() for agent in env1.agents}
        actions2 = {agent: env2.action_space(agent).sample() for agent in env2.agents}

        observations1, rewards1, terminations1, truncations1, infos1 = env1.step(actions1)
        observations2, rewards2, terminations2, truncations2, infos2 = env2.step(actions2)

        returns1 = (observations1, rewards1, terminations1, truncations1)
        returns2 = (observations2, rewards2, terminations2, truncations2)

        for stuff1, stuff2 in zip(returns1, returns2):
            if isinstance(stuff1, bool):
                assert stuff1 == stuff2, "Incorrect returns on iteration."
            elif isinstance(stuff1, np.ndarray):
                assert (stuff1 == stuff2).all(), "Incorrect returns on iteration."
            elif isinstance(stuff1, str):
                assert stuff1 == stuff2, "Incorrect returns on iteration."
