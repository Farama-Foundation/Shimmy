"""Tests the functionality of the MeltingPotCompatibility wrapper on meltingpot substrates."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import data_equivalence
from meltingpot.python.configs.substrates import SUBSTRATES
from pettingzoo.test import parallel_api_test

from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0


@pytest.mark.skip(
    reason="Melting Pot environments are stochastic and do not currently support seeding."
)
@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_seeding(substrate_name):
    """Tests the seeding of the melting pot conversion wrapper."""
    # load and convert the envs
    env1 = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode=None)
    env2 = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode=None)

    env1.reset(seed=42)
    env2.reset(seed=42)

    a_space1 = env1.action_space(env1.agents[0])
    a_space1.seed(42)
    a_space2 = env2.action_space(env2.agents[0])
    a_space2.seed(42)

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


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_passing_substrates(substrate_name):
    """Tests the conversion of all melting pot envs."""
    env = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode=None)

    # api test the env
    parallel_api_test(env)

    env.reset(seed=0)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_rendering(substrate_name):
    """Tests rendering for all melting pot envs (using pygame)."""
    env = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode="human")

    env.reset()
    for _ in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        env.render()
