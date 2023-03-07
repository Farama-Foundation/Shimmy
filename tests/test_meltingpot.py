"""Tests the functionality of the MeltingPotCompatibility wrapper on meltingpot substrates."""
import gymnasium.utils.env_checker
import numpy as np
import pytest
from meltingpot.python.configs.substrates import SUBSTRATES
from pettingzoo.test import parallel_api_test

from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_passing_substrates(substrate_name):
    """Tests the conversion of all melting pot envs."""
    env = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode=None)

    # api test the env
    parallel_api_test(env)

    env.reset()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)


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
    a_space2 = env2.action_space(env1.agents[0])
    a_space2.seed(42)

    while env1.agents:
        actions1 = {agent: env1.action_space(agent).sample() for agent in env1.agents}
        actions2 = {agent: env2.action_space(agent).sample() for agent in env2.agents}

        assert gymnasium.utils.env_checker.data_equivalence(
            env1.step(actions1), env2.step(actions2)
        ), "Incorrect returns on iteration."


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_rendering(substrate_name):
    """Tests rendering for all melting pot envs (using pygame)."""
    env = MeltingPotCompatibilityV0(substrate_name=substrate_name, render_mode="human")

    env.reset()
    for _ in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        env.render()
