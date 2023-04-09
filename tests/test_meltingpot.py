"""Tests the functionality of the MeltingPotCompatibility wrapper on meltingpot substrates."""
# pyright: reportUndefinedVariable=false
# flake8: noqa F821 E402
# isort: skip_file
import pickle

import pytest
from gymnasium.utils.env_checker import data_equivalence
from pettingzoo.test import parallel_api_test

pytest.importorskip("meltingpot")

import meltingpot.python
from meltingpot.python.configs.substrates import SUBSTRATES

from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_substrate


@pytest.mark.skip(
    reason="Melting Pot environments are stochastic and do not currently support seeding."
)
@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_seeding(substrate_name):
    """Tests the seeding of the melting pot conversion wrapper."""
    env1 = load_substrate(substrate_name)
    env1 = MeltingPotCompatibilityV0(env1, render_mode=None)

    env2 = load_substrate(substrate_name)
    env2 = MeltingPotCompatibilityV0(env2, render_mode=None)

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


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_substrate(substrate_name):
    """Tests the conversion of all melting pot envs, loaded from substrate name."""
    env = load_substrate(substrate_name)
    env = MeltingPotCompatibilityV0(env, render_mode=None)

    # api test the env
    parallel_api_test(env)

    env.reset()
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()


def test_custom_substrate():
    """Tests the conversion of melting pot substrates which have already been loaded (supporting custom envs)."""
    # Take the first element of the frozen set of substrate names
    CUSTOM_SUBSTRATE, *_ = SUBSTRATES

    env = load_substrate(CUSTOM_SUBSTRATE)

    # Test that the already created environment can be converted to pettingzoo
    env = MeltingPotCompatibilityV0(env, render_mode="None")

    env.reset()
    for _ in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        env.render()
    env.close()


@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_rendering(substrate_name):
    """Tests rendering for all melting pot envs (using pygame)."""
    env = load_substrate(substrate_name)
    env = MeltingPotCompatibilityV0(env, render_mode="human")

    env.reset()
    for _ in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)
        env.render()


@pytest.mark.skip(
    reason="Melting Pot environments are stochastic and do not currently support seeding."
)
@pytest.mark.parametrize("substrate_name", SUBSTRATES)
def test_pickle(substrate_name):
    """Test that environments can be saved and loaded with pickle."""
    # load and convert the envs
    env = load_substrate(substrate_name)
    env1 = MeltingPotCompatibilityV0(env, render_mode=None)
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
