"""Tests the functionality of the OpenSpielCompatibility wrapper on OpenSpiel envs."""
import pickle

import numpy as np
import pyspiel
import pytest
from gymnasium.utils.env_checker import data_equivalence

from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

# todo add api_test however chess causes a OOM error
# from pettingzoo.test import api_test


_PASSING_GAMES = [
    "2048",
    "amazons",
    "backgammon",
    "bargaining",
    "battleship",
    "blackjack",
    "blotto",
    "breakthrough",
    "bridge",
    "bridge_uncontested_bidding",
    "catch",
    "checkers",
    "chess",
    "cliff_walking",
    "clobber",
    "coin_game",
    "colored_trails",
    "connect_four",
    "coop_box_pushing",
    "coop_to_1p",
    "coordinated_mp",
    "cursor_go",
    "dark_chess",
    "dark_hex",
    "dark_hex_ir",
    "deep_sea",
    "euchre",
    "first_sealed_auction",
    "gin_rummy",
    "go",
    "goofspiel",
    "hanabi",
    "havannah",
    "hearts",
    "hex",
    "kriegspiel",
    "kuhn_poker",
    "laser_tag",
    "leduc_poker",
    "lewis_signaling",
    "liars_dice",
    "liars_dice_ir",
    "mancala",
    "markov_soccer",
    "matching_pennies_3p",
    "matrix_cd",
    "matrix_coordination",
    "matrix_mp",
    "matrix_pd",
    "matrix_rps",
    "matrix_rpsw",
    "matrix_sh",
    "matrix_shapleys_game",
    "mfg_crowd_modelling",
    "mfg_crowd_modelling_2d",
    "mfg_garnet",
    "morpion_solitaire",
    "negotiation",
    "nim",
    "oh_hell",
    "oshi_zumo",
    "othello",
    "oware",
    "pathfinding",
    "pentago",
    "phantom_go",
    "phantom_ttt",
    "phantom_ttt_ir",
    "pig",
    "quoridor",
    "rbc",
    "sheriff",
    "skat",
    "solitaire",
    "stones_and_gems",
    "tarok",
    "tic_tac_toe",
    "tiny_bridge_2p",
    "tiny_bridge_4p",
    "tiny_hanabi",
    "trade_comm",
    "ultimate_tic_tac_toe",
    "universal_poker",
    "y",
    "mfg_dynamic_routing",
]

_FAILING_GAMES = [
    "efg_game",
    "misere",
    "normal_form_extensive_game",
    "repeated_game",
    "restricted_nash_response",
    "start_at",
    "turn_based_simultaneous_game",
]

_UNKNOWN_BUGS_GAMES = ["nfg_game"]


@pytest.mark.parametrize("game_name", _PASSING_GAMES)
def test_passing_games(game_name):
    """Tests the conversion of all OpenSpiel environments using the OpenSpielCompatibility wrapper."""
    for _ in range(5):
        env = pyspiel.load_game(game_name)
        env = OpenSpielCompatibilityV0(env=env, render_mode=None)

        # api test the env (disabled because some environments fail the test)
        # api_test(env)

        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            action = env.action_space(agent).sample(mask=info["action_mask"])
            env.step(action)


@pytest.mark.parametrize("game_name", _FAILING_GAMES)
def test_failing_games(game_name):
    """Ensures that failing OpenSpiel games are still failing."""
    with pytest.raises(pyspiel.SpielError):
        test_passing_games(game_name)


@pytest.mark.parametrize("game_name", _PASSING_GAMES)
def test_loading_env(game_name):
    """Tests the loading of all OpenSpiel environments using the OpenSpielCompatibility wrapper."""
    env = OpenSpielCompatibilityV0(game_name=game_name, render_mode=None)

    # api test the env
    # api_test(env)

    # run through the environment
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample(mask=info["action_mask"])
        env.step(action)
    env.close()


@pytest.mark.parametrize("game_name", _PASSING_GAMES)
def test_seeding(game_name):
    """Tests the seeding of the OpenSpielCompatibility wrapper."""
    # load envs
    env1 = pyspiel.load_game(game_name)
    env2 = pyspiel.load_game(game_name)

    # convert the environment
    env1 = OpenSpielCompatibilityV0(env1, render_mode=None)
    env2 = OpenSpielCompatibilityV0(env2, render_mode=None)
    env1.reset(seed=42)
    env2.reset(seed=42)

    agent1 = env1.agents[0]
    agent2 = env2.agents[0]

    a_space1 = env1.action_space(agent1)
    a_space1.seed(42)
    a_space2 = env2.action_space(agent2)
    a_space2.seed(42)

    for agent1, agent2 in zip(env1.agent_iter(), env2.agent_iter()):
        assert data_equivalence(agent1, agent2), f"Incorrect agent: {agent1} {agent2}"

        obs1, rew1, term1, trunc1, info1 = env1.last()
        obs2, rew2, term2, trunc2, info2 = env2.last()

        assert data_equivalence(obs1, obs2), f"Incorrect observations: {obs1} {obs2}"
        assert data_equivalence(rew1, rew2), f"Incorrect rewards: {rew1} {rew2}"
        assert data_equivalence(term1, term2), f"Incorrect terms: {term1} {term2}"
        assert data_equivalence(trunc1, trunc2), f"Incorrect truncs: {trunc1} {trunc2}"
        assert data_equivalence(info1, info2), f"Incorrect info: {info1} {info2}"

        action1 = a_space1.sample(mask=info1["action_mask"])
        action2 = a_space2.sample(mask=info2["action_mask"])

        assert data_equivalence(
            action1, action2
        ), f"Incorrect actions: {action1} {action2}"

        env1.step(action1)
        env2.step(action2)
    env1.close()
    env2.close()


@pytest.mark.parametrize("game_name", _PASSING_GAMES)
def test_pickle(game_name):
    """Tests that environments using the OpenSpielCompatibility wrapper can be serialized and deserialized with pickle."""
    env1 = pyspiel.load_game(game_name)
    env1 = OpenSpielCompatibilityV0(env1, render_mode=None)

    env2 = pickle.loads(pickle.dumps(env1))

    assert data_equivalence(
        env1.reset(seed=42), env2.reset(seed=42)
    ), "Incorrect return on reset()"

    agent1 = env1.agent_selection
    agent2 = env2.agent_selection
    assert data_equivalence(agent1, agent2), f"Incorrect agent: {agent1} {agent2}"

    a_space1 = env1.action_space(agent1)
    a_space1.seed(42)
    a_space2 = env2.action_space(agent2)
    a_space2.seed(42)

    for agent1, agent2 in zip(env1.agent_iter(), env2.agent_iter()):
        assert data_equivalence(agent1, agent2), f"Incorrect agent: {agent1} {agent2}"

        obs1, rew1, term1, trunc1, info1 = env1.last()
        obs2, rew2, term2, trunc2, info2 = env2.last()

        assert data_equivalence(obs1, obs2), f"Incorrect observations: {obs1} {obs2}"
        assert data_equivalence(rew1, rew2), f"Incorrect rewards: {rew1} {rew2}"
        assert data_equivalence(term1, term2), f"Incorrect terms: {term1} {term2}"
        assert data_equivalence(trunc1, trunc2), f"Incorrect truncs: {trunc1} {trunc2}"
        assert data_equivalence(info1, info2), f"Incorrect info: {info1} {info2}"

        action1 = a_space1.sample(mask=info1["action_mask"])
        action2 = a_space2.sample(mask=info2["action_mask"])

        assert data_equivalence(
            action1, action2
        ), f"Incorrect actions: {action1} {action2}"

        env1.step(action1)
        env2.step(action2)
    env1.close()
    env2.close()
