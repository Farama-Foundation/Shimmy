"""Tests the functionality of the OpenspielWrapper on openspiel envs."""

import numpy as np
import pyspiel
import pytest
from pettingzoo.test import api_test

from shimmy import OpenspielWrapperV0

_PASSING_GAMES = [
    "2048",
    "amazons",
    "backgammon",
    "bargaining",
    "blackjack",
    "breakthrough",
    "bridge",
    "catch",
    "checkers",
    "chess",
    "cliff_walking",
    "clobber",
    "colored_trails",
    "connect_four",
    "coop_to_1p",
    "cursor_go",
    "dark_chess",
    "dark_hex",
    "dark_hex_ir",
    "deep_sea",
    "first_sealed_auction",
    "gin_rummy",
    "go",
    "hanabi",
    "havannah",
    "hex",
    "kriegspiel",
    "kuhn_poker",
    "leduc_poker",
    "lewis_signaling",
    "liars_dice",
    "liars_dice_ir",
    "mancala",
    "mfg_crowd_modelling",
    "mfg_crowd_modelling_2d",
    "mfg_garnet",
    "negotiation",
    "nim",
    "othello",
    "oware",
    "pentago",
    "phantom_go",
    "phantom_ttt",
    "phantom_ttt_ir",
    "pig",
    "quoridor",
    "rbc",
    "skat",
    "solitaire",
    "stones_and_gems",
    "tic_tac_toe",
    "tiny_bridge_2p",
    "tiny_bridge_4p",
    "tiny_hanabi",
    "trade_comm",
    "ultimate_tic_tac_toe",
    "universal_poker",
    "y",
]

_FAILING_GAMES = [
    "battleship",
    "blotto",
    "bridge_uncontested_bidding",
    "coin_game",
    "coop_box_pushing",
    "coordinated_mp",
    "efg_game",
    "euchre",
    "goofspiel",
    "hearts",
    "laser_tag",
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
    "mfg_dynamic_routing",
    "misere",
    "morpion_solitaire",
    "nfg_game",
    "normal_form_extensive_game",
    "oh_hell",
    "oshi_zumo",
    "pathfinding",
    "repeated_game",
    "restricted_nash_response",
    "sheriff",
    "start_at",
    "tarok",
    "turn_based_simultaneous_game",
]


@pytest.mark.parametrize("game", _PASSING_GAMES)
def test_passing_games(game):
    """Tests the conversion of all openspiel envs."""
    game = pyspiel.load_game(game)
    env = OpenspielWrapperV0(game=game, render_mode=None)

    # api test the env
    # api_test(env)

    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample(mask=info["action_mask"])
        env.step(action)

@pytest.mark.parametrize("game", _FAILING_GAMES)
def test_failing_games(game):
    with pytest.raises(Exception):
        test_passing_games(game)

def test_seeding():
    """Tests the seeding of the openspiel conversion wrapper."""
    # load envs
    env1 = pyspiel.load_game("2048")
    env2 = pyspiel.load_game("2048")

    # convert the environment
    env1 = OpenspielWrapperV0(env1, render_mode=None)
    env2 = OpenspielWrapperV0(env2, render_mode=None)
    env1.reset(seed=42)
    env2.reset(seed=42)

    agent1 = env1.agent_iter()
    agent2 = env2.agent_iter()

    a_space1 = env1.action_space(agent1)
    a_space1.seed(42)
    a_space2 = env2.action_space(agent2)
    a_space2.seed(42)

    for agent1, agent2 in zip(env1.agent_iter(), env2.agent_iter()):
        obs1, rew1, term1, trunc1, info1 = env1.last()
        obs2, rew2, term2, trunc2, info2 = env2.last()

        returns1 = (obs1, rew1, term1, trunc1)
        returns2 = (obs2, rew2, term2, trunc2)

        act1 = a_space1.sample(mask=info1["action_mask"])
        act2 = a_space2.sample(mask=info2["action_mask"])

        env1.step(act1)
        env2.step(act2)

        for stuff1, stuff2 in zip(returns1, returns2):
            if isinstance(stuff1, bool):
                assert stuff1 == stuff2, "Incorrect returns on iteration."
            elif isinstance(stuff1, np.ndarray):
                assert (stuff1 == stuff2).all(), "Incorrect returns on iteration."
