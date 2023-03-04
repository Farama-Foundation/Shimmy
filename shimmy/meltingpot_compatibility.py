"""Wrapper to convert a meltingpot substrate into a pettingzoo compatible environment

Taken from
https://github.com/deepmind/meltingpot/blob/main/examples/pettingzoo/utils.py
and modified to modern pettingzoo API
"""
from __future__ import annotations

import functools
from typing import Tuple, Dict, Optional

import gymnasium
import numpy as np
from gymnasium.utils import EzPickle
import matplotlib.pyplot as plt
from ml_collections import config_dict
from pettingzoo.utils.env import ParallelEnv, AgentID
from pettingzoo.utils.env import ObsDict, ActionDict

import shimmy.utils.meltingpot as utils
from meltingpot.python import substrate


class MeltingPotCompatibilityV0(ParallelEnv, EzPickle):
    """This compatibility wrapper converts a meltingpot substrate into a pettingzoo environment.

    Melting Pot is a research tool developed to facilitate work on multi-agent artificial intelligence.
    It assesses generalization to novel social situations involving both familiar and unfamiliar individuals,
    and has been designed to test a broad range of social interactions such as: cooperation, competition,
    deception, reciprocation, trust, stubbornness and so on.
    Melting Pot offers researchers a set of over 50 multi-agent reinforcement learning substrates (multi-agent games)
    on which to train agents, and over 256 unique test scenarios on which to evaluate these trained agents.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    PLAYER_STR_FORMAT = 'player_{index}'
    MAX_CYCLES = 1000

    def __init__(
            self,
            substrate_name: str,
            render_mode: str | None,
            max_cycles: int = MAX_CYCLES
    ):
        """Wrapper that converts a openspiel environment into a pettingzoo environment.

        Args:
            substrate_name (str): name of meltingpot substrate to load
            render_mode (Optional[str]): render_mode
            max_cycles (Optional[int]): maximum number of cycles (steps) before termination
        """
        # Load substrate from pickle
        self.substrate_name = substrate_name
        self.render_mode = render_mode
        self.player_roles = substrate.get_config(self.substrate_name).default_player_roles
        self.max_cycles = max_cycles
        self.env_config = {"substrate": self.substrate_name, "roles": self.player_roles}
        EzPickle.__init__(self, self.render_mode, self.env_config, self.max_cycles)

        # Build substrate
        self.env_config = config_dict.ConfigDict(self.env_config)
        self._env = substrate.build(self.env_config['substrate'], roles=self.env_config['roles'])

        self.state_space = utils.spec_to_space(
            self._env.observation_spec()[0]['WORLD.RGB'])  # type: ignore

        # Set agents
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [
            self.PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]
        self.agents = [agent for agent in self.possible_agents]

    def state_space(self) -> gymnasium.spaces.Space:
        """observation_space.

        Get the state space from the underlying meltingpot substrate.

        Returns:
            state_space: spaces.Space
        """

        state_space = utils.spec_to_space(
            self._env.observation_spec()[0]['WORLD.RGB'])
        return state_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """observation_space.

        Get the observation space from the underlying meltingpot substrate.

        Args:
            agent (AgentID): agent

        Returns:
            observation_space: spaces.Space
        """
        observation_space = utils.remove_world_observations_from_space(
            utils.spec_to_space(self._env.observation_spec()[0])  # type: ignore
        )
        return observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """action_space.

        Get the action space from the underlying meltingpot substrate.

        Args:
            agent (AgentID): agent

        Returns:
            action_space: spaces.Space
        """
        action_space = utils.spec_to_space(self._env.action_spec()[0])
        return action_space

    def state(self) -> np.ndarray:
        return self._env.observation()

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> ObsDict:
        """reset.

        Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        return utils.timestep_to_observations(timestep)

    def step(
            self, actions: ActionDict
    ) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        """step.

        Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        actions = [actions[agent] for agent in self.agents]
        timestep = self._env.step(actions)
        rewards = {
            agent: timestep.reward[index] for index, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        termination = timestep.last()
        terminations = {agent: termination for agent in self.agents}
        truncation = self.num_cycles >= self.max_cycles
        truncations = {agent: truncation for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if termination or truncation:
            self.agents = []

        observations = utils.timestep_to_observations(timestep)
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self._env.close()

    def render(self) -> None | np.ndarray:
        """Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        rgb_arr = self.state()[0]['WORLD.RGB']
        if self.render_mode == 'human':
            plt.cla()
            plt.imshow(rgb_arr, interpolation='nearest')
            plt.show(block=False)
            return None
        return rgb_arr
