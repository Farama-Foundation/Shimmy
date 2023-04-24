"""Wrapper to convert a Melting Pot substrate into a PettingZoo compatible environment.

Taken from
https://github.com/deepmind/meltingpot/blob/main/examples/pettingzoo/utils.py
and modified to modern PettingZoo API
"""
# pyright: reportOptionalSubscript=false
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

import gymnasium
import numpy as np
import pygame
from gymnasium.utils.ezpickle import EzPickle
from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv

import shimmy.utils.meltingpot as utils

if TYPE_CHECKING:
    import meltingpot.python


class MeltingPotCompatibilityV0(ParallelEnv, EzPickle):
    """This compatibility wrapper converts a Melting Pot substrate into a PettingZoo environment.

    Due to how the underlying environment is set up, this environment is nondeterministic, so seeding doesn't work.

    Melting Pot is a research tool developed to facilitate work on multi-agent artificial intelligence.
    It assesses generalization to novel social situations involving both familiar and unfamiliar individuals,
    and has been designed to test a broad range of social interactions such as: cooperation, competition,
    deception, reciprocation, trust, stubbornness and so on.
    Melting Pot offers researchers a set of over 50 multi-agent reinforcement learning substrates (multi-agent games)
    on which to train agents, and over 256 unique test scenarios on which to evaluate these trained agents.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "MeltingPotCompatibilityV0",
    }

    PLAYER_STR_FORMAT = "player_{index}"
    MAX_CYCLES = 1000

    def __init__(
        self,
        env: meltingpot.python.utils.substrates.substrate.Substrate | None = None,
        substrate_name: str | None = None,
        max_cycles: int = MAX_CYCLES,
        render_mode: str | None = None,
    ):
        """Wrapper that converts a Melting Pot environment into a PettingZoo environment.

        Args:
            env (Optional[meltingpot.python.utils.substrates.substrate.Substrate]): existing Melting Pot environment to wrap
            substrate_name (Optional[str]): name of Melting Pot substrate to load (instead of existing environment)
            max_cycles (Optional[int]): maximum number of cycles before truncation
            render_mode (Optional[str]): rendering mode
        """
        EzPickle.__init__(
            self,
            env,
            substrate_name,
            max_cycles,
            render_mode,
        )

        # Only one of substrate_name and env can be provided, the other should be None
        if env is None and substrate_name is None:
            raise ValueError(
                "No environment provided. Use `env` to specify an existing environment, or load an environment with `substrate_name`."
            )
        elif env is not None and substrate_name is not None:
            raise ValueError(
                "Two environments provided. Use `env` to specify an existing environment, or load an environment with `substrate_name`."
            )
        elif substrate_name is not None:
            self._env = utils.load_meltingpot(substrate_name)
        elif env is not None:
            self._env = env

        self.max_cycles = max_cycles

        # Set up PettingZoo variables
        self.render_mode = render_mode
        self.state_space = utils.dm_spec2gym_space(
            self._env.observation_spec()[0]["WORLD.RGB"]
        )
        self._num_players = len(self._env.observation_spec())
        self.possible_agents = [
            self.PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]
        self.agents = [agent for agent in self.possible_agents]
        self.num_cycles = 0

        # Set up pygame rendering
        if self.render_mode == "human":
            self.display_scale = 4
            self.display_fps = 5

            pygame.init()
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Melting Pot")
            shape = self.state_space.shape
            self.game_display = pygame.display.set_mode(
                (
                    int(shape[1] * self.display_scale),
                    int(shape[0] * self.display_scale),
                )
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """observation_space.

        Get the observation space from the underlying Melting Pot substrate.

        Args:
            agent (AgentID): agent

        Returns:
            observation_space: spaces.Space
        """
        observation_space = utils.remove_world_observations_from_space(
            utils.dm_spec2gym_space(self._env.observation_spec()[0])  # type: ignore
        )
        return observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """action_space.

        Get the action space from the underlying Melting Pot substrate.

        Args:
            agent (AgentID): agent

        Returns:
            action_space: spaces.Space
        """
        action_space = utils.dm_spec2gym_space(self._env.action_spec()[0])
        return action_space

    def state(self) -> np.ndarray:
        """State.

        Get an observation of the current environment's state. Used in rendering.

        Returns:
            observation
        """
        return self._env.observation()

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> ObsDict:
        """reset.

        Resets the environment.

        Args:
            seed: the seed to reset the environment with (not used, due to nondeterministic underlying environment)
            options: the options to reset the environment with

        Returns:
            observations
        """
        timestep = self._env.reset()
        self.agents = self.possible_agents[:]
        self.num_cycles = 0

        observations = utils.timestep_to_observations(timestep)

        return observations

    def step(
        self, actions: ActionDict
    ) -> tuple[
        ObsDict, dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]
    ]:
        """step.

        Steps through all agents with one action

        Args:
            actions: actions to step through the environment with

        Returns:
            (observations, rewards, terminations, truncations, infos)
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

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def close(self):
        """close.

        Closes the environment.
        """
        self._env.close()

    def render(self) -> None | np.ndarray:
        """render.

        Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        rgb_arr = self.state()[0]["WORLD.RGB"]

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        elif self.render_mode == "human":
            rgb_arr = np.transpose(rgb_arr, (1, 0, 2))
            surface = pygame.surfarray.make_surface(rgb_arr)
            rect = surface.get_rect()
            surf = pygame.transform.scale(
                surface,
                (int(rect[2] * self.display_scale), int(rect[3] * self.display_scale)),
            )

            self.game_display.blit(surf, dest=(0, 0))
            pygame.display.update()
            self.clock.tick(self.display_fps)
            return None
        elif self.render_mode == "rgb_array":
            return rgb_arr
