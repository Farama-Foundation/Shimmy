"""Wrapper to convert a dm_env multiagent environment into a pettingzoo compatible environment."""
from __future__ import annotations

import functools
from itertools import repeat
from typing import Any

import dm_control.composer
import dm_env
import gymnasium
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils import EzPickle
from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv

from shimmy.utils.dm_env import dm_obs2gym_obs, dm_spec2gym_space


def _unravel_ma_timestep(
    timestep: dm_env.TimeStep, agents: list[AgentID]
) -> tuple[
    dict[AgentID, Any],
    dict[AgentID, float],
    dict[AgentID, bool],
    dict[AgentID, bool],
    dict[AgentID, Any],
]:
    """Opens up the timestep to return obs, reward, terminated, truncated, info."""
    # set terminated and truncated
    term, trunc = False, False
    if timestep.last():
        if timestep.discount == 0:
            trunc = True
        else:
            term = True

    # expand the observations
    list_observations = [dm_obs2gym_obs(obs) for obs in timestep.observation]
    observations: dict[AgentID, Any] = dict(zip(agents, list_observations))

    # sometimes deepmind decides not to reward people
    rewards: dict[AgentID, float] = dict(zip(agents, repeat(0.0)))
    if timestep.reward:
        rewards = dict(zip(agents, timestep.reward))

    # expand everything else
    terminations: dict[AgentID, bool] = dict(zip(agents, repeat(term)))
    truncations: dict[AgentID, bool] = dict(zip(agents, repeat(trunc)))

    # duplicate infos
    info = {
        "timestep.discount": timestep.discount,
        "timestep.step_type": timestep.step_type,
    }
    info: dict[AgentID, Any] = dict(zip(agents, repeat(info)))

    return (
        observations,
        rewards,
        terminations,
        truncations,
        info,
    )


class DmControlMultiAgentCompatibilityV0(ParallelEnv, EzPickle):
    """This compatibility wrapper converts multi-agent dm-control environments, primarily soccer, into a Pettingzoo environment.

    Dm-control is DeepMind's software stack for physics-based simulation and Reinforcement Learning environments,
    using MuJoCo physics. This compatibility wrapper converts a dm-control environment into a gymnasium environment.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: dm_control.composer.Environment,
        render_mode: str | None = None,
    ):
        """Wrapper to convert a dm control multi-agent environment into a pettingzoo environment.

        Due to how the underlying environment is set up, this environment is nondeterministic, so seeding doesn't work.

        Args:
            env (dm_env.Environment): dm control multi-agent environment
            render_mode (Optional[str]): render_mode
        """
        EzPickle.__init__(self, env=env, render_mode=render_mode)
        super().__init__()
        self._env = env
        self.render_mode = render_mode

        # get action and observation specs first
        all_obs_spaces = [
            dm_spec2gym_space(spec) for spec in self._env.observation_spec()
        ]
        all_act_spaces = [dm_spec2gym_space(spec) for spec in self._env.action_spec()]
        num_players = len(all_obs_spaces)

        # agent definitions
        self.possible_agents = ["player_" + str(r) for r in range(num_players)]
        self.agent_id_name_mapping = dict(zip(range(num_players), self.possible_agents))
        self.agent_name_id_mapping = dict(zip(self.possible_agents, range(num_players)))

        # the official spaces
        self.obs_spaces = dict(zip(self.possible_agents, all_obs_spaces))
        self.act_spaces = dict(zip(self.possible_agents, all_act_spaces))

        if self.render_mode == "human":
            self.viewer = MujocoRenderer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """observation_space.

        Get the observation space from the underlying meltingpot substrate.

        Args:
            agent (AgentID): agent

        Returns:
            observation_space: spaces.Space
        """
        return self.obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """action_space.

        Get the action space from the underlying dm-control env.

        Args:
            agent (AgentID): agent

        Returns:
            action_space: spaces.Space
        """
        return self.act_spaces[agent]

    def render(self) -> np.ndarray | None:
        """render.

        Renders the environment.

        Returns:
            The rendering of the environment, depending on the render mode
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

    def close(self):
        """close.

        Closes the environment.
        """
        self._env.physics.free()
        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()

    def reset(
        self, seed: int | None = None, options: dict[AgentID, Any] | None = None
    ) -> ObsDict:
        """reset.

        Resets the dm-control environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            observations
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        timestep = self._env.reset()

        observations, _, _, _, _ = _unravel_ma_timestep(timestep, self.agents)

        return observations

    def step(
        self, actions: ActionDict
    ) -> tuple[
        ObsDict,
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, Any],
    ]:
        """step.

        Steps through all agents with the actions.

        Args:
            actions: dict of actions to step through the environment with

        Returns:
            (observations, rewards, terminations, truncations, infos)
        """
        # assert that the actions _must_ have actions for all agents
        assert len(actions) == len(
            self.agents
        ), f"Must have actions for all {len(self.agents)} agents, currently only found {len(actions)}."

        actions = actions.values()
        timestep = self._env.step(actions)

        obs, rewards, terminations, truncations, infos = _unravel_ma_timestep(
            timestep, self.agents
        )

        if self.render_mode == "human":
            self.viewer.render(self.render_mode)

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos
