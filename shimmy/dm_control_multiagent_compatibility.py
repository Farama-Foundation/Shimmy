"""Wrapper to convert a dm_env multiagent environment into a pettingzoo compatible environment."""
from __future__ import annotations

import functools
from itertools import repeat
from typing import Any

import dm_control.composer
import dm_env
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from pettingzoo import ParallelEnv

from shimmy.utils.dm_env import dm_obs2gym_obs, dm_spec2gym_space


def _unravel_ma_timestep(
    timestep: dm_env.TimeStep, agents: list[str]
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, Any],
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
    observations: dict[str, Any] = dict(zip(agents, list_observations))

    # sometimes deepmind decides not to reward people
    rewards: dict[str, float] = dict(zip(agents, repeat(0.0)))
    if timestep.reward:
        rewards = dict(zip(agents, timestep.reward))

    # expand everything else
    terminations: dict[str, bool] = dict(zip(agents, repeat(term)))
    truncations: dict[str, bool] = dict(zip(agents, repeat(trunc)))

    # duplicate infos
    info = {
        "timestep.discount": timestep.discount,
        "timestep.step_type": timestep.step_type,
    }
    info: dict[str, Any] = dict(zip(agents, repeat(info)))

    return (
        observations,
        rewards,
        terminations,
        truncations,
        info,
    )


class DmControlMultiAgentCompatibilityV0(ParallelEnv):
    """Compatibility environment for multi-agent dm-control environments, primarily soccer."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: dm_control.composer.Environment,
        render_mode: str | None = None,
    ):
        """Wrapper that converts a dm control multi-agent environment into a pettingzoo environment.

        Due to how the underlying environment is setup, this environment is nondeterministic, so seeding doesn't work.

        Args:
            env (dm_env.Environment): dm control multi-agent environment
            render_mode (Optional[str]): render_mode
        """
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
    def observation_space(self, agent):
        """The observation space for agent."""
        return self.obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """The action space for agent."""
        return self.act_spaces[agent]

    def render(self):
        """Renders the environment."""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

    def close(self):
        """Closes the environment."""
        self._env.physics.free()
        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()

    def reset(self, seed=None, return_info=False, options=None):
        """Resets the dm-control environment."""
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        timestep = self._env.reset()

        observations, _, _, _, infos = _unravel_ma_timestep(timestep, self.agents)

        if not return_info:
            return observations
        else:
            return observations, infos

    def step(self, actions):
        """Steps through all agents with the actions."""
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
