"""Wrapper to convert a dm_env multiagent environment into a pettingzoo compatible environment. """
from __future__ import annotations

import functools
from itertools import repeat

import dm_env
import gymnasium
from pettingzoo import ParallelEnv
from gymnasium.envs.mujoco.mujoco_rendering import Viewer

from shimmy.utils.dm_env import dm_obs2gym_obs, dm_spec2gym_space


def _unravel_ma_timestep(timestep, agents):
    """Opens up the timestep to return obs, reward, terminated, truncated, info."""
    # set terminated and truncated
    term, trunc = False, False
    if timestep.last():
        if timestep.discount == 0:
            trunc = True
        else:
            term = True

    # expand the observations
    observations = [dm_obs2gym_obs(o) for o in timestep.observation]
    observations = dict(zip(agents, observations))

    # sometimes deepmind decides not to reward people
    rewards = dict(zip(agents, repeat(0)))
    if timestep.reward:
        rewards = dict(zip(agents, timestep.reward))

    # expand everything else
    terminations = dict(zip(agents, repeat(term)))
    truncations = dict(zip(agents, repeat(trunc)))

    # duplicate infos
    info = {
        "timestep.discount": timestep.discount,
        "timestep.step_type": timestep.step_type,
    }
    info = dict(zip(agents, repeat(info)))

    return (  # pyright: ignore[reportGeneralTypeIssues]
        observations,
        rewards,
        terminations,
        truncations,
        info,
    )


class DMCMACompatibility(ParallelEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: dm_env.Environment,
        render_mode: str | None,
    ):
        """Wrapper that converts a dm control multiagent environment into a pettingzoo environment.

        Args:
            env (dm_env.Environment): dm control multiagent environment
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
            self.viewer = Viewer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.act_spaces[agent]

    def render(self):
        """
        Renders the environment.
        """
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
        """
        step(action) takes in an action for each agent and returns the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # assert that the actions _must_ have actions for all agents
        assert len(actions) == len(
            self.agents
        ), f"Must have actions for all {len(self.agents)} agents, currently only found {len(actions)}."

        actions = actions.values()
        timestep = self._env.step(actions)

        obs, rews, terms, truncs, infos = _unravel_ma_timestep(timestep, self.agents)

        if self.render_mode == "human":
            self.viewer.render()

        return obs, rews, terms, truncs, infos
