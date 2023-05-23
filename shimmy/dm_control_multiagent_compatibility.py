"""Wrapper to convert a dm_env multiagent environment into a pettingzoo compatible environment."""
from __future__ import annotations

import functools
from itertools import repeat
from typing import TYPE_CHECKING, Any

import dm_control.composer
import dm_env
import gymnasium
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils import EzPickle
from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv

from shimmy.utils.dm_control_multiagent import load_dm_control_soccer
from shimmy.utils.dm_env import dm_obs2gym_obs, dm_spec2gym_space

if TYPE_CHECKING:
    from dm_control.locomotion import soccer as dm_soccer


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
    """This compatibility wrapper converts multi-agent dm-control environments, primarily soccer, into a PettingZoo environment.

    Dm-control is DeepMind's software stack for physics-based simulation and Reinforcement Learning environments,
    using MuJoCo physics. This compatibility wrapper converts a dm-control environment into a gymnasium environment.
    """

    metadata = {"render_modes": ["human"], "name": "DmControlMultiAgentCompatibilityV0"}

    def __init__(
        self,
        env: dm_control.composer.Environment | None = None,
        team_size: int | None = None,
        time_limit: float | None = None,
        disable_walker_contacts: bool | None = None,
        enable_field_box: bool | None = None,
        terminate_on_goal: bool | None = None,
        walker_type: dm_soccer.WalkerType | None = None,
        render_mode: str | None = None,
    ):
        """Wrapper to convert a dm control multi-agent environment into a PettingZoo environment.

        Due to how the underlying environment is set up, this environment is nondeterministic, so seeding does not work.

        Note: to wrap an existing environment, only the env and render_mode arguments can be specified.
        All other arguments (marked [DM CONTROL ARG]) are specific to DM Lab and will be used to load a new environment.

        Args:
            env (Optional[dm_env.Environment]): existing dm control multi-agent environment to wrap
            team_size (Optional[int]): number of players for each team                                    [DM CONTROL ARG]
            time_limit (Optional[float]): time limit for the game                                         [DM CONTROL ARG]
            disable_walker_contacts (Optional[bool]): flag to disable walker contacts                     [DM CONTROL ARG]
            enable_field_box (Optional[bool]): flag to enable field box                                   [DM CONTROL ARG]
            terminate_on_goal (Optional[bool]): flag to terminate the environment on goal                 [DM CONTROL ARG]
            walker_type (Optional[dm_soccer.WalkerType]): specify walker type (BOXHEAD, ANT, or HUMANOID) [DM CONTROL ARG]
            render_mode (Optional[str]): rendering mode
        """
        EzPickle.__init__(self, env=env, render_mode=render_mode)
        ParallelEnv.__init__(self)

        DM_CONTROL_ARGS = [
            team_size,
            time_limit,
            disable_walker_contacts,
            enable_field_box,
            terminate_on_goal,
            walker_type,
        ]

        # Only one of env and DM_CONTROL_ARGS can be provided, the other should be None.
        if env is None and all(arg is None for arg in DM_CONTROL_ARGS):
            raise ValueError(
                "No environment provided. Use `env` to specify an existing environment, or load an environment by specifying at least one of `team_size`, `time_limit`, `disable_walker_contacts`, `enable_field_box` `terminate_on_goal`, or `walker_type`."
            )
        elif env is not None and any(arg is not None for arg in DM_CONTROL_ARGS):
            raise ValueError(
                "Two environments provided. Use `env` to specify an existing environment, or load an environment by specifying at least one of `team_size`, `time_limit`, `disable_walker_contacts`, `enable_field_box` `terminate_on_goal`, or `walker_type`."
            )
        elif any(arg is not None for arg in DM_CONTROL_ARGS):
            self._env = load_dm_control_soccer(
                team_size,
                time_limit,
                disable_walker_contacts,
                enable_field_box,
                terminate_on_goal,
                walker_type,
            )
        elif env is not None:
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
            assert self._env.physics is not None
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
        assert self._env.physics is not None
        self._env.physics.free()
        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()

    def reset(
        self, seed: int | None = None, options: dict[AgentID, Any] | None = None
    ) -> tuple[ObsDict, dict[str, Any]]:
        """reset.

        Resets the dm-control environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with (unused)

        Returns:
            observations
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        self._env._random_state = np.random.RandomState(seed)
        timestep = self._env.reset()
        observations, _, _, _, info = _unravel_ma_timestep(timestep, self.agents)

        return observations, info

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

        timestep = self._env.step(actions.values())

        obs, rewards, terminations, truncations, infos = _unravel_ma_timestep(
            timestep, self.agents
        )

        if self.render_mode == "human":
            self.viewer.render(self.render_mode)

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos
