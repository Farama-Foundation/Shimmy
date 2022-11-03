"""Wrapper to convert a dm_env environment into a gymnasium compatible environment.

Taken from
https://github.com/ikostrikov/dmcgym/blob/main/dmcgym/env.py
and modified to modern gymnasium API
"""
from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from dm_control.rl.control import Environment
from gymnasium.core import ObsType
from numpy.random import RandomState

from shimmy.utils.dm_utils import dm_obs2gym_obs, dm_spec2gym_space


class DmControlCompatibility(gymnasium.Env[ObsType, np.ndarray]):
    """A compatibility wrapper that converts a dm-control environment into a gymnasium environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        env: Environment,
        render_mode: str | None = None,
        render_height: int = 84,
        render_width: int = 84,
        camera_id: int = 0,
    ):
        """Initialises the environment with a render mode along with render information."""
        self._env = env

        self.observation_space = dm_spec2gym_space(env.observation_spec())
        self.action_space = dm_spec2gym_space(env.action_spec())

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_height, self.render_width = render_height, render_width
        self.camera_id = camera_id

        if self.render_mode == "human":
            from gymnasium.envs.mujoco.mujoco_rendering import Viewer

            self.viewer = Viewer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the dm-control environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = RandomState(seed=seed)

        timestep = self._env.reset()
        obs = dm_obs2gym_obs(timestep.observation)
        info = {
            "timestep.discount": timestep.discount,
            "timestep.step_type": timestep.step_type,
        }
        return obs, info  # pyright: ignore[reportGeneralTypeIssues]

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-control environment."""
        # Step through the dm-control environment
        timestep = self._env.step(action)

        # open up the timestep and process reward and observation
        obs = dm_obs2gym_obs(timestep.observation)
        reward = timestep.reward or 0

        # set terminated and truncated
        terminated, truncated = False, False
        if timestep.last():
            if timestep.discount == 0:
                truncated = True
            else:
                terminated = True

        info = {
            "timestep.discount": timestep.discount,
            "timestep.step_type": timestep.step_type,
        }

        if self.render_mode == "human":
            self.viewer.render()

        return (  # pyright: ignore[reportGeneralTypeIssues]
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray | None:
        """Renders the dm-control env."""
        if self.render_mode == "rgb_array":
            return self._env.physics.render(
                height=self.render_height,
                width=self.render_width,
                camera_id=self.camera_id,
            )

    def close(self):
        """Closes the environment."""
        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()

    @property
    def np_random(self) -> np.random.RandomState:
        """This should be np.random.Generator but dm-control uses np.random.RandomState."""
        return self._env.task._random

    @np_random.setter
    def np_random(self, value: np.random.RandomState):
        self._env.task._random = value

    def __getattr__(self, item: str):
        """If the attribute is missing, try getting the attribute from dm_control env."""
        return getattr(self._env, item)
