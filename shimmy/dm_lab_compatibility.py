"""Wrapper to convert a dm_lab environment into a gymnasium compatible environment.

Taken from
https://github.com/ikostrikov/dmcgym/blob/main/dmcgym/env.py
and modified to modern gymnasium API
"""
from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from dm_env import Environment
from gymnasium.core import ObsType

from shimmy.utils.dm_lab_utils import dm_lab_spec2gym_space


class DmLabCompatibility(gymnasium.Env[ObsType, np.ndarray]):
    """A compatibility wrapper that converts a dm_lab-control environment into a gymnasium environment."""

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

        self.observation_space = dm_lab_spec2gym_space(env.observation_spec())
        self.action_space = dm_lab_spec2gym_space(env.action_spec())

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
        """Resets the dm-lab environment."""
        super().reset(seed=seed)

        timestep = self._env.reset()

        obs, reward, terminated, truncated, info = expose_timestep(timestep)

        obs = None
        info = {}

        return obs, info  # pyright: ignore[reportGeneralTypeIssues]

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-lab environment."""
        # there's some funky quantization happening here, dm_lab only accepts ints as actions
        action = np.array([a[0] for a in action.values()], dtype=np.intc)
        timestep = self._env.step(action)
        print(timestep)

        obs, reward, terminated, truncated, info = expose_timestep(timestep)

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
        """Renders the dm_lab env."""
        raise NotImplementedError

    def close(self):
        """Closes the environment."""
        self._env.close()

    def __getattr__(self, item: str):
        """If the attribute is missing, try getting the attribute from dm_lab env."""
        return getattr(self._env, item)
