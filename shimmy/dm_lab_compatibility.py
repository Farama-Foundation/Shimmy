"""Wrapper to convert a dm_lab environment into a gymnasium compatible environment.

Taken from
https://github.com/ikostrikov/dmcgym/blob/main/dmcgym/env.py
and modified to modern gymnasium API
"""
from __future__ import annotations

from typing import Any, TypeVar

import gymnasium
import numpy as np
from gymnasium.core import ObsType

from shimmy.utils.dm_lab_utils import dm_lab_obs2gym_obs_space, dm_lab_spec2gym_space

deepmind_lab_env = TypeVar("deepmind_lab_env")


class DmLabCompatibility(gymnasium.Env[ObsType, np.ndarray]):
    """A compatibility wrapper that converts a dm_lab-control environment into a gymnasium environment."""

    metadata = {"render_modes": None, "render_fps": 10}

    def __init__(
        self,
        env: deepmind_lab_env,
        render_mode: str | None = None,
        render_height: int = 84,
        render_width: int = 84,
        camera_id: int = 0,
    ):
        """Initialises the environment with a render mode along with render information."""
        self._env = env

        # need to do this to figure out what observation spec the user used
        self._env.reset()
        self.observation_space = dm_lab_obs2gym_obs_space(self._env.observations())
        self.action_space = dm_lab_spec2gym_space(env.action_spec())

        assert (
            render_mode is None
        ), "Render mode must be set on dm_lab environment init. Pass `renderer='sdl'` to enable human rendering."
        self.render_mode = render_mode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the dm-lab environment."""
        super().reset(seed=seed)

        self._env.reset(seed=seed)
        info = {}

        return (
            self._env.observations(),
            info,
        )  # pyright: ignore[reportGeneralTypeIssues]

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-lab environment."""
        # there's some funky quantization happening here, dm_lab only accepts ints as actions
        action = np.array([a[0] for a in action.values()], dtype=np.intc)
        reward = self._env.step(action)

        obs = self._env.observations()
        terminated = False
        truncated = False
        info = {}

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
