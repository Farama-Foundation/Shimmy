"""Wrapper to convert a dm_lab environment into a gymnasium compatible environment."""
from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from shimmy.utils.dm_lab import dm_lab_obs2gym_obs_space, dm_lab_spec2gym_space


class DmLabCompatibilityV0(gym.Env[ObsType, Dict[str, np.ndarray]]):
    """A compatibility wrapper that converts a dm_lab-control environment into a gymnasium environment."""

    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(
        self,
        env: Any,
        render_mode: None = None,
    ):
        """Initialises the environment with a render mode along with render information."""
        self._env = env

        # need to do this to figure out what observation spec the user used
        self._env.reset()
        self.observation_space = dm_lab_obs2gym_obs_space(self._env.observations())
        self.action_space = dm_lab_spec2gym_space(env.action_spec())

        assert (
            render_mode is None
        ), "Render mode must be set on dm_lab environment init. Pass `renderer='sdl'` to the config of the base env to enable human rendering."
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
        )

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-lab environment."""
        # there's some funky quantization happening here, dm_lab only accepts ints as actions
        action_array = np.array([a[0] for a in action.values()], dtype=np.intc)
        reward = self._env.step(action_array)

        obs = self._env.observations()
        terminated = not self._env.is_running()
        truncated = False
        info = {}

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        """Renders the dm_lab env."""
        raise NotImplementedError

    def close(self):
        """Closes the environment."""
        self._env.close()

    def __getattr__(self, item: str):
        """If the attribute is missing, try getting the attribute from dm_lab env."""
        return getattr(self._env, item)
