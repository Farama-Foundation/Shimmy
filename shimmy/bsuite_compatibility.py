"""Wrapper to convert a BSuite environment into a gymnasium compatible environment.
"""
from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium.core import ObsType

from shimmy.utils.dm_env import dm_env_step2gym_step, dm_spec2gym_space

from bsuite.environments import Environment


class BSuiteCompatibilityV0(gymnasium.Env[ObsType, np.ndarray]):
    """A compatibility wrapper that converts a BSuite environment into a gymnasium environment.

    Note:
        Bsuite uses `np.random.RandomState`, a legacy random number generator while gymnasium
        uses `np.random.Generator`, therefore the return type of `np_random` is different from expected.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: Environment,
        render_mode: str | None = None,
    ):
        """Initialises the environment with a render mode along with render information."""
        self._env = env

        self.observation_space = dm_spec2gym_space(env.observation_spec())
        self.action_space = dm_spec2gym_space(env.action_spec())

        assert render_mode is None, f"No render modes available in BSuite."

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the bsuite environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed=seed)
            self._env._rng = self.np_random # pyright: ignore[reportGeneralTypeIssues]

        timestep = self._env.reset()

        obs, reward, terminated, truncated, info = dm_env_step2gym_step(timestep)

        return obs, info  # pyright: ignore[reportGeneralTypeIssues]

    def step(
        self, action: int
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the bsuite environment."""
        timestep = self._env.step(action)

        obs, reward, terminated, truncated, info = dm_env_step2gym_step(timestep)

        return (  # pyright: ignore[reportGeneralTypeIssues]
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray | None:
        """Renders the bsuite env."""
        raise AssertionError("Rendering is not built into BSuite, print the observation instead.")

    def close(self):
        """Closes the environment."""

        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()

    @property
    def np_random(self) -> np.random.RandomState:
        """This should be np.random.Generator but bsuite uses np.random.RandomState."""
        return self._env._rng # pyright: ignore[reportGeneralTypeIssues]

    @np_random.setter
    def np_random(self, value: np.random.RandomState):
        self._env._rng = value # pyright: ignore[reportGeneralTypeIssues]

    def __getattr__(self, item: str):
        """If the attribute is missing, try getting the attribute from bsuite env."""
        return getattr(self._env, item)
