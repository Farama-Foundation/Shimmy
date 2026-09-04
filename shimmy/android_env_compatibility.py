"""Wrapper to convert an android_env environment into a gymnasium compatible environment."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from android_env import env_interface
from gymnasium.core import ObsType
from gymnasium.utils import EzPickle

from shimmy.utils.dm_env import dm_env_step2gym_step, dm_spec2gym_space


class AndroidEnvCompatibilityV0(
    gymnasium.Env[ObsType, "dict[str, np.ndarray]"], EzPickle
):
    """A compatibility wrapper that converts an android_env environment into a Gymnasium environment.

    AndroidEnv is DeepMind's reinforcement-learning environment for Android applications,
    driving the Android Emulator under the hood. It exposes a `dm_env.Environment` with
    dict-valued observations and actions; this wrapper converts the specs into Gymnasium
    spaces while preserving the dict structure expected by the upstream environment.

    The standard AndroidEnv observation contains a ``pixels`` key (an RGB array of the
    device screen) which is also surfaced by ``render()`` when ``render_mode="rgb_array"``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env: env_interface.AndroidEnvInterface,
        render_mode: str | None = None,
    ):
        """Wraps an existing AndroidEnv instance.

        Args:
            env: An :class:`android_env.env_interface.AndroidEnvInterface` instance,
                typically produced by :func:`android_env.loader.load`.
            render_mode: One of the supported render modes. Currently only
                ``"rgb_array"`` is implemented, which returns the latest ``pixels``
                observation. The emulator window is controlled separately via
                ``run_headless`` on the underlying AndroidEnv config.
        """
        EzPickle.__init__(self, env, render_mode)
        self._env = env

        self.observation_space = dm_spec2gym_space(env.observation_spec())
        self.action_space = dm_spec2gym_space(env.action_spec())

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._last_obs: dict[str, Any] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the underlying AndroidEnv.

        AndroidEnv does not expose a seeding mechanism on reset, so ``seed`` is only
        used to seed Gymnasium's ``np_random``.
        """
        super().reset(seed=seed)

        timestep = self._env.reset()
        obs, _reward, _terminated, _truncated, info = dm_env_step2gym_step(timestep)
        self._last_obs = obs  # type: ignore[assignment]
        return obs, info

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps the underlying AndroidEnv with the given dict action."""
        timestep = self._env.step(action)
        obs, reward, terminated, truncated, info = dm_env_step2gym_step(timestep)
        self._last_obs = obs  # type: ignore[assignment]
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Returns the latest ``pixels`` observation as an RGB array."""
        if self.render_mode == "rgb_array":
            if self._last_obs is None or "pixels" not in self._last_obs:
                return None
            return np.asarray(self._last_obs["pixels"])
        return None

    def close(self) -> None:
        """Closes the underlying AndroidEnv."""
        self._env.close()

    def __getattr__(self, item: str) -> Any:
        """Forward attribute access (e.g. ``task_extras``, ``execute_adb_call``) to the wrapped env."""
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._env, item)
