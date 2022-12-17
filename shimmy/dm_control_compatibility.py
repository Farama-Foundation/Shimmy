"""Wrapper to convert a dm_env environment into a gymnasium compatible environment.

Taken from
https://github.com/ikostrikov/dmcgym/blob/main/dmcgym/env.py
and modified to modern gymnasium API
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import dm_env
import gymnasium
import numpy as np
from dm_control import composer
from dm_control.rl import control
from gymnasium.core import ObsType
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from shimmy.utils.dm_env import dm_control_step2gym_step, dm_spec2gym_space


class EnvType(Enum):
    """The environment type."""

    COMPOSER = 0
    RL_CONTROL = 1


class DmControlCompatibilityV0(gymnasium.Env[ObsType, np.ndarray]):
    """A compatibility wrapper that converts a dm-control environment into a gymnasium environment.

    Dm-control actually has two Environments classes, `dm_control.composer.Environment` and
    `dm_control.rl.control.Environment` that while both inherit from `dm_env.Environment`, they differ
    in implementation.

    For environment in `dm_control.suite` are `dm-control.rl.control.Environment` while
    dm-control locomotion and manipulation environments use `dm-control.composer.Environment`.

    This wrapper supports both Environment class through determining the base environment type.

    Note:
        dm-control uses `np.random.RandomState`, a legacy random number generator while gymnasium
        uses `np.random.Generator`, therefore the return type of `np_random` is different from expected.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        env: composer.Environment | control.Environment | dm_env.Environment,
        render_mode: str | None = None,
        render_height: int = 84,
        render_width: int = 84,
        camera_id: int = 0,
    ):
        """Initialises the environment with a render mode along with render information."""
        self._env = env
        self.env_type = self._find_env_type(env)

        self.observation_space = dm_spec2gym_space(env.observation_spec())
        self.action_space = dm_spec2gym_space(env.action_spec())

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_height, self.render_width = render_height, render_width
        self.camera_id = camera_id

        if self.render_mode == "human":
            # We use the gymnasium mujoco rendering, dm-control provides more complex rendering options.
            self.viewer = MujocoRenderer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the dm-control environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed=seed)

        timestep = self._env.reset()

        obs, reward, terminated, truncated, info = dm_control_step2gym_step(timestep)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-control environment."""
        timestep = self._env.step(action)

        obs, reward, terminated, truncated, info = dm_control_step2gym_step(timestep)

        if self.render_mode == "human":
            self.viewer.render(self.render_mode)

        return (
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
        self._env.physics.free()
        self._env.close()

        if hasattr(self, "viewer"):
            self.viewer.close()

    @property
    def np_random(self) -> np.random.RandomState:
        """This should be np.random.Generator but dm-control uses np.random.RandomState."""
        if self.env_type is EnvType.RL_CONTROL:
            return self._env.task._random
        else:
            return self._env._random_state

    @np_random.setter
    def np_random(self, value: np.random.RandomState):
        if self.env_type is EnvType.RL_CONTROL:
            self._env.task._random = value
        else:
            self._env._random_state = value

    def __getattr__(self, item: str):
        """If the attribute is missing, try getting the attribute from dm_control env."""
        return getattr(self._env, item)

    def _find_env_type(self, env) -> EnvType:
        """Tries to discover env types, in particular for environments with wrappers."""
        if isinstance(env, composer.Environment):
            return EnvType.COMPOSER
        elif isinstance(env, control.Environment):
            return EnvType.RL_CONTROL
        else:
            assert isinstance(env, dm_env.Environment)

            if hasattr(env, "_env"):
                return self._find_env_type(env._env)
            elif hasattr(env, "env"):
                return self._find_env_type(env.env)
            else:
                raise AttributeError(
                    f"Can't know the dm-control environment type, actual type: {type(env)}"
                )
