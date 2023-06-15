"""Wrapper to convert a dm_env environment into a gymnasium compatible environment.

Taken from
https://github.com/ikostrikov/dmcgym/blob/main/dmcgym/env.py
and modified to modern gymnasium API
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable, Optional

import dm_env
import gymnasium
import numpy as np
from dm_control import composer
from dm_control.mujoco.engine import Physics as MujocoEnginePhysics
from dm_control.rl import control
from gymnasium.core import ObsType
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils import EzPickle
from mujoco._structs import MjvScene

from shimmy.utils.dm_env import dm_env_step2gym_step, dm_spec2gym_space


class EnvType(Enum):
    """The environment type."""

    COMPOSER = 0
    RL_CONTROL = 1


class DmControlCompatibilityV0(gymnasium.Env[ObsType, np.ndarray], EzPickle):
    """This compatibility wrapper converts a dm-control environment into a gymnasium environment.

    Dm-control is DeepMind's software stack for physics-based simulation and Reinforcement Learning environments, using MuJoCo physics.

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

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "multi_camera"],
        "render_fps": 10,  # this value is updated to use the `env.control_timesteps() * 1000`
    }

    def __init__(
        self,
        env: composer.Environment | control.Environment | dm_env.Environment,
        render_mode: str | None = None,
        render_kwargs: dict[str, Any] | None = None,
    ):
        """Initialises the environment with a render mode along with render information.

        Note: this wrapper supports multi-camera rendering via the `render_mode` argument (render_mode = "multi_camera")

        For more information on DM Control rendering, see https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L178

        Args:
            env (Optional[composer.Environment | control.Environment | dm_env.Environment]): DM Control env to wrap
            render_mode (Optional[str]): rendering mode (options: "human", "rgb_array", "depth_array", "multi_camera")
            render_kwargs (Optional[dict[str, Any]]): Additional keyword arguments for rendering.
                For the width, height and camera id use "width", "height" and "camera_id" respectively.
                See the dm_control implementation for the list of possible kwargs, https://github.com/deepmind/dm_control/blob/330c91f41a21eacadcf8316f0a071327e3f5c017/dm_control/mujoco/engine.py#L178
                Note: kwargs are not used for human rendering, which uses simpler Gymnasium MuJoCo rendering.
        """
        EzPickle.__init__(self, env, render_mode, render_kwargs)
        self._env: Any = env
        self.env_type = self._find_env_type(env)
        self.metadata["render_fps"] = self._env.control_timestep() * 1000

        self.observation_space = dm_spec2gym_space(env.observation_spec())
        self.action_space = dm_spec2gym_space(env.action_spec())

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_kwargs is None:
            render_kwargs = {}
        self.render_kwargs = render_kwargs

        if self.render_mode == "human":
            # We use the gymnasium mujoco rendering, dm-control provides more complex rendering options.
            self.viewer = MujocoRenderer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )

    @property
    def dt(self):
        """Returns the environment control timestep which is equivalent to the number of actions per second."""
        return self._env.control_timestep()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the dm-control environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed=seed)

        timestep = self._env.reset()
        obs, reward, terminated, truncated, info = dm_env_step2gym_step(timestep)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-control environment."""
        timestep = self._env.step(action)

        obs, reward, terminated, truncated, info = dm_env_step2gym_step(timestep)

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
                **self.render_kwargs,
            )
        elif self.render_mode == "depth_array":
            return self._env.physics.render(
                depth=True,
                **self.render_kwargs,
            )
        elif self.render_mode == "multi_camera":
            physics = self._env.physics
            num_cameras = physics.model.ncam
            num_columns = int(math.ceil(math.sqrt(num_cameras)))
            num_rows = int(math.ceil(float(num_cameras) / num_columns))

            # 240 and 320 are the default values in dm-control
            height = self.render_kwargs.get("height", 240)
            width = self.render_kwargs.get("width", 320)
            frame = np.zeros(
                (num_rows * height, num_columns * width, 3),
                dtype=np.uint8,
            )
            assert (
                "camera_id" not in self.render_kwargs
            ), "The camera_id is specified in `multi_camera` render so don't include it in the render_kwargs"
            for col in range(num_columns):
                for row in range(num_rows):
                    camera_id = row * num_columns + col
                    if camera_id >= num_cameras:
                        break
                    subframe = physics.render(
                        camera_id=camera_id,
                        **self.render_kwargs,
                    )
                    frame[
                        row * height : (row + 1) * height,
                        col * width : (col + 1) * width,
                    ] = subframe
            return frame

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
                return self._find_env_type(
                    env._env  # pyright: ignore[reportGeneralTypeIssues]
                )
            elif hasattr(env, "env"):
                return self._find_env_type(
                    env.env  # pyright: ignore[reportGeneralTypeIssues]
                )
            else:
                raise AttributeError(
                    f"Can't know the dm-control environment type, actual type: {type(env)}"
                )
