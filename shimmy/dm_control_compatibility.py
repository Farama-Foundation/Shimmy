# Taken from
# https://github.com/ikostrikov/dmcgym/blob/main/dmcgym/env.py
# and modified to modern gymnasium API

import copy
from typing import Optional, OrderedDict

import dm_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gymnasium.utils import seeding


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        else:
            raise ValueError()

        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    else:
        raise NotImplementedError


def dmc_obs2gym_obs(obs):
    if isinstance(obs, OrderedDict) or isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = dmc_obs2gym_obs(v)
        return obs
    else:
        return np.asarray(obs)


class dm_control_wrapper(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env: dm_env.Environment,
        render_mode: str,
        render_height: int = 84,
        render_width: int = 84,
        camera_id: int = 0,
    ):
        self._env = env

        # convert spaces
        self.observation_space = dmc_spec2gym_space(self._env.observation_spec())
        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        # camera rendering properties
        self.render_mode = render_mode
        self.render_height = render_height
        self.render_width = render_width
        self.camera_id = camera_id

        # render viewer
        if self.render_mode == "human":
            from gymnasium.envs.mujoco.mujoco_rendering import Viewer

            self.viewer = Viewer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )
        else:
            self.viewer = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __repr__(self):
        description = f"All I can tell you is {self._env._task}."

    def seed(self, seed: int):
        if hasattr(self._env, "random_state"):
            self._env.random_state.seed(seed)
        else:
            self._env.task.random.seed(seed)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)

        # get stuff from the environment by stepping
        time_step = self._env.step(action)

        # open up the timestep
        obs = dmc_obs2gym_obs(time_step.observation)
        reward = time_step.reward or 0
        terminated = False
        truncated = False
        info = {}

        # set terminated and truncated
        if time_step.last() and time_step.discount == 1.0:
            truncated = True
        elif time_step.last() and time_step.discount != 1.0:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            self.seed(seed=seed)

        time_step = self._env.reset()
        obs = dmc_obs2gym_obs(time_step.observation)
        info = {}

        return obs, info

    def render(self):
        assert (
            self.render_mode in dm_control_wrapper.metadata["render_modes"]
        ), f"Can't find {self.render_mode=} in metadata with possible modes {dm_control_wrapper.metadata['render_modes']}."

        if self.render_mode == "rgb_array":
            return self._env.physics.render(
                height=self.render_height,
                width=self.render_width,
                camera_id=self.camera_id,
            )
        elif self.render_mode == "human":
            self.viewer.render()
