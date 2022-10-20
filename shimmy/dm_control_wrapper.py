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
    """Converts a dm_control spec to a gymnasium space."""
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
    """Converts a dm_control observation to a numpy array."""
    if isinstance(obs, OrderedDict) or isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = dmc_obs2gym_obs(v)
        return obs
    else:
        return np.asarray(obs)


class dm_control_wrapper(gym.Env):
    """Wrapper that converts a dm_control environment into a gymnasium environment.
    """


    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env: dm_env.Environment,
        render_mode: str,
        render_height: int = 84,
        render_width: int = 84,
        camera_id: int = 0,
    ):
        """Wrapper that converts a dm_control environment into a gymnasium environment.

        Args:
            env (dm_env.Environment): the base environment
            render_mode (str): render mode of the environment, must be specified at start
            render_height (int): the height of the render window
            render_width (int): the width of the render window
            camera_id (int): the ID of the camera that is used for rendering
        """
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

    def __getattr__(self, name):
        """__getattr__.

        Args:
            name:
        """
        return getattr(self._env, name)

    def __repr__(self):
        """__repr__.
        """
        description = f"All I can tell you is {self._env._task}."
        return description

    def seed(self, seed: int):
        """Seeds the base environment.

        Args:
            seed (int): seed
        """
        if hasattr(self._env, "random_state"):
            self._env.random_state.seed(seed)
        else:
            self._env.task.random.seed(seed)

    def step(self, action: np.ndarray):
        """Steps the underlying environment.

        Args:
            action (np.ndarray): action

        Returns:
            obs (np.ndarray): the observation of this step
            reward (float): the reward of this step
            terminated (bool): whether the environment has ended because of a terminal state
            truncated (bool): whether the enviornment has ended by exceeding the maximum time step
            info (dict): a dictionary of auxiliary information
        """
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

        if self.render_mode == "human":
            self.viewer.render()

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the underlying environment.

        Args:
            seed (Optional[int]): seed
            options (Optional[dict]): options
        """
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            self.seed(seed=seed)

        time_step = self._env.reset()
        obs = dmc_obs2gym_obs(time_step.observation)
        info = {}

        if self.render_mode == "human":
            self.viewer.render()

        return obs, info

    def render(self):
        """Renders the environment depending on what `render_modes` is set to.
        """
        assert (
            self.render_mode in dm_control_wrapper.metadata["render_modes"]
        ), f"Can't find render_mode '{self.render_mode}' in metadata with possible modes {dm_control_wrapper.metadata['render_modes']}."

        if self.render_mode == "rgb_array":
            return self._env.physics.render(
                height=self.render_height,
                width=self.render_width,
                camera_id=self.camera_id,
            )
