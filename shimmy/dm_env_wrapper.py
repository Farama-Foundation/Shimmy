"""Wrapper to convert a dm_env environment into a gymnasium compatible environment."""
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


def dm_spec2gym_spec(spec):
    """Converts a dm_env spec to a gymnasium space."""
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dm_spec2gym_spec(v)
        return spaces.Dict(spec)
    elif type(spec) is dm_env.specs.BoundedArray:
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    elif type(spec) is dm_env.specs.Array:
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        else:
            raise ValueError(f"Unknown dtype {spec.dtype} for spec {spec}.")

        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    elif type(spec) is dm_env.specs.DiscreteArray:
        return spaces.Discrete(spec.num_values)
    else:
        raise NotImplementedError(
            f"Unknown spec {spec} for environment, not converting."
        )


def dm_obs2gym_obs(obs):
    """Converts a dm_env observation to a numpy array."""
    if isinstance(obs, OrderedDict) or isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = dm_obs2gym_obs(v)
        return obs
    else:
        return np.asarray(obs)


class DMEnvWrapperV0(gym.Env):
    """Wrapper that converts a dm_env environment into a gymnasium environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "version": 0}

    def __init__(
        self,
        env: dm_env.Environment,
        render_mode: str,
        render_height: int = 84,
        render_width: int = 84,
        camera_id: int = 0,
    ):
        """Wrapper that converts a dm_env environment into a gymnasium environment.

        Args:
            env (dm_env.Environment): the base environment
            render_mode (str): render mode of the environment, must be specified at start
            render_height (int): the height of the render window
            render_width (int): the width of the render window
            camera_id (int): the ID of the camera that is used for rendering
        """
        self._env = env

        # convert spaces
        self.observation_space = dm_spec2gym_spec(self._env.observation_spec())
        self.action_space = dm_spec2gym_spec(self._env.action_spec())

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

    def __getattr__(self, name: str):
        """__getattr__.

        Args:
            name (str): name of variable to get from the underlying environment
        """
        return getattr(self._env, name)

    def __str__(self):
        """Gives a str representation of this environment."""
        description = f"All I can tell you is {self._env._task}."
        return description

    def step(self, action: np.ndarray):
        """Steps the underlying environment.

        Args:
            action (np.ndarray): action

        Returns:
            obs (np.ndarray): the observation of this step
            reward (float): the reward of this step
            terminated (bool): whether the environment has ended because of a terminal state
            truncated (bool): whether the environment has ended by exceeding the maximum time step
            info (dict): a dictionary of auxiliary information
        """
        assert self.action_space.contains(action)

        # get stuff from the environment by stepping
        timestep = self._env.step(action)

        # open up the timestep and process reward and observation
        obs = dm_obs2gym_obs(timestep.observation)
        reward = timestep.reward or 0

        # set terminated and truncated
        if timestep.last() and timestep.discount != 0:
            terminated = False
            truncated = True
        elif timestep.last() and timestep.discount == 0:
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = False

        info = dict()
        info["timestep.last"] = timestep.last()
        info["timestep.discount"] = timestep.discount
        info["timestep.step_type"] = timestep.step_type

        if self.render_mode == "human":
            self.viewer.render()

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the underlying environment.

        Args:
            seed (Optional[int]): seed
            options (Optional[dict]): options
        """
        super().reset(seed=seed)

        if seed is not None:
            if hasattr(self._env, "random_state"):
                self._env.random_state.seed(seed)
            else:
                self._env.task.random.seed(seed)

        time_step = self._env.reset()
        obs = dm_obs2gym_obs(time_step.observation)
        info = {}

        if self.render_mode == "human":
            self.viewer.render()

        return obs, info

    def render(self):
        """Renders the environment depending on what `render_modes` is set to."""
        assert (
            self.render_mode in DMEnvWrapper.metadata["render_modes"]
        ), f"Can't find render_mode '{self.render_mode}' in metadata with possible modes {DMEnvWrapper.metadata['render_modes']}."

        if self.render_mode == "rgb_array":
            return self._env.physics.render(
                height=self.render_height,
                width=self.render_width,
                camera_id=self.camera_id,
            )
