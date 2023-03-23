"""Utility functions for the compatibility wrappers."""
from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import dm_env
import numpy as np
import tree
from gymnasium import spaces


def dm_spec2gym_space(spec) -> spaces.Space[Any]:
    """Converts a dm_env nested structure of specs to a Gymnasium Space.

    BoundedArray is converted to Box Gymnasium spaces. DiscreteArray is converted to
    Discrete Gymnasium spaces. Using Tuple and Dict spaces recursively as needed.

    Args:
        spec: The nested structure of specs

    Returns:
        The Gymnasium space corresponding to the given spec.
    """
    if isinstance(spec, dm_env.specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low, high, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.inexact):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        elif spec.dtype == "bool":
            return spaces.Box(int(0), int(1), spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f"Unsupported dtype {spec.dtype}")
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([dm_spec2gym_space(s) for s in spec])
    elif isinstance(spec, (OrderedDict, dict)):
        return spaces.Dict(
            {key: dm_spec2gym_space(value) for key, value in copy.copy(spec).items()}
        )
    else:
        raise ValueError(
            f"Unexpected spec of type {type(spec)}: {spec}. Please report."
        )


def dm_obs2gym_obs(obs) -> np.ndarray | dict[str, Any]:
    """Converts a dm_env observation to a gymnasium observation.

    Array observations are converted to numpy arrays. Dict observations are converted recursively per key.

    Args:
        obs: The dm_env observation

    Returns:
        The Gymnasium-compatible observation.
    """
    if isinstance(obs, (OrderedDict, dict)):
        return {key: dm_obs2gym_obs(value) for key, value in copy.copy(obs).items()}
    else:
        return np.asarray(obs)


def dm_env_step2gym_step(timestep) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Converts a dm_env timestep to the required return info from Gymnasium step() function.

    Args:
        timestep: The dm_env timestep

    Returns:
        observation, reward, terminated, truncated, info.
    """
    obs = dm_obs2gym_obs(timestep.observation)
    reward = timestep.reward or 0

    # set terminated and truncated
    terminated, truncated = False, False
    if timestep.last():
        # https://github.com/deepmind/dm_env/blob/master/docs/index.md#example-sequences
        if timestep.discount > 0:
            truncated = True
        else:
            terminated = True

    info = {
        "timestep.discount": timestep.discount,
        "timestep.step_type": timestep.step_type,
    }

    return (
        obs,
        reward,
        terminated,
        truncated,
        info,
    )
