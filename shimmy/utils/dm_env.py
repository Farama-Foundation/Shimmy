"""Utility functions for the compatibility wrappers."""

# pyright:  reportGeneralTypeIssues=false
from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import dm_env
import numpy as np
import tree

# from dm_env.specs import Array, BoundedArray, DiscreteArray
from gymnasium import spaces


def dm_spec2gym_space(spec: tree.Structure[dm_env.specs.Array]) -> spaces.Space:
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
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.floating):
            return spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
        elif np.issubdtype(spec.dtype, np.integer):
            info = np.iinfo(spec.dtype)
            return spaces.Box(info.min, info.max, spec.shape, spec.dtype)
        elif np.issubtype(spec.dtype, bool):
            return spaces.Box(int(0), int(1), spec.shape, spec.dtype)
        else:
            raise NotImplementedError(f"Unsupported dtype {spec.dtype}")
    elif isinstance(spec, (list, tuple)):
        return spaces.Tuple([dm_spec2gym_space(s) for s in spec])
    elif isinstance(spec, (OrderedDict, dict)):
        return spaces.Dict({key: dm_spec2gym_space(s) for key, s in spec.items()})
    else:
        raise ValueError(f"Unexpected spec of type {type(spec)}: {spec}")


# def dm_spec2gym_space(spec) -> spaces.Space[Any]:
#     """Converts a dm_env spec to a gymnasium space."""
#     if isinstance(spec, (OrderedDict, dict)):
#         return spaces.Dict(
#             {key: dm_spec2gym_space(value) for key, value in copy.copy(spec).items()}
#         )
#     # not possible to use isinstance due to inheritance
#     elif type(spec) is BoundedArray:
#         low = np.broadcast_to(spec.minimum, spec.shape)
#         high = np.broadcast_to(spec.maximum, spec.shape)
#         return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
#     elif type(spec) is Array:
#         if np.issubdtype(spec.dtype, np.integer):
#             low = np.iinfo(spec.dtype).min
#             high = np.iinfo(spec.dtype).max
#         elif np.issubdtype(spec.dtype, np.inexact):
#             low = float("-inf")
#             high = float("inf")
#         elif spec.dtype == "bool":
#             low = int(0)
#             high = int(1)
#         else:
#             raise ValueError(f"Unknown dtype {spec.dtype} for spec {spec}.")
#
#         return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
#     elif type(spec) is DiscreteArray:
#         return spaces.Discrete(spec.num_values)
#     else:
#         raise NotImplementedError(
#             f"Cannot convert dm_spec to gymnasium space, unknown spec: {spec}, please report."
#         )


def dm_obs2gym_obs(obs: dm_env.TimeStep.observation) -> np.ndarray | dict[str, Any]:
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


def dm_control_step2gym_step(
    timestep: dm_env.TimeStep,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
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
