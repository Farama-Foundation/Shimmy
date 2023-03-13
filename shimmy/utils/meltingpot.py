"""Shared utils for meltingpot."""

# pyright:  reportGeneralTypeIssues=false
from typing import Any, Mapping, OrderedDict

import numpy as np
import tree
import dm_env
from gymnasium import spaces
from pettingzoo.utils.env import ObsDict


PLAYER_STR_FORMAT = "player_{index}"
_WORLD_PREFIX = "WORLD."

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

def timestep_to_observations(timestep: dm_env.TimeStep) -> ObsDict:
    """Extracts Gymnasium-compatible observations from a melting pot timestep.

    Args:
        timestep: The dm_env timestep

    Returns:
        observation, reward, terminated, truncated, info.
    """
    gym_observations = {}
    for index, observation in enumerate(timestep.observation):
        gym_observations[PLAYER_STR_FORMAT.format(index=index)] = {
            key: value for key, value in observation.items() if _WORLD_PREFIX not in key
        }
    return gym_observations


def remove_world_observations_from_space(observation: spaces.Dict) -> spaces.Dict:
    """Removes the world observations key from a Gymnasium observation dict.

    This is used to limit the information an individual agent has access to (it cannot see the entire world).

    Args:
        observation: The melting pot observation

    Returns:
        observation: The melting pot observation, without world observations.
    """
    return spaces.Dict(
        {key: observation[key] for key in observation if _WORLD_PREFIX not in key}
    )
