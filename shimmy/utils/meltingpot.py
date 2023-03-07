"""Shared utils for meltingpot."""

# pyright:  reportGeneralTypeIssues=false
from typing import Any, Mapping, OrderedDict

import dm_env
from gymnasium import spaces
from pettingzoo.utils.env import ObsDict

from shimmy.utils.dm_env import dm_spec2gym_space

PLAYER_STR_FORMAT = "player_{index}"
_WORLD_PREFIX = "WORLD."


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
