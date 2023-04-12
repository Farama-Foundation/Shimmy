"""Utility functions for Melting Pot."""
# pyright: reportGeneralTypeIssues=false
# flake8: noqa F821
import dm_env
from gymnasium import spaces
from pettingzoo.utils.env import ObsDict

from shimmy.utils.dm_env import dm_spec2gym_space

PLAYER_STR_FORMAT = "player_{index}"
_WORLD_PREFIX = "WORLD."


def load_meltingpot(substrate_name: str):
    """Helper function to load Melting Pot substrates.

    Args:
        substrate_name: str

    Returns:
        env: meltingpot.python.utils.substrates.substrate.Substrate
    """
    import meltingpot.python
    from ml_collections import config_dict

    # Create env config
    substrate_name = substrate_name
    player_roles = meltingpot.python.substrate.get_config(
        substrate_name
    ).default_player_roles
    env_config = {
        "substrate": substrate_name,
        "roles": player_roles,
    }

    # Build substrate from pickle
    env_config = config_dict.ConfigDict(env_config)
    env = meltingpot.python.substrate.build(
        env_config["substrate"], roles=env_config["roles"]
    )
    return env


def timestep_to_observations(timestep: dm_env.TimeStep) -> ObsDict:
    """Extracts Gymnasium-compatible observations from a Melting Pot timestep.

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
        observation: The Melting Pot observation

    Returns:
        observation: The Melting Pot observation, without world observations.
    """
    return spaces.Dict(
        {key: observation[key] for key in observation if _WORLD_PREFIX not in key}
    )
