"""Utility functions for the compatibility wrappers."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from gymnasium import spaces


def load_dm_lab(
    level_name: str = "lt_chasm",
    observations: str | None = "RGBD",
    renderer: str | None = "hardware",
    width: int | None = 320,
    height: int | None = 240,
    fps: int | None = 60,
    mixerSeed: int | None = 0,
    levelDirectory: str | None = "",
    appendCommand: str | None = "",
    botCount: int | None = None,
):
    """Helper function to load a DM Lab environment.

    Handles arguments which are None or unspecified (which will throw errors otherwise).

    Args:
        level_name (str): name of level to load
        observations: (Optional[str]): type of observations to use (default: "RGBD")
        renderer (Optional[str]): renderer to use (default: "hardware")
        width (Optional[int]): horizontal resolution of the observation frames (default: 240)
        height (Optional[int]): vertical resolution of the observation frames (default: 320)
        fps (Optional[int]): frames-per-second (default: 60)
        mixerSeed (Optional[int]):	value combined with each of the seeds fed to the environment to define unique subsets of seeds (default: 0)
        levelDirectory (Optional[str]): optional path to level directory (relative paths are relative to game_scripts/levels)
        appendCommand (Optional[str]): Commands for the internal Quake console
        botCount (Optional[int]): number of bots to use

    Returns:
        env: DM Lab environment
    """
    import deepmind_lab

    if observations is not None:
        obs = [observations]
    else:
        obs = ["RGBD"]

    renderer = renderer

    # botCount is a specific config option for certain level and may result in errors
    try:
        config = {
            "width": str(width),
            "height": str(height),
            "fps": str(fps),
            "levelDirectory": levelDirectory,
            "appendCommand": appendCommand,
            "mixerSeed": str(mixerSeed),
            "botCount": str(botCount),
        }
        return deepmind_lab.Lab(level_name, obs, config=config, renderer=renderer)

    except Exception:
        pass

    # try without botCount configuration option as it is not used for all environments
    try:
        config = {
            "width": str(width),
            "height": str(height),
            "fps": str(fps),
            "levelDirectory": levelDirectory,
            "appendCommand": appendCommand,
        }
        return deepmind_lab.Lab("lt_chasm", obs, config=config, renderer=renderer)

    except Exception as e:
        print("Could not load DM Lab environment with given configuration: ", e)


def dm_lab_obs2gym_obs_space(observation: dict) -> spaces.Space[Any]:
    """Gets the observation spec from a single observation."""
    assert isinstance(
        observation, (OrderedDict, dict)
    ), f"Observation must be a dict, got {observation}"

    all_spaces = dict()
    for key, value in observation.items():
        dtype = value.dtype

        low = None
        high = None
        if np.issubdtype(dtype, np.integer):
            low = np.iinfo(dtype).min
            high = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        else:
            raise ValueError(f"Unknown dtype {dtype}.")

        all_spaces[key] = spaces.Box(low=low, high=high, shape=value.shape, dtype=dtype)

    return spaces.Dict(all_spaces)


def dm_lab_spec2gym_space(spec) -> spaces.Space[Any]:
    """Converts a dm_lab spec to a gymnasium space."""
    if isinstance(spec, list):
        expanded = {}
        for desc in spec:
            assert (
                "name" in desc
            ), f"Can't find name for the description: {desc} in spec."

            # some observation spaces have a string description, we ignore those for now
            if "dtype" in desc:
                if desc["dtype"] == str:
                    continue

            expanded[desc["name"]] = dm_lab_spec2gym_space(desc)

        return spaces.Dict(expanded)
    if isinstance(spec, (OrderedDict, dict)):
        # this is an action space
        if "min" in spec and "max" in spec:
            return spaces.Box(low=spec["min"], high=spec["max"], dtype=np.float64)

        # we dk wtf it is here
        else:
            raise NotImplementedError(
                f"Unknown spec definition: {spec}, please report."
            )

    else:
        raise NotImplementedError(
            f"Cannot convert dm_spec to gymnasium space, unknown spec: {spec}, please report."
        )
