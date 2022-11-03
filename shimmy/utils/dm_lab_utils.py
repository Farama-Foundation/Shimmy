"""Utility functions for the compatibility wrappers."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from gymnasium import spaces

def dm_lab_spec2gym_space(spec) -> spaces.Space[Any]:
    """Converts a dm_lab spec to a gymnasium space."""
    if isinstance(spec, list):
        expanded = {}
        for desc in spec:
            assert "name" in desc, f"Can't find name for the description: {desc} in spec."

            # some observation spaces have a string description, we ignore those for now
            if "dtype" in desc:
                if desc["dtype"] == str:
                    continue

            expanded[desc["name"]] = dm_lab_spec2gym_space(desc)

        return spaces.Dict(expanded)
    if isinstance(spec, (OrderedDict, dict)):
        # this is an observation space
        if "shape" in spec and "dtype" in spec:
            assert "dtype" in spec, f"Can't find dtype for spec: {spec}."
            assert "shape" in spec, f"Can't find shape for spec: {spec}."

            low = 0
            high = 0
            if np.issubdtype(spec["dtype"], np.integer):
                low = np.iinfo(spec["dtype"]).min
                high = np.iinfo(spec["dtype"]).max
            elif np.issubdtype(spec["dtype"], np.inexact):
                low = float("-inf")
                high = float("inf")

            return spaces.Box(low=low, high=high, shape=spec["shape"], dtype=spec["dtype"])

        # this is an action space
        elif "min" in spec and "max" in spec:
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


def dm_lab_obs2gym_obs(obs) -> np.ndarray | dict[str, Any]:
    """Converts a dm_lab observation to a gymnasium observation."""
    print(obs)
