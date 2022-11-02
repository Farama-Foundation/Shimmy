"""API for converting popular non-gymnasium environments to a gymnasium compatible environment."""

from shimmy.dm_control_compatibility import (
    DmControlCompatibility as DmControlCompatibilityV0,
)
from shimmy.openspiel_wrapper import OpenspielWrapper as OpenspielWrapperV0

__version__ = "0.0.1a"
