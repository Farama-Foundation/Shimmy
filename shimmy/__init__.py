"""API for converting popular non-gymnasium environments to a gymnasium compatible environment."""

__version__ = "0.1.0"


try:
    from shimmy.dm_control_compatibility import DmControlCompatibilityV0
except ImportError:
    pass

try:
    from shimmy.openspiel_wrapper import OpenspielWrapperV0
except ImportError:
    pass

from shimmy.openai_gym_compatibility import GymV22CompatibilityV0, GymV26CompatibilityV0

__all__ = [
    "DmControlCompatibilityV0",
    "OpenspielWrapperV0",
    "GymV22CompatibilityV0",
    "GymV26CompatibilityV0",
]
