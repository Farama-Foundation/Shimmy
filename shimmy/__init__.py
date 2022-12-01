"""API for converting popular non-gymnasium environments to a gymnasium compatible environment."""

__version__ = "0.1.0"


try:
    from shimmy.dm_control_compatibility import DmControlCompatibility
except ImportError:
    pass

try:
    from shimmy.openspiel_wrapper import OpenspielWrapperV0
except ImportError:
    pass

from shimmy.openai_gym_compatibility import GymV22Compatibility, GymV26Compatibility

__all__ = [
    "DmControlCompatibility",
    "OpenspielWrapperV0",
    "GymV22Compatibility",
    "GymV26Compatibility",
]
