"""API for converting popular non-gymnasium environments to a gymnasium compatible environment."""

__version__ = "0.2.0"


try:
    from shimmy.dm_control_compatibility import DmControlCompatibilityV0
except ImportError:
    pass

try:
    from shimmy.dm_control_multiagent_compatibility import (
        DmControlMultiAgentCompatibilityV0,
    )
except ImportError:
    pass

try:
    from shimmy.openspiel_compatibility import OpenspielCompatibilityV0
except ImportError:
    pass

try:
    from shimmy.dm_lab_compatibility import DmLabCompatibilityV0
except ImportError:
    pass

__all__ = [
    "DmControlCompatibilityV0",
    "DmControlMultiAgentCompatibilityV0",
    "OpenspielCompatibilityV0",
    "DmLabCompatibilityV0",
    "GymV22CompatibilityV0",
    "GymV26CompatibilityV0",
]
