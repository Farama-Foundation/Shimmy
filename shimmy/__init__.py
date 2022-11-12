"""API for converting popular non-gymnasium environments to a gymnasium compatible environment."""

__version__ = "0.1.0"


try:
    from shimmy.dm_control_compatibility import (
        DmControlCompatibility as DmControlCompatibilityV0,
    )
except ImportError:
    pass

try:
    from shimmy.openspiel_wrapper import OpenspielWrapper as OpenspielWrapperV0
except ImportError:
    pass
