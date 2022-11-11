"""API for converting popular non-gymnasium environments to a gymnasium compatible environment."""

try:
    from shimmy.dm_control_compatibility import (
        DmControlCompatibility as DmControlCompatibilityV0,
    )
except ImportError:
    pass

try:
    from shimmy.openspiel_compatibility import OpenspielCompatibility as OpenspielCompatibilityV0
except ImportError:
    pass

try:
    from shimmy.dmcma_compatibility import DMCMACompatibility as DMCMACompatibilityV0
except ImportError as e:
    print(e)

__version__ = "0.0.1a"
