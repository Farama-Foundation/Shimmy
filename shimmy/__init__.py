"""API for converting popular non-gymnasium environments to a gymnasium compatible environment."""
from __future__ import annotations

from typing import Any

from shimmy.dm_lab_compatibility import DmLabCompatibilityV0
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0, GymV26CompatibilityV0


class NotInstallClass:
    """Rather than an attribute error, this raises a more helpful import error with install instructions for shimmy."""

    def __init__(self, install_message: str, import_exception: ImportError):
        self.install_message = install_message
        self.import_exception = import_exception

    def __call__(self, *args: list[Any], **kwargs: Any):
        """Acts like the `__init__` for the class."""
        raise ImportError(self.install_message) from self.import_exception


try:
    from shimmy.dm_control_compatibility import DmControlCompatibilityV0
except ImportError as e:
    DmControlCompatibilityV0 = NotInstallClass(
        "Dm-control is not installed, run `pip install 'shimmy[dm-control]'`", e
    )


try:
    from shimmy.dm_control_multiagent_compatibility import (
        DmControlMultiAgentCompatibilityV0,
    )
except ImportError as e:
    DmControlMultiAgentCompatibilityV0 = NotInstallClass(
        "Dm-control or PettingZoo is not installed, run `pip install 'shimmy[dm-control-multi-agent]'`",
        e,
    )

try:
    from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0
except ImportError as e:
    OpenSpielCompatibilityV0 = NotInstallClass(
        "OpenSpiel or PettingZoo is not installed, run `pip install 'shimmy[openspiel]'`",
        e,
    )

try:
    from shimmy.bsuite_compatibility import BSuiteCompatibilityV0
except ImportError as e:
    BSuiteCompatibilityV0 = NotInstallClass(
        "BSuite is not installed, run `pip install 'shimmy[bsuite]'`",
        e,
    )

try:
    from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0
except ImportError as e:
    MeltingPotCompatibilityV0 = NotInstallClass(
        "Melting Pot or PettingZoo is not installed, run `pip install 'shimmy[melting-pot]' and install Melting Pot via https://github.com/deepmind/meltingpot#installation`",
        e,
    )

__all__ = [
    "BSuiteCompatibilityV0",
    "DmControlCompatibilityV0",
    "DmControlMultiAgentCompatibilityV0",
    "OpenSpielCompatibilityV0",
    "DmLabCompatibilityV0",
    "GymV21CompatibilityV0",
    "GymV26CompatibilityV0",
    "MeltingPotCompatibilityV0",
]


__version__ = "1.0.0"


try:
    import sys

    from farama_notifications import notifications

    if "shimmy" in notifications and __version__ in notifications["shimmy"]:
        print(notifications["shimmy"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass
