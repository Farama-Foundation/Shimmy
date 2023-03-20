"""Tests the multi-agent dm-control soccer environment."""

import pytest
from dm_control.locomotion import soccer as dm_soccer
from pettingzoo.test import parallel_api_test

from shimmy.dm_control_multiagent_compatibility import (
    DmControlMultiAgentCompatibilityV0,
)

WALKER_TYPES = [
    dm_soccer.WalkerType.BOXHEAD,
    dm_soccer.WalkerType.ANT,
    dm_soccer.WalkerType.HUMANOID,
]


@pytest.mark.parametrize("walker_type", WALKER_TYPES)
def test_check_env(walker_type):
    """Check that environment pass the pettingzoo check_env."""
    env = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=walker_type,
    )

    env = DmControlMultiAgentCompatibilityV0(env)

    parallel_api_test(env)

    env.close()
