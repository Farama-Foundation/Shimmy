"""Utility functions for DM Control Multi-Agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

from dm_control.locomotion import soccer as dm_soccer

if TYPE_CHECKING:
    import dm_control.composer


def load_dm_control_soccer(
    team_size: int | None = 2,
    time_limit: float | None = 10.0,
    disable_walker_contacts: bool | None = False,
    enable_field_box: bool | None = True,
    terminate_on_goal: bool | None = False,
    walker_type: dm_soccer.WalkerType | None = dm_soccer.WalkerType.BOXHEAD,
) -> dm_control.composer.Environment:
    """Helper function to load a DM Control Soccer environment.

    Handles arguments which are None or unspecified (which will throw errors otherwise).

    Args:
        team_size (Optional[int]): number of players for each team
        time_limit (Optional[float]): time limit for the game
        disable_walker_contacts (Optional[bool]): flag to disable walker contacts
        enable_field_box (Optional[bool]): flag to enable field box
        terminate_on_goal (Optional[bool]): flag to terminate the environment on goal
        walker_type (Optional[dm_soccer.WalkerType]): specify walker type (BOXHEAD, ANT, or HUMANOID)

    Returns:
        env (dm_control.composer.Environment): dm control soccer environment
    """
    env = dm_soccer.load(
        team_size if team_size is not None else 2,
        time_limit if time_limit is not None else 10.0,
        disable_walker_contacts if disable_walker_contacts is not None else False,
        enable_field_box if enable_field_box is not None else True,
        terminate_on_goal if terminate_on_goal is not None else False,
        walker_type if walker_type is not None else dm_soccer.WalkerType.BOXHEAD,
    )
    return env
