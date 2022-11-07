"""Registers environments within gymnasium for optional modules."""
from __future__ import annotations

from functools import partial
from typing import Any

from gymnasium.envs.registration import register

from shimmy.dm_control_compatibility import DmControlCompatibility

DM_CONTROL_ENVS = (
    ("acrobot", "swingup"),
    ("acrobot", "swingup_sparse"),
    ("ball_in_cup", "catch"),
    ("cartpole", "balance"),
    ("cartpole", "balance_sparse"),
    ("cartpole", "swingup"),
    ("cartpole", "swingup_sparse"),
    ("cartpole", "two_poles"),
    ("cartpole", "three_poles"),
    ("cheetah", "run"),
    ("dog", "stand"),
    ("dog", "walk"),
    ("dog", "trot"),
    ("dog", "run"),
    ("dog", "fetch"),
    ("finger", "spin"),
    ("finger", "turn_easy"),
    ("finger", "turn_hard"),
    ("fish", "upright"),
    ("fish", "swim"),
    ("hopper", "stand"),
    ("hopper", "hop"),
    ("humanoid", "stand"),
    ("humanoid", "walk"),
    ("humanoid", "run"),
    ("humanoid", "run_pure_state"),
    ("humanoid_CMU", "stand"),
    ("humanoid_CMU", "run"),
    ("lqr", "lqr_2_1"),
    ("lqr", "lqr_6_2"),
    ("manipulator", "bring_ball"),
    ("manipulator", "bring_peg"),
    ("manipulator", "insert_ball"),
    ("manipulator", "insert_peg"),
    ("pendulum", "swingup"),
    ("point_mass", "easy"),
    ("point_mass", "hard"),
    ("quadruped", "walk"),
    ("quadruped", "run"),
    ("quadruped", "escape"),
    ("quadruped", "fetch"),
    ("reacher", "easy"),
    ("reacher", "hard"),
    ("stacker", "stack_2"),
    ("stacker", "stack_4"),
    ("swimmer", "swimmer6"),
    ("swimmer", "swimmer15"),
    ("walker", "stand"),
    ("walker", "walk"),
    ("walker", "run"),
)


def _register_dm_control_envs():
    """Registers all dm-control environments in gymnasium."""
    try:
        import dm_control.suite
    except ImportError:
        return

    def _make_dm_control_env(
        domain_name: str,
        task_name: str,
        task_kwargs: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        visualize_reward: bool = False,
        **render_kwargs,
    ):
        env = dm_control.suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
        return DmControlCompatibility(env, **render_kwargs)

    for _domain_name, _task_name in DM_CONTROL_ENVS:
        register(
            f"dm_control/{_domain_name}-{_task_name}-v0",
            partial(
                _make_dm_control_env, domain_name=_domain_name, task_name=_task_name
            ),
        )


def register_gymnasium_envs():
    """This function is called when gymnasium is imported."""
    _register_dm_control_envs()

    register(
        "GymV26Compatibility", "shimmy.openai_gym_compatibility:GymV26Compatibility"
    )
    register(
        "GymV22Compatibility", "shimmy.openai_gym_compatibility:GymV22Compatibility"
    )
