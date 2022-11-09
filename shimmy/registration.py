"""Registers environments within gymnasium for optional modules."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
from gymnasium.envs.registration import register

from shimmy.dm_control_compatibility import DmControlCompatibility

DM_CONTROL_SUITE_ENVS = (
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


DM_CONTROL_MANIPULATION_ENVS = (
    "stack_2_bricks_features",
    "stack_2_bricks_vision",
    "stack_2_bricks_moveable_base_features",
    "stack_2_bricks_moveable_base_vision",
    "stack_3_bricks_features",
    "stack_3_bricks_vision",
    "stack_3_bricks_random_order_features",
    "stack_2_of_3_bricks_random_order_features",
    "stack_2_of_3_bricks_random_order_vision",
    "reassemble_3_bricks_fixed_order_features",
    "reassemble_3_bricks_fixed_order_vision",
    "reassemble_5_bricks_random_order_features",
    "reassemble_5_bricks_random_order_vision",
    "lift_brick_features",
    "lift_brick_vision",
    "lift_large_box_features",
    "lift_large_box_vision",
    "place_brick_features",
    "place_brick_vision",
    "place_cradle_features",
    "place_cradle_vision",
    "reach_duplo_features",
    "reach_duplo_vision",
    "reach_site_features",
    "reach_site_vision",
)


def _register_dm_control_envs():
    """Registers all dm-control environments in gymnasium."""
    try:
        import dm_control
    except ImportError:
        return

    # Add generic environment support
    def _make_dm_control_generic_env(env, **render_kwargs):
        return DmControlCompatibility(env, **render_kwargs)

    register("dm_control/compatibility-env-v0", _make_dm_control_generic_env)

    # Register all suite environments
    import dm_control.suite

    def _make_dm_control_suite_env(
        domain_name: str,
        task_name: str,
        task_kwargs: dict[str, Any] | None = None,
        environment_kwargs: dict[str, Any] | None = None,
        visualize_reward: bool = False,
        **render_kwargs,
    ):
        """The entry_point function for registration of dm-control environments."""
        env = dm_control.suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
        return DmControlCompatibility(env, **render_kwargs)

    for _domain_name, _task_name in DM_CONTROL_SUITE_ENVS:
        register(
            f"dm_control/{_domain_name}-{_task_name}-v0",
            partial(
                _make_dm_control_suite_env,
                domain_name=_domain_name,
                task_name=_task_name,
            ),
        )

    # Register all example locomotion environments
    # Listed in https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/examples/examples_test.py
    from dm_control import composer
    from dm_control.locomotion.examples import (
        basic_cmu_2019,
        basic_rodent_2020,
        cmu_2020_tracking,
    )

    def _make_dm_control_example_locomotion_env(
        env_fn: Callable[[np.random.RandomState | None], composer.Environment],
        random_state: np.random.RandomState | None = None,
        **render_kwargs,
    ):
        return DmControlCompatibility(env_fn(random_state), **render_kwargs)

    for locomotion_env, nondeterministic in (
        (basic_cmu_2019.cmu_humanoid_run_walls, False),
        (basic_cmu_2019.cmu_humanoid_run_gaps, False),
        (basic_cmu_2019.cmu_humanoid_go_to_target, False),
        (basic_cmu_2019.cmu_humanoid_maze_forage, True),
        (basic_cmu_2019.cmu_humanoid_heterogeneous_forage, True),
        (basic_rodent_2020.rodent_escape_bowl, False),
        (basic_rodent_2020.rodent_run_gaps, False),
        (basic_rodent_2020.rodent_maze_forage, True),
        (basic_rodent_2020.rodent_two_touch, True),
        # (cmu_2020_tracking.cmu_humanoid_tracking, False),
    ):
        register(
            f"dm_control/{locomotion_env.__name__.title().replace('_', '')}-v0",
            partial(_make_dm_control_example_locomotion_env, env_fn=locomotion_env),
            nondeterministic=nondeterministic,
        )

    # Register all manipulation environments
    import dm_control.manipulation

    def _make_dm_control_manipulation_env(env_name: str, **render_kwargs):
        env = dm_control.manipulation.load(env_name)
        return DmControlCompatibility(env, **render_kwargs)

    for env_name in DM_CONTROL_MANIPULATION_ENVS:
        register(
            f"dm_control/{env_name}-v0",
            partial(_make_dm_control_manipulation_env, env_name=env_name),
            nondeterministic=env_name.startswith("reassemble_5_bricks_random_order"),
        )


def register_gymnasium_envs():
    """This function is called when gymnasium is imported."""
    _register_dm_control_envs()
