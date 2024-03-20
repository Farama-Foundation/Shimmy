"""Registers environments within gymnasium for optional modules."""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

import numpy as np
from gymnasium.envs.registration import register, registry

from shimmy.utils.envs_configs import (
    BSUITE_ENVS,
    DM_CONTROL_MANIPULATION_ENVS,
    DM_CONTROL_SUITE_ENVS,
)


def _register_bsuite_envs():
    """Registers all bsuite environments in gymnasium."""
    try:
        import bsuite
    except ImportError:
        return

    from bsuite.environments import Environment

    from shimmy.bsuite_compatibility import BSuiteCompatibilityV0

    # Add generic environment support
    def _make_bsuite_generic_env(env: Environment, render_mode: str):
        return BSuiteCompatibilityV0(env, render_mode=render_mode)

    register(
        "bsuite/compatibility-env-v0",
        _make_bsuite_generic_env,  # pyright: ignore[reportGeneralTypeIssues]
    )

    # register all prebuilt envs
    def _make_bsuite_env(env_id: str, **env_kwargs: Mapping[str, Any]):
        env = bsuite.load(env_id, env_kwargs)
        return BSuiteCompatibilityV0(env)

    # non deterministic envs
    nondeterministic = ["deep_sea", "bandit", "discounting_chain"]

    for env_id in BSUITE_ENVS:
        register(
            f"bsuite/{env_id}-v0",
            partial(_make_bsuite_env, env_id=env_id),
            nondeterministic=env_id in nondeterministic,
        )


def _register_dm_control_envs():
    """Registers all dm-control environments in gymnasium."""
    try:
        import dm_control
    except ImportError:
        return

    from shimmy.dm_control_compatibility import DmControlCompatibilityV0

    # Add generic environment support
    def _make_dm_control_generic_env(env, **render_kwargs):
        return DmControlCompatibilityV0(env, **render_kwargs)

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
        return DmControlCompatibilityV0(env, **render_kwargs)

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
    from dm_control.locomotion.examples import basic_cmu_2019, basic_rodent_2020

    def _make_dm_control_example_locomotion_env(
        env_fn: Callable[[np.random.RandomState | None], composer.Environment],
        random_state: np.random.RandomState | None = None,
        **render_kwargs,
    ):
        return DmControlCompatibilityV0(env_fn(random_state), **render_kwargs)

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
        return DmControlCompatibilityV0(env, **render_kwargs)

    for env_name in DM_CONTROL_MANIPULATION_ENVS:
        register(
            f"dm_control/{env_name}-v0",
            partial(_make_dm_control_manipulation_env, env_name=env_name),
            nondeterministic=env_name.startswith("reassemble_5_bricks_random_order"),
        )


def _register_dm_lab():
    try:
        import deepmind_lab
    except ImportError:
        return

    from shimmy.dm_lab_compatibility import DmLabCompatibilityV0

    def _make_dm_lab_env(
        env_id: str, observations, config: dict[str, Any], renderer: str
    ):
        env = deepmind_lab.Lab(env_id, observations, config=config, renderer=renderer)
        return DmLabCompatibilityV0(env)

    register(
        id="DmLabCompatibility-v0",
        entry_point=_make_dm_lab_env,  # pyright: ignore[reportGeneralTypeIssues]
    )


def register_gymnasium_envs():
    """This function is called when gymnasium is imported."""
    if "GymV26Environment-v0" in registry:
        registry.pop("GymV26Environment-v0")
    register(
        id="GymV26Environment-v0",
        entry_point="shimmy.openai_gym_compatibility:GymV26CompatibilityV0",
    )
    if "GymV21Environment-v0" in registry:
        registry.pop("GymV21Environment-v0")
    register(
        id="GymV21Environment-v0",
        entry_point="shimmy.openai_gym_compatibility:GymV21CompatibilityV0",
    )

    _register_bsuite_envs()
    _register_dm_control_envs()
    _register_dm_lab()
