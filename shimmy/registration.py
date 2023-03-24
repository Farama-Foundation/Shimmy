"""Registers environments within gymnasium for optional modules."""
from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np
from gymnasium.envs.registration import register, registry

from shimmy.utils.envs_configs import (
    ALL_ATARI_GAMES,
    BSUITE_ENVS,
    DM_CONTROL_MANIPULATION_ENVS,
    DM_CONTROL_SUITE_ENVS,
    LEGACY_ATARI_GAMES,
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


class GymFlavour(NamedTuple):
    """A Gym Flavour."""

    suffix: str
    kwargs: Mapping[str, Any] | Callable[[str], Mapping[str, Any]]


class GymConfig(NamedTuple):
    """A Gym Configuration."""

    version: str
    kwargs: Mapping[str, Any]
    flavours: Sequence[GymFlavour]


def _register_atari_configs(
    roms: Sequence[str],
    obs_types: Sequence[str],
    configs: Sequence[GymConfig],
    prefix: str = "",
):
    from ale_py.roms import utils as rom_utils

    for rom in roms:
        for obs_type in obs_types:
            for config in configs:
                for flavour in config.flavours:
                    name = rom_utils.rom_id_to_name(rom)
                    if obs_type == "ram":
                        name = f"{name}-ram"

                    # Parse config kwargs
                    if callable(config.kwargs):
                        config_kwargs = config.kwargs(rom)
                    else:
                        config_kwargs = config.kwargs

                    # Parse flavour kwargs
                    if callable(flavour.kwargs):
                        flavour_kwargs = flavour.kwargs(rom)
                    else:
                        flavour_kwargs = flavour.kwargs

                    # Register the environment
                    register(
                        id=f"{prefix}{name}{flavour.suffix}-{config.version}",
                        entry_point="shimmy.atari_env:AtariEnv",
                        kwargs={
                            "game": rom,
                            "obs_type": obs_type,
                            **config_kwargs,
                            **flavour_kwargs,
                        },
                    )


def _register_atari_envs():
    try:
        import ale_py
    except ImportError:
        return

    frameskip: dict[str, int] = defaultdict(lambda: 4, [("space_invaders", 3)])

    configs = [
        GymConfig(
            version="v0",
            kwargs={
                "repeat_action_probability": 0.25,
                "full_action_space": False,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[
                # Default for v0 has 10k steps, no idea why...
                GymFlavour("", {"frameskip": (2, 5)}),
                # Deterministic has 100k steps, close to the standard of 108k (30 mins gameplay)
                GymFlavour("Deterministic", lambda rom: {"frameskip": frameskip[rom]}),
                # NoFrameSkip imposes a max episode steps of frameskip * 100k, weird...
                GymFlavour("NoFrameskip", {"frameskip": 1}),
            ],
        ),
        GymConfig(
            version="v4",
            kwargs={
                "repeat_action_probability": 0.0,
                "full_action_space": False,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[
                # Unlike v0, v4 has 100k max episode steps
                GymFlavour("", {"frameskip": (2, 5)}),
                GymFlavour("Deterministic", lambda rom: {"frameskip": frameskip[rom]}),
                # Same weird frameskip * 100k max steps for v4?
                GymFlavour("NoFrameskip", {"frameskip": 1}),
            ],
        ),
    ]
    _register_atari_configs(
        LEGACY_ATARI_GAMES, obs_types=("rgb", "ram"), configs=configs
    )

    # max_episode_steps is 108k frames which is 30 mins of gameplay.
    # This corresponds to 108k / 4 = 27,000 steps
    configs = [
        GymConfig(
            version="v5",
            kwargs={
                "repeat_action_probability": 0.25,
                "full_action_space": False,
                "frameskip": 4,
                "max_num_frames_per_episode": 108_000,
            },
            flavours=[GymFlavour("", {})],
        )
    ]
    _register_atari_configs(
        ALL_ATARI_GAMES, obs_types=("rgb", "ram"), configs=configs, prefix="ALE/"
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
    _register_atari_envs()
    _register_dm_lab()
