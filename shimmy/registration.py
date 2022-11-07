"""Registers environments within gymnasium for optional modules."""
from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from ale_py.roms import utils as rom_utils
from gymnasium.envs.registration import register

from shimmy.dm_control_compatibility import DmControlCompatibility
from shimmy.utils.envs_configs import (
    ALL_ATARI_GAMES,
    DM_CONTROL_ENVS,
    LEGACY_ATARI_GAMES,
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
                        entry_point="shimmy.ale_py_env:AtariEnv",
                        kwargs={
                            "game": rom,
                            "obs_type": obs_type,
                            **config_kwargs,
                            **flavour_kwargs,
                        },
                    )


def _register_atari_envs():
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


def register_gymnasium_envs():
    """This function is called when gymnasium is imported."""
    _register_dm_control_envs()
    _register_atari_envs()
