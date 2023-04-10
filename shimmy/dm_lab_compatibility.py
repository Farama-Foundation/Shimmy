"""Wrapper to convert a DM Lab environment into a gymnasium compatible environment."""
# pyright: reportOptionalMemberAccess=false
from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.utils import EzPickle

from shimmy.utils.dm_lab import (
    dm_lab_obs2gym_obs_space,
    dm_lab_spec2gym_space,
    load_dm_lab,
)


class DmLabCompatibilityV0(gym.Env[ObsType, Dict[str, np.ndarray]], EzPickle):
    """This compatibility wrapper converts a DM Lab environment into a gymnasium environment.

    DeepMind Lab is a 3D learning environment based on id Software's Quake III Arena via ioquake3 and
    other open source software. DeepMind Lab provides a suite of challenging 3D navigation and
    puzzle-solving tasks for learning agents.
    Its primary purpose is to act as a testbed for research in artificial intelligence, especially deep reinforcement learning.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        env: Any | None = None,
        level_name: str | None = None,
        observations: str | None = None,
        renderer: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        mixerSeed: int | None = None,
        levelDirectory: str | None = None,
        appendCommand: str | None = None,
        botCount: int | None = None,
        render_mode: None = None,
    ):
        """Wrapper to convert a DM Lab environment into a Gymnasium environment.

        Note: to wrap an existing environment, only the env and render_mode arguments can be specified.
        All other arguments are specific to DM Lab and will be used to load a new environment.

        Args:
            env (Optional[Any]): existing DM Lab environment to wrap
            level_name (Optional[str]): name of level to load
            observations: (Optional[str]): type of observations to use (default: "RGBD")
            renderer (Optional[str]): renderer to use (default: "hardware")
            width (Optional[int]): horizontal resolution of the observation frames (default: 240)
            height (Optional[int]): vertical resolution of the observation frames (default: 320)
            fps (Optional[int]): frames-per-second (default: 60)
            mixerSeed (Optional[int]):	value combined with each of the seeds fed to the environment to define unique subsets of seeds (default: 0)
            levelDirectory (Optional[str]): optional path to level directory (relative paths are relative to game_scripts/levels)
            appendCommand (Optional[str]): Commands for the internal Quake console
            botCount (Optional[int]): number of bots to use
            render_mode (Optional[str]): rendering mode to use (choices: "human", "none"
        """
        EzPickle.__init__(
            self,
            env,
            level_name,
            observations,
            renderer,
            width,
            height,
            fps,
            mixerSeed,
            levelDirectory,
            appendCommand,
            botCount,
            render_mode,
        )

        DM_LAB_ARGS = [
            level_name,
            observations,
            renderer,
            width,
            height,
            fps,
            mixerSeed,
            levelDirectory,
            appendCommand,
            botCount,
        ]

        # Only one of env and DM_LAB_ARGS can be provided, the others should be None.
        if env is None and all(arg is None for arg in DM_LAB_ARGS):
            raise ValueError(
                "No environment provided. Use `env` to specify an existing environment, or load an environment by specifying at least one of `team_size`, `time_limit`, `disable_walker_contacts`, `enable_field_box` `terminate_on_goal`, or `walker_type`."
            )
        elif env is not None and any(arg is not None for arg in DM_LAB_ARGS):
            raise ValueError(
                "Two environments provided. Use `env` to specify an existing environment, or load an environment by specifying at least one of `team_size`, `time_limit`, `disable_walker_contacts`, `enable_field_box` `terminate_on_goal`, or `walker_type`."
            )
        elif any(arg is not None for arg in DM_LAB_ARGS):
            if level_name is None:
                raise ValueError(
                    "Level name must be specified to load a DM Lab environment."
                )
            else:
                if render_mode == "human":
                    renderer = "sdl"

                self._env = load_dm_lab(
                    level_name,
                    observations,
                    renderer,
                    width,
                    height,
                    fps,
                    mixerSeed,
                    levelDirectory,
                    appendCommand,
                    botCount,
                )
        elif env is not None:
            self._env = env

        self._env.reset()
        self.observation_space = dm_lab_obs2gym_obs_space(self._env.observations())
        self.action_space = dm_lab_spec2gym_space(env.action_spec())

        self.render_mode = render_mode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the dm-lab environment."""
        super().reset(seed=seed)

        self._env.reset(seed=seed)
        info = {}

        if seed is not None:
            print(
                "Warning: DM-lab environments must be seeded in initialization, rather than with reset(seed)."
            )
        return (
            self._env.observations(),
            info,
        )

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Steps through the dm-lab environment."""
        # there's some funky quantization happening here, dm_lab only accepts ints as actions
        action_array = np.array([a[0] for a in action.values()], dtype=np.intc)
        reward = self._env.step(action_array)

        obs = self._env.observations()
        terminated = not self._env.is_running()
        truncated = False
        info = {}

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        """Renders the DM Lab env."""
        raise NotImplementedError(
            "Rendering environment directly is not supported. Pass `renderer='sdl'` to enable human rendering."
        )

    def close(self):
        """Closes the environment."""
        self._env.close()

    def __getattr__(self, item: str):
        """If the attribute is missing, try getting the attribute from DM Lab env."""
        return getattr(self._env, item)
