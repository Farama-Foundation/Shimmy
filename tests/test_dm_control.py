"""Tests the functionality of the DMEnvWrapper on dm_control envs."""

import numpy as np
import pytest
from dm_control import suite
from dm_control.suite import (
    acrobot,
    ball_in_cup,
    cartpole,
    cheetah,
    dog,
    finger,
    fish,
    hopper,
    humanoid,
    humanoid_CMU,
    lqr,
    manipulator,
    pendulum,
    point_mass,
    quadruped,
    reacher,
    stacker,
    swimmer,
    walker,
)
from gymnasium.utils.env_checker import check_env
from PIL import Image

from shimmy import DMEnvWrapperV0

# Find all domains imported.
_PASSING_DOMAINS = [
    acrobot,
    ball_in_cup,
    cartpole,
    cheetah,
    dog,
    finger,
    fish,
    hopper,
    humanoid,
    humanoid_CMU,
    manipulator,
    pendulum,
    point_mass,
    quadruped,
    reacher,
    stacker,
    swimmer,
    walker,
]

_FAILING_DOMAINS = [lqr]


@pytest.mark.parametrize("domain", _PASSING_DOMAINS)
def test_passing_domains(domain):
    """Tests the conversion of all dm_control envs."""
    # for each possible task in the domain:
    for task in domain.SUITE.values():

        # convert the task to gymnasium environment
        env = DMEnvWrapperV0(task(), render_mode="rgb_array")

        # check the environment using gymnasium
        check_env(env)

        # reset and begin test
        env.reset()
        term, trunc = False, False

        # run until termination
        while not term and not trunc:
            obs, rew, term, trunc, info = env.step(env.action_space.sample())


@pytest.mark.parametrize("domain", _FAILING_DOMAINS)
def test_failing_games(domain):
    """Ensures that failing domains are still failing."""
    with pytest.raises(Exception):
        test_passing_domains(domain)


def test_seeding():
    """Tests the seeding of the dm_control conversion wrapper."""
    # load envs
    env1 = suite.load("hopper", "stand")
    env2 = suite.load("hopper", "stand")

    # convert the environment
    env1 = DMEnvWrapperV0(env1, render_mode="rgb_array")
    env2 = DMEnvWrapperV0(env2, render_mode="rgb_array")
    env1.reset(seed=42)
    env2.reset(seed=42)

    for i in range(100):
        returns1 = env1.step(env1.action_space.sample())
        returns2 = env2.step(env2.action_space.sample())

        for stuff1, stuff2 in zip(returns1, returns2):
            if isinstance(stuff1, bool):
                assert stuff1 == stuff2, f"Incorrect returns on iteration {i}."
            elif isinstance(stuff1, np.ndarray):
                assert (stuff1 == stuff2).all(), f"Incorrect returns on iteration {i}."


@pytest.mark.parametrize("camera_id", [0, 1])
def test_render(camera_id):
    """Tests the rendering of the dm_control conversion wrapper."""
    # load an env
    env = suite.load("hopper", "stand")

    # convert the environment
    env = DMEnvWrapperV0(env, render_mode="rgb_array", camera_id=camera_id)
    env.reset()

    frames = []
    for _ in range(100):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        frames.append(env.render())

    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(
        "array.gif", save_all=True, append_images=frames[1:], duration=50, loop=0
    )
