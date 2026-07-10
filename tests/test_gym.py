"""Tests that gym compatibility environment work as expected."""

import warnings

import gym as openai_gym
import gymnasium
import pytest
from gym.spaces import Box as openai_Box
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env

import shimmy.openai_gym_compatibility
from shimmy import GymV21CompatibilityV0, GymV26CompatibilityV0

gymnasium.register_envs(shimmy.openai_gym_compatibility)

CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "This version of the mujoco environments depends on the mujoco-py bindings, which are no longer maintained and may stop working. Please upgrade to the v4 versions of the environments (which depend on the mujoco python bindings instead), unless you are trying to precisely replicate previous works).",
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "The environment CartPole-v0 is out of date. You should consider upgrading to version `v1`.",
        # Gym v21 warnings
        "Official support for the `seed` function is dropped. Standard practice is to reset gymnasium environments "
        "using `env.reset(seed=<desired seed>)`",
        "Gym v21 environment do not accept options as a reset parameter, options={}",
    ]
]
CHECK_ENV_IGNORE_WARNINGS.append(
    "`np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)"
)

# Gym V26 introduced render_mode / the new step API; V21 uses the legacy API.
if openai_gym.__version__ >= "0.26":
    GYM_COMPAT_ENV_ID = "GymV26Environment-v0"
else:
    GYM_COMPAT_ENV_ID = "GymV21Environment-v0"

# We do not test Atari environment's here because we check all variants of Pong in test_envs.py (There are too many Atari environments)
if openai_gym.__version__ >= "0.24.0":
    CLASSIC_CONTROL_ENVS = [
        env_id
        for env_id, spec in openai_gym.envs.registry.items()
        if "classic_control" in spec.entry_point
    ]
else:
    CLASSIC_CONTROL_ENVS = [
        env_id
        for env_id in openai_gym.envs.registry.env_specs
        if "classic_control" in openai_gym.envs.registry.spec(env_id).entry_point
    ]


@pytest.mark.parametrize(
    "env_id", CLASSIC_CONTROL_ENVS, ids=[env_id for env_id in CLASSIC_CONTROL_ENVS]
)
def test_gym_conversion_by_id(env_id):
    """Tests that the gym conversion works through specifying the env_id."""
    env = gymnasium.make(GYM_COMPAT_ENV_ID, env_id=env_id).unwrapped

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env, skip_render_check=True)

    for warning in caught_warnings:
        assert isinstance(warning.message, Warning)
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning.message}")

    env.close()


@pytest.mark.parametrize(
    "env_id", CLASSIC_CONTROL_ENVS, ids=[env_id for env_id in CLASSIC_CONTROL_ENVS]
)
def test_gym_conversion_instantiated(env_id):
    """Tests that the gym conversion works with an instantiated gym environment."""
    env = openai_gym.make(env_id)
    env = gymnasium.make(GYM_COMPAT_ENV_ID, env=env).unwrapped

    print("render-mode", env.render_mode)
    print("render-modes", env.metadata)
    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env, skip_render_check=True)

    for warning in caught_warnings:
        assert isinstance(warning.message, Warning)

        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning.message}")

    env.close()


class EnvWithData(openai_gym.Env):
    """Environment with data that users might want to access."""

    def __init__(self):
        """Initialises the environment with hidden data."""
        # gym 0.21 requires an explicit shape when low/high are scalars.
        self.observation_space = openai_Box(low=0, high=1, shape=())
        self.action_space = openai_Box(low=0, high=1, shape=())
        # Present so GymV26CompatibilityV0 can read it on gym<0.26 installs.
        self.render_mode = None

        self.data = 123

    def get_env_data(self):
        """Gets the environment data."""
        return self.data


def test_compatibility_get_attr():
    """Tests that the compatibility environment works with `__getattr__` for those attributes."""
    env = GymV21CompatibilityV0(env=EnvWithData())
    assert env.data == 123
    assert env.get_env_data() == 123
    env.close()

    env = GymV26CompatibilityV0(env=EnvWithData())
    assert env.data == 123
    assert env.get_env_data() == 123
    env.close()


@pytest.mark.skipif(
    openai_gym.__version__ < "0.23",
    reason="gym.spaces.Discrete gained the `start` parameter in gym 0.23",
)
def test_convert_discrete_space_preserves_start():
    """Discrete.start offset must be preserved when converting gym -> gymnasium."""
    space = openai_gym.spaces.Discrete(5, start=2)
    converted = shimmy.openai_gym_compatibility._convert_space(space)
    assert isinstance(converted, gymnasium.spaces.Discrete)
    assert converted.n == 5
    assert converted.start == 2
