"""Tests the functionality of the BSuiteCompatibilityV0 on bsuite envs."""
import pickle
import warnings

import bsuite
import gymnasium as gym
import pytest
from gymnasium.envs.registration import registry
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env, data_equivalence

BSUITE_ENV_IDS = [
    env_id
    for env_id in registry
    if env_id.startswith("bsuite") and env_id != "bsuite/compatibility-env-v0"
]


def test_bsuite_suite_envs():
    """Tests that all BSUITE_ENVS are equal to the known bsuite tasks."""
    env_ids = [env_id.split("/")[-1].split("-")[0] for env_id in BSUITE_ENV_IDS]
    assert list(bsuite._bsuite.EXPERIMENT_NAME_TO_ENVIRONMENT.keys()) == env_ids


BSUITE_ENV_SETTINGS = dict()
BSUITE_ENV_SETTINGS["bsuite/bandit-v0"] = dict()
BSUITE_ENV_SETTINGS["bsuite/bandit_noise-v0"] = dict(
    noise_scale=1, seed=42, mapping_seed=42
)
BSUITE_ENV_SETTINGS["bsuite/bandit_scale-v0"] = dict(
    reward_scale=1, seed=42, mapping_seed=42
)
BSUITE_ENV_SETTINGS["bsuite/cartpole-v0"] = dict()
BSUITE_ENV_SETTINGS["bsuite/cartpole_noise-v0"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/cartpole_scale-v0"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/cartpole_swingup-v0"] = dict()
BSUITE_ENV_SETTINGS["bsuite/catch-v0"] = dict()
BSUITE_ENV_SETTINGS["bsuite/catch_noise-v0"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/catch_scale-v0"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/deep_sea-v0"] = dict(size=42)
BSUITE_ENV_SETTINGS["bsuite/deep_sea_stochastic-v0"] = dict(size=42)
BSUITE_ENV_SETTINGS["bsuite/discounting_chain-v0"] = dict()
BSUITE_ENV_SETTINGS["bsuite/memory_len-v0"] = dict(memory_length=8)
BSUITE_ENV_SETTINGS["bsuite/memory_size-v0"] = dict(num_bits=8)
BSUITE_ENV_SETTINGS["bsuite/mnist-v0"] = dict()
BSUITE_ENV_SETTINGS["bsuite/mnist_noise-v0"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/mnist_scale-v0"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/mountain_car-v0"] = dict()
BSUITE_ENV_SETTINGS["bsuite/mountain_car_noise-v0"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/mountain_car_scale-v0"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["bsuite/umbrella_distract-v0"] = dict(n_distractor=3)
BSUITE_ENV_SETTINGS["bsuite/umbrella_length-v0"] = dict(chain_length=3)

# todo - gymnasium v27 should remove the need for some of these warnings
CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (28, 28)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (42, 42)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (10, 5)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (1, 1)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (1, 2)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (1, 3)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (1, 6)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (1, 8)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (1, 10)",
    ]
]


@pytest.mark.parametrize("env_id", BSUITE_ENV_IDS)
def test_check_env(env_id):
    """Check that environment pass the gymnasium check_env."""
    env = gym.make(env_id, **BSUITE_ENV_SETTINGS[env_id])

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_id", BSUITE_ENV_IDS)
def test_seeding(env_id):
    """Test that dm-control seeding works."""
    if gym.spec(env_id).nondeterministic:
        return

    env_1 = gym.make(env_id, **BSUITE_ENV_SETTINGS[env_id])
    env_2 = gym.make(env_id, **BSUITE_ENV_SETTINGS[env_id])

    obs_1, info_1 = env_1.reset(seed=42)
    obs_2, info_2 = env_2.reset(seed=42)
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(100):
        actions = int(env_1.action_space.sample())
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()


# Without EzPickle:_register_bsuite_envs.<locals>._make_bsuite_env cannot be pickled
# With EzPickle: maximum recursion limit reached
FAILING_PICKLE_ENVS = [
    "bsuite/bandit_noise-v0",
    "bsuite/bandit_scale-v0",
    "bsuite/cartpole-v0",
    "bsuite/cartpole_noise-v0",
    "bsuite/cartpole_scale-v0",
    "bsuite/cartpole_swingup-v0",
    "bsuite/catch_noise-v0",
    "bsuite/catch_scale-v0",
    "bsuite/mnist_noise-v0",
    "bsuite/mnist_scale-v0",
    "bsuite/mountain_car_noise-v0",
    "bsuite/mountain_car_scale-v0",
]

PASSING_PICKLE_ENVS = [
    "bsuite/mnist-v0",
    "bsuite/umbrella_length-v0",
    "bsuite/discounting_chain-v0",
    "bsuite/deep_sea-v0",
    "bsuite/umbrella_distract-v0",
    "bsuite/catch-v0",
    "bsuite/memory_len-v0",
    "bsuite/mountain_car-v0",
    "bsuite/memory_size-v0",
    "bsuite/deep_sea_stochastic-v0",
    "bsuite/bandit-v0",
]


@pytest.mark.parametrize("env_id", PASSING_PICKLE_ENVS)
def test_pickle(env_id):
    """Test that pickling works."""
    env_1 = gym.make(env_id, **BSUITE_ENV_SETTINGS[env_id])
    env_2 = pickle.loads(pickle.dumps(env_1))

    obs_1, info_1 = env_1.reset(seed=42)
    obs_2, info_2 = env_2.reset(seed=42)
    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)
    for _ in range(100):
        actions = int(env_1.action_space.sample())
        obs_1, reward_1, term_1, trunc_1, info_1 = env_1.step(actions)
        obs_2, reward_2, term_2, trunc_2, info_2 = env_2.step(actions)
        assert data_equivalence(obs_1, obs_2)
        assert reward_1 == reward_2
        assert term_1 == term_2 and trunc_1 == trunc_2
        assert data_equivalence(info_1, info_2)

    env_1.close()
    env_2.close()
