"""Tests the functionality of the BSuiteCompatibilityV0 on bsuite envs."""
import warnings

import pytest
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env, data_equivalence

import bsuite
from shimmy.bsuite_compatibility import BSuiteCompatibilityV0

BSUITE_NAME_TO_LOADERS = bsuite._bsuite.EXPERIMENT_NAME_TO_ENVIRONMENT
BSUITE_ENV_SETTINGS = dict()
BSUITE_ENV_SETTINGS["bandit"] = dict()
BSUITE_ENV_SETTINGS["bandit_noise"] = dict(noise_scale=1, seed=42, mapping_seed=42)
BSUITE_ENV_SETTINGS["bandit_scale"] = dict(reward_scale=1, seed=42, mapping_seed=42)
BSUITE_ENV_SETTINGS["cartpole"] = dict()
BSUITE_ENV_SETTINGS["cartpole_noise"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["cartpole_scale"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["cartpole_swingup"] = dict()
BSUITE_ENV_SETTINGS["catch"] = dict()
BSUITE_ENV_SETTINGS["catch_noise"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["catch_scale"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["deep_sea"] = dict(size=42)
BSUITE_ENV_SETTINGS["deep_sea_stochastic"] = dict(size=42)
BSUITE_ENV_SETTINGS["discounting_chain"] = dict()
BSUITE_ENV_SETTINGS["memory_len"] = dict(memory_length=8)
BSUITE_ENV_SETTINGS["memory_size"] = dict(num_bits=8)
BSUITE_ENV_SETTINGS["mnist"] = dict()
BSUITE_ENV_SETTINGS["mnist_noise"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["mnist_scale"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["mountain_car"] = dict()
BSUITE_ENV_SETTINGS["mountain_car_noise"] = dict(noise_scale=1, seed=42)
BSUITE_ENV_SETTINGS["mountain_car_scale"] = dict(reward_scale=1, seed=42)
BSUITE_ENV_SETTINGS["umbrella_distract"] = dict(n_distractor=3)
BSUITE_ENV_SETTINGS["umbrella_length"] = dict(chain_length=3)

# todo - gymnasium v27 should remove the need for some of these warnings
CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (28, 28)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (42, 42)",
        "A Box observation space has an unconventional shape (neither an image, nor a 1D vector). We recommend flattening the observation to have only a 1D vector or use a custom policy to properly process the data. Actual observation shape: (10, 5)",
    ]
]


@pytest.mark.parametrize("env_id", BSUITE_NAME_TO_LOADERS)
def test_check_env(env_id):
    """Check that environment pass the gymnasium check_env."""
    env = bsuite.load(env_id, BSUITE_ENV_SETTINGS[env_id])
    env = BSuiteCompatibilityV0(env)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for warning_message in caught_warnings:
        assert isinstance(warning_message.message, Warning)
        if warning_message.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {warning_message.message}")

    env.close()


@pytest.mark.parametrize("env_id", BSUITE_NAME_TO_LOADERS)
def test_seeding(env_id):
    """Test that dm-control seeding works."""

    # bandit and deep_sea and SOMETIMES discounting_chain fail this test
    if env_id in ["bandit", "deep_sea", "discounting_chain"]:
        return

    env_1 = bsuite.load(env_id, BSUITE_ENV_SETTINGS[env_id])
    env_1 = BSuiteCompatibilityV0(env_1)
    env_2 = bsuite.load(env_id, BSUITE_ENV_SETTINGS[env_id])
    env_2 = BSuiteCompatibilityV0(env_2)

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
