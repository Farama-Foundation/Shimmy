"""Registers environments within gymnasium for optional modules."""
from functools import partial

import gymnasium
from dm_control import suite

from shimmy.dm_env_wrapper import DMEnvWrapper


def _register_dm_control_envs():
    try:
        import dm_control
    except ImportError:
        return

    def _make_dm_control_env(
        domain_name: str,
        task_name: str,
        task_kwargs=None,
        environment_kwargs=None,
        visualize_reward=None,
        **kwargs,
    ):
        env = dm_control.suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
        return DMEnvWrapper(env, **kwargs)

    for _domain_name, _task_name in suite.ALL_TASKS:
        gymnasium.register(
            f"dm_control/{_domain_name}-{_task_name}-v0",
            partial(
                _make_dm_control_env, domain_name=_domain_name, task_name=_task_name
            ),
        )


def register_gymnasium_environments():
    """This function is called when gymnasium is imported."""
    _register_dm_control_envs()
