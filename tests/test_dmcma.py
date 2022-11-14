from pettingzoo.test import parallel_api_test
from dm_control.locomotion import soccer as dm_soccer
from shimmy import DMCMACompatibilityV0

def test_check_env():
    """Check that environment pass the pettingzoo check_env."""

    env = dm_soccer.load(team_size=2,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

    env = DMCMACompatibilityV0(env, render_mode=None)

    parallel_api_test(env)

    env.close()
