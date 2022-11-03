import deepmind_lab
from shimmy import DmLabCompatibilityV0

observations = ['RGBD']
env = deepmind_lab.Lab('lt_chasm', observations,
                       config={'width': '640',    # screen size, in pixels
                               'height': '480',   # screen size, in pixels
                               'botCount': '2'},  # lt_chasm option.
                       renderer='hardware')       # select renderer.

env.reset()
env = DmLabCompatibilityV0(env=env, render_mode=None)

# reset and begin test
env.reset()
term, trunc = False, False

# run until termination
while not term and not trunc:
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
