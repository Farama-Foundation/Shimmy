# Shimmy

An API conversion tool for popular external reinforcement learning environments to [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) APIs.

upported APIs for Gymnasium
* OpenAI Gym
* DMControl

Supported APIs for PettingZoo
* OpenSpiel
* DmControl Multiagent Soccer
* DMLab Environments

We are open to supporting more external APIs, please create an issue or ideally, a pull request implementing the new API.


## Get started

The supported `dm_control` environments are as follows:

```python
import gymnasium as gym
from shimmy.registration import DM_CONTROL_SUITE_ENVS

env_ids = [f"dm_control/{'-'.join(item)}-v0" for item in DM_CONTROL_SUITE_ENVS]
print(env_ids)

env = gym.make(env_ids[0])
env_flatten = gym.wrappers.FlattenObservation(env)
print(env_ids[0])
print("===🌎", env.observation_space)
print("===🕹️", env.action_space)
print("---flattened 🌎", env_flatten.observation_space)
print("---flattened 🕹️", env_flatten.action_space)
```
```bash
['dm_control/acrobot-swingup-v0', 'dm_control/acrobot-swingup_sparse-v0', 'dm_control/ball_in_cup-catch-v0', 'dm_control/cartpole-balance-v0', 'dm_control/cartpole-balance_sparse-v0', 'dm_control/cartpole-swingup-v0', 'dm_control/cartpole-swingup_sparse-v0', 'dm_control/cartpole-two_poles-v0', 'dm_control/cartpole-three_poles-v0', 'dm_control/cheetah-run-v0', 'dm_control/dog-stand-v0', 'dm_control/dog-walk-v0', 'dm_control/dog-trot-v0', 'dm_control/dog-run-v0', 'dm_control/dog-fetch-v0', 'dm_control/finger-spin-v0', 'dm_control/finger-turn_easy-v0', 'dm_control/finger-turn_hard-v0', 'dm_control/fish-upright-v0', 'dm_control/fish-swim-v0', 'dm_control/hopper-stand-v0', 'dm_control/hopper-hop-v0', 'dm_control/humanoid-stand-v0', 'dm_control/humanoid-walk-v0', 'dm_control/humanoid-run-v0', 'dm_control/humanoid-run_pure_state-v0', 'dm_control/humanoid_CMU-stand-v0', 'dm_control/humanoid_CMU-run-v0', 'dm_control/lqr-lqr_2_1-v0', 'dm_control/lqr-lqr_6_2-v0', 'dm_control/manipulator-bring_ball-v0', 'dm_control/manipulator-bring_peg-v0', 'dm_control/manipulator-insert_ball-v0', 'dm_control/manipulator-insert_peg-v0', 'dm_control/pendulum-swingup-v0', 'dm_control/point_mass-easy-v0', 'dm_control/point_mass-hard-v0', 'dm_control/quadruped-walk-v0', 'dm_control/quadruped-run-v0', 'dm_control/quadruped-escape-v0', 'dm_control/quadruped-fetch-v0', 'dm_control/reacher-easy-v0', 'dm_control/reacher-hard-v0', 'dm_control/stacker-stack_2-v0', 'dm_control/stacker-stack_4-v0', 'dm_control/swimmer-swimmer6-v0', 'dm_control/swimmer-swimmer15-v0', 'dm_control/walker-stand-v0', 'dm_control/walker-walk-v0', 'dm_control/walker-run-v0']
dm_control/acrobot-swingup-v0
===🌎 Dict('orientations': Box(-inf, inf, (4,), float64), 'velocity': Box(-inf, inf, (2,), float64))
===🕹️ Box(-1.0, 1.0, (1,), float64)
---flattened 🌎 Box(-inf, inf, (6,), float64)
---flattened 🕹️ Box(-1.0, 1.0, (1,), float64)
```

For most usage, we recommend applying the `gym.wrappers.FlattenObservation(env)` wrapper to reduce the `Dict` observation space to a `Box` observation space.
