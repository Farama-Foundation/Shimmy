---
hide-toc: true
firstpage:
lastpage:
---


# Shimmy is an API conversion tool for popular external reinforcement learning environments to [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) APIs.

```{figure} _static/img/shimmy-white.svg
   :alt: Shimmy Logo
   :width: 200
```

## Supported APIs

### OpenAI Gym
- Bindings to convert OpenAI Gym environments to Gymnasium Environments.

### Atari Environments for OpenAI Gym
- Bindings to ALE-py to provide Atari environments in Gymnasium.

### [DeepMind Control](https://github.com/deepmind/dm_control)
- Gymnasium bindings for single agent environments.
- Pettingzoo bindings for multiagent soccer environments.

### [DMLab](https://github.com/deepmind/lab)
- Pettingzoo bindings for all environments.

### [OpenSpiel](https://github.com/deepmind/open_spiel)
- Pettingzoo bindings for all environments.

### Incoming Projects

The following are a list of existing environment suites that we are looking into bringing into Shimmy.
We are actively looking for developers to contribute to this project, if you are interested in helping, please reach out to us.

- [The DeepMing Env API](https://github.com/deepmind/dm_env)
- [Behaviour Suite](https://github.com/deepmind/bsuite)
- [Melting Pot](https://github.com/deepmind/meltingpot)

## Installation and Usage

To install Shimmy from PyPI:
```
pip install shimmy
```
Out of the box, Shimmy doesn't install any of the dependencies required for the environments it supports.
To install them, you'll have to install the optional extras.
All single agent environments have registration under the Gymnasium API, while all multiagent environments must be wrapped using the corresponding compatibility wrappers.

### OpenAI Gym

#### Installation
```
pip install shimmy[gym]
```

## At a glance

This is an example of using Shimmy to convert DM Control environments into a Gymnasium compatible environment:

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

## Citation

If you use this in your research, please cite:
```
TBD
```

```{toctree}
:hidden:
:caption: Environments
contents/index
```

```{toctree}
:hidden:
:caption: Development
Github <https://github.com/Farama-Foundation/shimmy>
```

