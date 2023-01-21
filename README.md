# Shimmy

An API conversion tool for popular external reinforcement learning environments to [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) APIs.

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

#### Usage
```python
import gymnasium as gym

env = gym.make("GymV22CompatibilityV0", env_name="...")
```

### Atari Environments

#### Installation
```
pip install shimmy[atari]
```

#### Usage
```python
import gymnasium as gym

env = gym.make("ALE/Pong-v5")
```

### DM Control

#### Installation
```
pip install shimmy[dm-control]
```

#### Usage (Multi agent)
```python
from dm_control.locomotion import soccer as dm_soccer
from shimmy.dm_control_multiagent_compatibility import (
    DmControlMultiAgentCompatibilityV0,
)

walker_type = dm_soccer.WalkerType.BOXHEAD,

env = dm_soccer.load(
    team_size=2,
    time_limit=10.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False,
    walker_type=walker_type,
)

env = DmControlMultiAgentCompatibilityV0(env)
```

#### Usage (Single agent)
```python
import gymnasium as gym

env = gym.make("dm_control/acrobot_swingup_sparse-v0")
```

### DM Lab

#### Installation

Courtesy to [Danijar Hafner](https://github.com/deepmind/lab/issues/242) for providing this install script.
```bash
#!/bin/sh
set -eu

# Dependencies
apt-get update && apt-get install -y \
    build-essential curl freeglut3 gettext git libffi-dev libglu1-mesa \
    libglu1-mesa-dev libjpeg-dev liblua5.1-0-dev libosmesa6-dev \
    libsdl2-dev lua5.1 pkg-config python-setuptools python3-dev \
    software-properties-common unzip zip zlib1g-dev g++
pip3 install numpy

# Bazel
apt-get install -y apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
apt-get update && apt-get install -y bazel

# Build
git clone https://github.com/deepmind/lab.git
cd lab
echo 'build --cxxopt=-std=c++17' > .bazelrc
bazel build -c opt //python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip3 install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl
cd ..
rm -rf lab
```

#### Usage
```python
import deepmind_lab

from shimmy.dm_lab_compatibility import DmLabCompatibilityV0

observations = ["RGBD"]
config = {"width": "640", "height": "480", "botCount": "2"}
renderer = "hardware"

env = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
env = DmLabCompatibilityV0(env)
```

### For Developers and Testing Only
```
pip install shimmy[testing]
```

### To just install everything
```
pip install shimmy[all, testing]
```

### OpenSpiel

#### Installation
```
pip install shimmy[pettingzoo]
```

#### Usage
```python
import pyspiel
from shimmy.openspiel_compatibility import OpenspielCompatibilityV0

env = pyspiel.load_game("2048")
env = OpenspielCompatibilityV0(game=env, render_mode=None)
```

## Citation

If you use this in your research, please cite:
```
@software{shimmy2022github,
  author = {Jordan Terry, Mark Towers, Jun Jet Tai},
  title = {Shimmy: Gymnasium and Pettingzoo Wrappers for Commonly Used Environments},
  url = {http://github.com/Farama-Foundation/Shimmy},
  version = {0.2.0},
  year = {2022},
}```
