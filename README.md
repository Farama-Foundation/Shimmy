# Shimmy

An API conversion tool for popular external reinforcement learning environments to [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) APIs.

Supported APIs for Gymnasium
* OpenAI Gym
* Atari Environments
* DMControl

Supported APIs for PettingZoo
* OpenSpiel
* DmControl Multiagent Soccer
* DMLab Environments

We are open to supporting more external APIs, please create an issue or ideally, a pull request implementing the new API.

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

### DM Control (both single and multiagent environments)

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

## Citation

If you use this in your research, please cite:
```
TBD
```
