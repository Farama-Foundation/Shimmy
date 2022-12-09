# Shimmy

An API conversion tool for popular external reinforcement learning environments to [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) APIs.

upported APIs for Gymnasium
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
