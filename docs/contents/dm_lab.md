# DM Lab

## [DeepMind Lab](https://github.com/deepmind/lab)

[DM Lab](https://github.com/deepmind/lab) is a suite of challenging 3D navigation and puzzle-solving tasks for learning agents, based on id Software's
[Quake III Arena](https://github.com/id-Software/Quake-III-Arena) via
[ioquake3](https://github.com/ioquake/ioq3) and
[other open source software](#upstream-sources).

Shimmy provides compatibility wrappers to convert all DM Lab environments to [Gymnasium](https://gymnasium.farama.org/).

```{figure} /_static/img/dm_lab.gif
    :name: DM lab
    :alt: DeepMind Lab
    :width: 80%
```


## Installation

```
pip install shimmy[dm-lab]
```

DeepMind Lab is not distributed via [pypi](https://pypi.org/) and must be installed manually. Courtesy to  [Danijar Hafner](https://github.com/deepmind/lab/issues/242) for providing an [install script](https://github.com/Farama-Foundation/Shimmy/blob/main/scripts/install_dm_lab.sh). For troubleshooting, refer to the official [installation instructions](https://github.com/deepmind/lab#getting-started-on-linux).

```{eval-rst}
.. warning::

    DM Lab is not currently supported on Windows or macOS.
```

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/dm_lab.Dockerfile) for reproducibility and cross-platform compatibility. 



## Usage
Load a `deepmind_lab` environment:
```python
import deepmind_lab
from shimmy.dm_lab_compatibility import DmLabCompatibilityV0

observations = ["RGBD"]
config = {"width": "640", "height": "480", "botCount": "2"}
renderer = "hardware"

env = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
env = DmLabCompatibilityV0(env)
```

Run the environment:
```python
observation, info = env.reset()
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

```{eval-rst}
.. warning::

    Using `gym.make()` to load DM Lab environments is not currently supported.     
```

We provide a [helper function](#shimmy.utils.dm_lab.load_dm_lab) to load DM Lab environments, but it is not guaranteed to work for all configurations.



## Class Description
```{eval-rst}
.. autoclass:: shimmy.dm_lab_compatibility.DmLabCompatibilityV0
    :members:
    :undoc-members:
```

## Utils
```{eval-rst}
.. automodule:: shimmy.utils.dm_lab
   :members:
```
