---
layout: "contents"
title: DM Control Soccer
firstpage:
---

# DM Control (multi-agent)

## [DeepMind Control: Soccer](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md)

[DM Control Soccer](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md) is a multi-agent robotics environment where teams of agents compete in soccer. It extends the single-agent [DM Control Locomotion](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/README.md) library, powered by the [MuJoCo](https://github.com/deepmind/mujoco#) physics engine.

Shimmy provides compatibility wrappers to convert all [DM Control Soccer](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md) environments to [PettingZoo](https://pettingzoo.farama.org/).

```{figure} /_static/img/dm_soccer.png
    :name: DM soccer
    :alt: DeepMind Soccer
    :width: 80%
```

## Installation

To install `shimmy` and required dependencies:

```
pip install shimmy[dm-control-multi-agent]
```

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/dm_control_multiagent.Dockerfile) for reproducibility and cross-platform compatibility:

```
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/dm_control_multiagent.Dockerfile | docker build -t dm_control_multiagent -f - . && docker run -it dm_control_multiagent
```

## Usage

[//]: # (env, team_size, time_limit, disable_walker_contracts, enable_field_box, terminate_on_boal, walker_type, render_mode)
[//]: # (`DmControlMultiAgentCompatibilityV0&#40;&#41;`)

Load a new `dm_control.locomotion.soccer` environment:
```python
from shimmy import DmControlMultiAgentCompatibilityV0

env = DmControlMultiAgentCompatibilityV0(team_size=5, render_mode="human")
```

Wrap an existing `dm_control.locomotion.soccer` environment:

```python
from dm_control.locomotion import soccer as dm_soccer
from shimmy import DmControlMultiAgentCompatibilityV0

env = dm_soccer.load(team_size=2)
env = DmControlMultiAgentCompatibilityV0(env)
```
Note: Using the `env` argument any argument other than `render_mode` will result in a `ValueError`:

* Use the `env` argument to wrap an existing environment.
* Use the `team_size`, `time_limit`, `disable_walker_contacts`, `enable_field_box`, `terminate_on_goal`, and `walker_type` arguments to load a new environment.


Run the environment:
```python
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

Environments are loaded as [`ParallelEnv`](https://pettingzoo.farama.org/api/parallel/), but can be converted to [`AECEnv`](https://pettingzoo.farama.org/api/aec/) using [PettingZoo Wrappers](https://pettingzoo.farama.org/api/pz_wrappers/).


## Class Description

```{eval-rst}
.. autoclass:: shimmy.dm_control_multiagent_compatibility.DmControlMultiAgentCompatibilityV0
    :members:
    :undoc-members:
```

## Utils
```{eval-rst}
.. automodule:: shimmy.utils.dm_control_multiagent
   :members:
```
