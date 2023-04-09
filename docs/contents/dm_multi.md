## DM Control (multi-agent)

### [DeepMind Control: Soccer](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md)
[DM Control Soccer](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md) is a multi-agent robotics environment where teams of agents compete in soccer. It extends the single-agent [DM Control Locomotion](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/README.md) library, powered by the [MuJoCo](https://github.com/deepmind/mujoco#) physics engine.

Shimmy provides compatibility wrappers to convert all [DM Control Soccer](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md) environments to [PettingZoo](https://pettingzoo.farama.org/).

```{figure} /_static/img/dm_soccer.png
    :name: DM soccer
    :alt: DeepMind Soccer
    :width: 80%
```

### Installation
```
pip install shimmy[dm-control-multi-agent]
```

### Usage (Multi agent)

Load a `dm_control.locomotion.soccer` environment:
```python
from shimmy.dm_control_multiagent_compatibility import DmControlMultiAgentCompatibilityV0)

env = DmControlMultiAgentCompatibilityV0(team_size=5)
```

Load an existing `dm_control.locomotion.soccer` environment:
```python
from dm_control.locomotion import soccer as dm_soccer
from shimmy.dm_control_multiagent_compatibility import DmControlMultiAgentCompatibilityV0

env = dm_soccer.load(team_size=2)
env = DmControlMultiAgentCompatibilityV0(env)
```
The first argument `env` wraps an existing environment, while specifying any of `team_size`, `time_limit`, `disable_walker_contacts`, `enable_field_box`, `terminate_on_goal` or `walker_type` loads a new environment and wraps it. 

```{eval-rst}
.. warning::

    Using the `env` argument with any of the following arguments will result in a ValueError:
     `team_size`, 
     `time_limit`, 
     `disable_walker_contacts`, 
     `enable_field_box` 
     `terminate_on_goal`, 
     `walker_type`    
```


Run the environment:
```python
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

### Class Description

```{eval-rst}
.. autoclass:: shimmy.dm_control_multiagent_compatibility.DmControlMultiAgentCompatibilityV0
    :members:
    :undoc-members:
```

### Utils
```{eval-rst}
.. automodule:: shimmy.utils.dm_control_multiagent
   :members:
```