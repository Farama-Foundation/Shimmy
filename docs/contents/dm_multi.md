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
from dm_control.locomotion import soccer as dm_soccer
from shimmy.dm_control_multiagent_compatibility import DmControlMultiAgentCompatibilityV0)

env = dm_soccer.load(team_size=2)
env = DmControlMultiAgentCompatibilityV0(env)
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