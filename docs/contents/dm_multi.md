## dm-control soccer (multi-agent)

### [DeepMind Control: Soccer](https://shimmy.farama.org/contents/dm_multi/)
Multi-agent robotics environment where teams of agents compete in soccer. It extends the single-agent [DM Control Locomotion](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/README.md) library, powered by the [MuJoCo](https://github.com/deepmind/mujoco#) physics engine.

```{figure} https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/soccer.png?raw=true
    :name: soccer
```

### Installation
```
pip install shimmy[dm-control-multi-agent]
```

### Usage (Multi agent)
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