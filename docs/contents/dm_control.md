## DM Control (both single and multiagent environments)

### Installation
```
pip install shimmy[dm-control]
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
```

```{eval-rst}
.. autoclass:: shimmy.dm_control_multiagent_compatibility.DmControlMultiAgentCompatibilityV0
    :members:
    :undoc-members:
    :show-inheritance:
```

### Usage (Single agent)
```python
import gymnasium as gym

env = gym.make("dm_control/acrobot_swingup_sparse-v0")
```

```{eval-rst}
.. autoclass:: shimmy.dm_control_compatibility.DmControlCompatibilityV0
    :members:
    :undoc-members:
    :show-inheritance:
```