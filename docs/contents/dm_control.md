## dm-control (single agent)

### [DeepMind Control: Control Suite](https://github.com/deepmind/dm_control/blob/main/dm_control/suite/README.md)

A set of Python Reinforcement Learning environments powered by the [MuJoCo](https://github.com/deepmind/mujoco#) physics engine.

```{figure} https://github.com/deepmind/dm_control/blob/main/dm_control/suite/all_domains.png?raw=true
    :name: control suite
```

### [DeepMind Control: Locomotion](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/README.md) 
Package containing reusable components for defining control tasks that are related to locomotion. 

```{figure} https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/gaps.png?raw=true
    :name: locomotion
```

### Installation
```
pip install shimmy[dm-control]
```

### Usage (Single agent)
To run a `dm_control` environment:
```python
import gymnasium as gym
env = gym.make("dm_control/acrobot-swingup_sparse-v0", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

To get a list of all available `dm_control` environments (85 total):
```python
from gymnasium.envs.registration import registry
DM_CONTROL_ENV_IDS = [
    env_id
    for env_id in registry
    if env_id.startswith("dm_control") and env_id != "dm_control/compatibility-env-v0"
]
print(DM_CONTROL_ENV_IDS)
```

### Class Description

```{eval-rst}
.. autoclass:: shimmy.dm_control_multiagent_compatibility.DmControlMultiAgentCompatibilityV0
    :members:
    :undoc-members:
```

```{eval-rst}
.. autoclass:: shimmy.dm_control_compatibility.DmControlCompatibilityV0
    :members:
    :undoc-members:
```