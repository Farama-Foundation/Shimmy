# DM Control

## [DeepMind Control](https://github.com/deepmind/dm_control/)

[DM Control](https://github.com/deepmind/dm_control/) is a framework for physics-based simulation and reinforcement learning environments using the [MuJoCo](https://github.com/deepmind/mujoco#) physics engine.

Shimmy provides compatibility wrappers to convert [Control Suite](https://github.com/deepmind/dm_control/blob/main/dm_control/suite/README.md) environments and custom [Locomotion](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/README.md) environments to [Gymnasium](https://gymnasium.farama.org/).

```{figure} /_static/img/dm_locomotion.png
    :name: DM Locomotion
    :alt: DeepMind Locomotion
    :width: 60%
```

## Installation
To install `shimmy` and required dependencies:

```
pip install shimmy[dm-control]
```

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/dm_control.Dockerfile) for reproducibility and cross-platform compatibility:

```
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/dm_control.Dockerfile | docker build -t dm_control -f - . && docker run -it dm_control
```


## Usage
Load a `dm_control` environment:
```python
import gymnasium as gym

env = gym.make("dm_control/acrobot-swingup_sparse-v0", render_mode="human")
```

Run the environment:
```python
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

## Class Description



```{eval-rst}
.. autoclass:: shimmy.dm_control_compatibility.DmControlCompatibilityV0
    :members:
    :undoc-members:
```
