# Atari Environments

## [Arcade Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment)

[ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) is a collection of 50+ Atari 2600 games powered by the [Stella](https://stella-emu.github.io/) emulator.

Shimmy provides compatibility wrappers to convert all [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) environments to [Gymnasium](https://gymnasium.farama.org/).


```{figure} /_static/img/ALE.png
    :name: ALE
    :alt: Arcade Learning Environment
    :width: 100%
```
For reference information and a complete list of environments, see [Gymnasium Atari](https://gymnasium.farama.org/environments/atari/).

Note: [PettingZoo](https://pettingzoo.farama.org/) also provides 20+ multi-agent Atari environments: [PettingZoo Atari](https://pettingzoo.farama.org/environments/atari/)


## Installation
To install `shimmy` and required dependencies:

```
pip install shimmy[atari]
```

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/atari.Dockerfile) for reproducibility and cross-platform compatibility:

```
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/atari.Dockerfile | docker build -t atari -f - . && docker run -it atari
```

## Usage
Load an `ALE` environment:
```python
import gymnasium as gym

env = gym.make("ALE/Pong-v5", render_mode="human")
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

To get a list of all available `atari` environments (208 total):
```python
from gymnasium.envs.registration import registry
ATARI_ENV_IDS = [
    env_id
    for env_id in registry
    if env_id.startswith("ALE") and env_id != "atari/compatibility-env-v0"
]
print(ATARI_ENV_IDS)
```

## Class Description
```{eval-rst}
.. autoclass:: shimmy.atari_env.AtariEnv
    :members:
    :undoc-members:
```
