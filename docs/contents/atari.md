# Atari Environments

## [Arcade Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment)

[ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) is a collection of 50+ Atari 2600 games powered by the [Stella](https://stella-emu.github.io/) emulator.

Shimmy provides compatibility wrappers to convert all [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) environments to [Gymnasium](https://gymnasium.farama.org/).

```{figure} /_static/img/ALE.png
    :name: ALE
    :alt: Arcade Learning Environment
    :width: 100%
```

Note: [PettingZoo](https://pettingzoo.farama.org/) also provides 20+ multi-agent Atari environments: [PettingZoo Atari](https://pettingzoo.farama.org/environments/atari/)

```{figure} https://pettingzoo.farama.org/_images/atari_double_dunk.gif
    :name: PettingZoo Atari
    :alt: PettingZoo Atari
    :width: 40%
```

## Installation
```
pip install shimmy[atari]
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

## Class Description
```{eval-rst}
.. autoclass:: shimmy.atari_env.AtariEnv
    :members:
    :undoc-members:
```
