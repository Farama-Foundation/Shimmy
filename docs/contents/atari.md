## Atari Environments

### [Arcade Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment)
A collection of 50+ Atari 2600 games powered by the [Stella](https://stella-emu.github.io/) emulator.

```{figure} /_static/img/ALE.png
    :name: ALE
    :alt: Arcade Learning Environment
    :width: 60%
```

### Installation
```
pip install shimmy[atari]
```

### Usage
```python
import gymnasium as gym

env = gym.make("ALE/Pong-v5")
```

### Class Description
```{eval-rst}
.. autoclass:: shimmy.atari_env.AtariEnv
    :members:
    :undoc-members:
```
