# Behavior Suite

## [DeepMind Behavior Suite](https://github.com/deepmind/bsuite)

[Behavior Suite](https://github.com/deepmind/bsuite) is a collection of carefully-designed experiments that investigate various aspects of agent behavior through shared benchmarks.

Shimmy provides compatibility wrappers to convert [Behavior Suite](https://github.com/deepmind/bsuite) environments to [Gymnasium](https://gymnasium.farama.org/).

```{figure} /_static/img/bsuite.png
    :name: Behavior Suite
    :alt: Behavior Suite Visualization
    :width: 80%
```

## Installation
To install `shimmy` and required dependencies:

```
pip install shimmy[bsuite]
```

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/bsuite.Dockerfile) for reproducibility and cross-platform compatibility:

```
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/bsuite.Dockerfile | docker build -t bsuite -f - . && docker run -it bsuite
```

## Usage
Load a `bsuite` environment:
```python
import gymnasium as gym

env = gym.make("bsuite/catch-v0", render_mode="human")
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

To get a list of all available `bsuite` environments (23 total):
```python
from gymnasium.envs.registration import registry
BSUITE_ENV_IDS = [
    env_id
    for env_id in registry
    if env_id.startswith("bsuite") and env_id != "bsuite/compatibility-env-v0"
]
print(BSUITE_ENV_IDS)
```

## Class Description

```{eval-rst}
.. autoclass:: shimmy.bsuite_compatibility.BSuiteCompatibilityV0
    :members:
    :undoc-members:
```
