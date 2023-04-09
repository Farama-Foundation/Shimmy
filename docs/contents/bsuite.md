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
```
pip install shimmy[bsuite]
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

## Class Description

```{eval-rst}
.. autoclass:: shimmy.bsuite_compatibility.BSuiteCompatibilityV0
    :members:
    :undoc-members:
```
