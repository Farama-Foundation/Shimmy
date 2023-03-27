## OpenAI Gym

### [OpenAI Gym](https://github.com/openai/gym)

[OpenAI Gym](https://github.com/openai/gym) is a widely-used standard API for developing reinforcement learning environments and algorithms. OpenAI stopped maintaining Gym in late 2020, leading to the [Farama Foundation](https://farama.org/)'s creation of [Gymnasium](https://gymnasium.farama.org/) a maintained fork and drop-in replacement for Gym (see [blog post](https://farama.org/Announcing-The-Farama-Foundation)).

Shimmy provides compatibility wrappers to convert Gym [V26](https://github.com/openai/gym/releases/tag/0.26.0) and [V21](https://github.com/openai/gym/releases/tag/v0.21.0) environments to [Gymnasium](https://gymnasium.farama.org/).

```{figure} /_static/img/openai_gym.png
    :name: OpenAI Gym
    :alt: OpenAI Gym
    :width: 100%
```

### Installation
```
pip install shimmy[gym]
```

### Usage

Load a `gym` environment:
```python
import gymnasium as gym

env = gym.make("GymV21CompatibilityV0", env_name="...")
```

[//]: # (Run the environment:)
[//]: # (```python)

[//]: # ()
[//]: # (```)
[//]: # (TODO: make this a full example)

### Class Description
```{eval-rst}
.. autoclass:: shimmy.openai_gym_compatibility.GymV26CompatibilityV0
    :members:
    :undoc-members:
.. autoclass:: shimmy.openai_gym_compatibility.LegacyV21Env
    :members:
    :undoc-members:
.. autoclass:: shimmy.openai_gym_compatibility.GymV21CompatibilityV0
    :members:
    :undoc-members:
.. autofunction:: shimmy.openai_gym_compatibility._strip_default_wrappers
.. autofunction:: shimmy.openai_gym_compatibility._convert_space
```
