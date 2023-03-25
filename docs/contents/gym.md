## OpenAI Gym

### [OpenAI Gym](https://github.com/openai/gym)

Shimmy provides Gymnasium compatibility wrappers for Gym V26 and V21.

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
```python
import gymnasium as gym

env = gym.make("GymV21CompatibilityV0", env_name="...")
```

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
