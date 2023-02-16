## OpenAI Gym

### Installation
```
pip install shimmy[gym]
```

### Usage
```python
import gymnasium as gym

env = gym.make("GymV22CompatibilityV0", env_name="...")
```

### Class Description
```{eval-rst}
.. autoclass:: shimmy.openai_gym_compatibility.GymV26CompatibilityV0
    :members:
    :undoc-members:
.. autoclass:: shimmy.openai_gym_compatibility.LegacyV22Env
    :members:
    :undoc-members:
.. autoclass:: shimmy.openai_gym_compatibility.GymV22CompatibilityV0
    :members:
    :undoc-members:
.. autofunction:: shimmy.openai_gym_compatibility._strip_default_wrappers
.. autofunction:: shimmy.openai_gym_compatibility._convert_space
```
