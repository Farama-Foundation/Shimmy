## Melting Pot (multi-agent)

### [DeepMind Melting Pot](https://github.com/deepmind/meltingpot) 
[Melting Pot](https://github.com/deepmind/meltingpot) is a suite of test scenarios for multi-agent reinforcement learning, using 2D game environments.

It assesses generalization to novel social situations (familiar and unfamiliar individuals),
and requires social reasoning such as cooperation, competition, deception, reciprocation, trust, and stubbornness. 

Shimmy provides compatibility wrappers to convert all [Melting Pot](https://github.com/deepmind/meltingpot) environments to [PettingZoo](https://pettingzoo.farama.org/).

```{figure} /_static/img/meltingpot.gif
    :name: melting pot
```

### Installation

```
pip install shimmy[melting-pot]
```

Melting Pot must be installed manually, see [installation](https://github.com/deepmind/meltingpot#installation).

### Usage (Multi agent)
Load a `meltingpot` environment:
```python
from shimmy import MeltingPotCompatibilityV0
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena", render_mode="human")
```

Run the environment:
```python
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.step(actions)
env.close()
```

### Class Description

```{eval-rst}
.. autoclass:: shimmy.meltingpot_compatibility.MeltingPotCompatibilityV0
    :members:
    :undoc-members:
```

