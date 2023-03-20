## Melting Pot (multi-agent)

### [DeepMind Melting Pot](https://github.com/deepmind/meltingpot) 
Melting Pot is a suite of test scenarios for multi-agent reinforcement learning.
* Assesses generalization to novel social situations:
  * familiar and unfamiliar individuals
  * social interactions: cooperation, competition, deception, reciprocation, trust, stubbornness
* 50+ substrates and 250+ test scenarios

```{figure} https://github.com/deepmind/meltingpot/raw/main/docs/images/meltingpot_montage.gif
    :name: melting pot
```

### Installation

```
pip install shimmy[melting-pot]
```

Melting Pot must be installed manually, see [installation](https://github.com/deepmind/meltingpot#installation).

### Usage (Multi agent)
```python
from shimmy import MeltingPotCompatibilityV0
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena", render_mode="human")

observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.step(actions)
env.close()
```

### Class Description

```{eval-rst}
.. autoclass:: shimmy.melting_pot_compatibility.MeltingPotCompatibilityV0
    :members:
    :undoc-members:
```

