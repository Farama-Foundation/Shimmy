# Melting Pot (multi-agent)

## [DeepMind Melting Pot](https://github.com/deepmind/meltingpot) 

[Melting Pot](https://github.com/deepmind/meltingpot) is a suite of test scenarios for multi-agent reinforcement learning, using 2D game environments.

It assesses generalization to novel social situations (familiar and unfamiliar individuals),
and requires social reasoning such as cooperation, competition, deception, reciprocation, trust, and stubbornness. 

Shimmy provides compatibility wrappers to convert all [Melting Pot](https://github.com/deepmind/meltingpot) environments to [PettingZoo](https://pettingzoo.farama.org/).

```{figure} /_static/img/meltingpot.gif
    :name: Melting Pot
    :width: 50%
```

## Installation

To install `shimmy` and required dependencies:

```
pip install shimmy[melting-pot]
```

Melting Pot is not distributed via [pypi](https://pypi.org/) and must be installed manually. We provide an [installation script](https://github.com/Farama-Foundation/Shimmy/blob/main/scripts/install_melting_pot.sh) (compatible with macOS and linux). For troubleshooting,  refer to the official [installation instructions](https://github.com/deepmind/meltingpot#installation).

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/meltingpot.Dockerfile) for reproducibility and cross-platform compatibility:

`curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/meltingpot.Dockerfile | docker build -t meltingpot -f - . && docker run -it meltingpot`

## Usage

Load a `meltingpot` environment:
```python
from shimmy import MeltingPotCompatibilityV0

env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena", render_mode="human")
```

Wrap an existing `meltingpot` environment:
```python
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot

env = load_meltingpot("prisoners_dilemma_in_the_matrix__arena") 
env = MeltingPotCompatibilityV0(env, render_mode="human")
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

```{eval-rst}
.. warning::

    Using the **env** and **substrate_name** arguments together will result in a **ValueError**.
    
    * Use the `env` argument to wrap an existing Melting Pot environment.
    * Use the `substrate_name` argument to load a new Melting Pot environment.
```

## Class Description

```{eval-rst}
.. autoclass:: shimmy.meltingpot_compatibility.MeltingPotCompatibilityV0
    :members:
    :undoc-members:
```


## Utils
```{eval-rst}
.. automodule:: shimmy.utils.meltingpot
   :members:
```
