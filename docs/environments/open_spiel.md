---
layout: "contents"
title: OpenSpiel
firstpage:
---

# OpenSpiel

## [DeepMind OpenSpiel](https://github.com/deepmind/open_spiel)

[OpenSpiel](https://github.com/deepmind/open_spiel) is a collection of 70+ environments for common board games, card games, as well as simple grid worlds and social dilemmas.

It supports n-player (single- and multi- agent) zero-sum, cooperative and general-sum, one-shot and sequential, strictly turn-taking and simultaneous-move, perfect and imperfect information games.

Shimmy provides compatibility wrappers to convert all [OpenSpiel](https://github.com/deepmind/open_spiel) environments to [PettingZoo](https://pettingzoo.farama.org/).


```{figure} /_static/img/openspiel.png
    :name: Open Spiel
    :alt: Open Spiel
    :width: 80%

```

Note: [PettingZoo](https://pettingzoo.farama.org/) also provides popular board & card game environments: [PettingZoo Classic](https://pettingzoo.farama.org/environments/classic/).

## Installation
To install `shimmy` and required dependencies:

```
pip install shimmy[openspiel]
```

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/openspiel.Dockerfile) for reproducibility and cross-platform compatibility:

```
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/openspiel.Dockerfile | docker build -t openspiel -f - . && docker run -it openspiel
```

```{eval-rst}
.. warning::

    OpenSpiel does not currently support Windows operating systems.
```

## Usage

Load an `openspiel` environment:
```python
from shimmy import OpenSpielCompatibilityV0

env = OpenSpielCompatibilityV0(game_name="backgammon", render_mode="human")
```

Wrap an existing `openspiel` environment:
```python
import pyspiel
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

env = pyspiel.load_game("2048")
env = OpenSpielCompatibilityV0(env)
```

Note: using the `env` and `game_name` arguments together will result in a `ValueError`.

* Use `env` to wrap an existing OpenSpiel environment.
* Use `game_name` to load a new OpenSpiel environment.

Run the environment:
```python
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(info["action_mask"])  # this is where you would insert your policy
    env.step(action)
    env.render()
env.close()
```
```{eval-rst}
.. warning::

    OpenSpiel does not support random seeding for all environments.
```

```{eval-rst}
.. warning::

    OpenSpiel does not support graphical rendering, and only supports ASCII text rendering for some environments. Environments which do not provide rendering will print the internal game state.
```

[//]: # ()
[//]: # (### Rendering)

[//]: # (OpenSpiel does not support graphical rendering, and only supports ASCII text rendering for some environments. Calling `env.render&#40;&#41;` for environments which do not provide rendering will print the internal game state.)

[//]: # ()
[//]: # (* ASCII text visualization &#40;Backgammon&#41;:)

[//]: # ()
[//]: # (```)

[//]: # (+------|------+)

[//]: # (|o...x.|x....o|)

[//]: # (|o...x.|x....o|)

[//]: # (|o...x.|x.....|)

[//]: # (|o.....|x.....|)

[//]: # (|o.....|x.....|)

[//]: # (|      |      |)

[//]: # (|x.....|o.....|)

[//]: # (|x.....|o.....|)

[//]: # (|x...o.|o.....|)

[//]: # (|x...o.|o....x|)

[//]: # (|x...o.|o....x|)

[//]: # (+------|------+)

[//]: # (Turn: o)

[//]: # (Dice: 16)

[//]: # (Bar:)

[//]: # (Scores, X: 0, O: 0)

[//]: # (```)

[//]: # ()
[//]: # (* Internal state representation &#40;Chess&#41;:)

[//]: # (```)

[//]: # (* rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1)

[//]: # (```)

## Class Description
```{eval-rst}
.. autoclass:: shimmy.openspiel_compatibility.OpenSpielCompatibilityV0
    :members:
    :undoc-members:
```
