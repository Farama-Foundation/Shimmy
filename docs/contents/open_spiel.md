## OpenSpiel (multi-agent)


### [DeepMind OpenSpiel](https://github.com/deepmind/open_spiel)

OpenSpiel is a collection of 70+ environments for common board games, card games, as well as simple grid worlds and social dilemmas.

It supports n-player (single- and multi- agent) zero-sum, cooperative and general-sum, one-shot and sequential, strictly turn-taking and simultaneous-move, perfect and imperfect information games.

```{figure} /_static/img/openspiel.png
    :name: Open Spiel
    :alt: Open Spiel
    :width: 100%

```


### Installation
```
pip install shimmy[openspiel]
```

### Usage
```python
import pyspiel
from shimmy.openspiel_compatibility import OpenspielCompatibilityV0

env = pyspiel.load_game("2048")
env = OpenspielCompatibilityV0(game=env, render_mode=None)
```

### Class Description
```{eval-rst}
.. autoclass:: shimmy.openspiel_compatibility.OpenspielCompatibilityV0
    :members:
    :undoc-members:
```
