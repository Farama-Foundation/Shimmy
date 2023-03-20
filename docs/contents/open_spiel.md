## OpenSpiel (multi-agent)


### [DeepMind OpenSpiel](https://github.com/deepmind/open_spiel)

Collection of environments and algorithms for research in general reinforcement learning and search/planning in games. 

Supports:
* n-player (single- and multi- agent) zero-sum, cooperative and general-sum games.
* one-shot and sequential, strictly turn-taking and simultaneous-move games.
* perfect and imperfect information games.
* traditional multiagent environments such as (partially- and fully- observable) grid worlds and social dilemmas.

```{figure} https://github.com/deepmind/open_spiel/blob/master/docs/_static/OpenSpielB.png?raw=true
    :name: open spiel
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
