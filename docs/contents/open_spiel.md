## OpenSpiel (multi-agent)


### [DeepMind OpenSpiel](https://github.com/deepmind/open_spiel)

[OpenSpiel](https://github.com/deepmind/open_spiel) is a collection of 70+ environments for common board games, card games, as well as simple grid worlds and social dilemmas.

It supports n-player (single- and multi- agent) zero-sum, cooperative and general-sum, one-shot and sequential, strictly turn-taking and simultaneous-move, perfect and imperfect information games.

Shimmy provides compatibility wrappers to convert all [OpenSpiel](https://github.com/deepmind/open_spiel) environments to [PettingZoo](https://pettingzoo.farama.org/).


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

Load an `openspiel` environment:
```python
import pyspiel
from shimmy.openspiel_compatibility import OpenSpielCompatibilityV0

env = pyspiel.load_game("2048")
env = OpenSpielCompatibilityV0(env)
```

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

Note: OpenSpiel does not provide rendering for all environments. 

Calling `env.render()` will return different results depending on the environment. 

* ASCII text visualization (Backgammon):

```
+------|------+
|o...x.|x....o|
|o...x.|x....o|
|o...x.|x.....|
|o.....|x.....|
|o.....|x.....|
|      |      |
|x.....|o.....|
|x.....|o.....|
|x...o.|o.....|
|x...o.|o....x|
|x...o.|o....x|
+------|------+
Turn: o
Dice: 16
Bar:
Scores, X: 0, O: 0
```

* Internal state representation (Chess):
```
* rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

### Class Description
```{eval-rst}
.. autoclass:: shimmy.openspiel_compatibility.OpenSpielCompatibilityV0
    :members:
    :undoc-members:
```
