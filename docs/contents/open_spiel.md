## OpenSpiel

### Installation
```
pip install shimmy[pettingzoo]
```

### Usage
```python
import pyspiel
from shimmy.openspiel_compatibility import OpenspielCompatibilityV0

env = pyspiel.load_game("2048")
env = OpenspielCompatibilityV0(game=env, render_mode=None)
```


```{eval-rst}
.. autoclass:: shimmy.openspiel_compatibility.OpenspielCompatibilityV0
    :members:
    :undoc-members:
    :show-inheritance:
```
