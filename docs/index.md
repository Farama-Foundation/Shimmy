---
hide-toc: true
firstpage:
lastpage:
---


# Shimmy is an API conversion tool for popular external reinforcement learning environments to [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) APIs.

```{eval-rst}
+------------------------------------------------+-------------------------------------------+------------------------------------------------+
| .. figure:: /_static/img/dm_control.gif        | .. figure:: /_static/img/dm_soccer.gif    | .. figure::  /_static/img/dm_lab_single.gif    |
|   :alt: map to buried treasure                 |   :alt: map to buried treasure            |   :alt: map to buried treasure                 |
|   :height: 200px                               |   :height: 200px                          |   :height: 200px                               |
|                                                |                                           |                                                |
|   **DM Control**: 3D physics-based             |   **DM Control Soccer**: Multi-agent      |   **DM Lab**: 3D navigation and a              |
|   robotics simulation.                         |   cooperative soccer game.                |   puzzle-solving.                              |
+------------------------------------------------+-------------------------------------------+------------------------------------------------+
+------------------------------------------------+-------------------------------------------+------------------------------------------------+
| .. figure:: /_static/img/bsuite.png            | .. figure:: /_static/img/ALE.png          | .. figure:: /_static/img/meltingpot.gif        | 
|    :alt: map to buried treasure                |   :alt: map to buried treasure            |   :alt: map to buried treasure                 |
|    :height: 200px                              |   :height: 200px                          |   :height: 200px                               |
|                                                |                                           |                                                |
|    **Behavior Suite**: Test suite for          |   **Atari Learning Environment**:         |   **Melting Pot**: Multi-agent social          |  
|    evaluating model behavior.                  |   Set of 50+ classic Atari 2600 games.    |   reasoning benchmark.                         |
+------------------------------------------------+-------------------------------------------+------------------------------------------------+
+------------------------------------------------+-------------------------------------------+------------------------------------------------+
| .. figure:: /_static/img/openai_gym.png        | .. figure:: /_static/img/openspiel.png    |                                                | 
|    :alt: map to buried treasure                |   :alt: map to buried treasure            |                                                |
|    :height: 200px                              |   :height: 200px                          |                                                |
|                                                |                                           |                                                |
|    **OpenAI Gym**: Compatibility support for   |   **OpenSpiel**: Collection of 70+ board  |                                                |  
|    Gym V21 & V26.                              |   & card game environments.               |                                                |
+------------------------------------------------+-------------------------------------------+------------------------------------------------+
```

## Supported APIs

### Single-agent
[Gymnasium](https://gymnasium.farama.org/) compatibility wrappers:
- [OpenAI Gym](https://shimmy.farama.org/contents/gym/)
- [Atari Learning Environments](https://shimmy.farama.org/contents/atari/)
- [DeepMind Lab](https://shimmy.farama.org/contents/dm_lab/)
- [Behavior Suite](https://shimmy.farama.org/contents/bsuite/)
- [DM Control](https://shimmy.farama.org/contents/dm_control/)

### Multi-agent
[PettingZoo](https://pettingzoo.farama.org/) compatibility wrappers:
- [DM Control Soccer](https://shimmy.farama.org/contents/dm_multi/)
- [OpenSpiel](https://shimmy.farama.org/contents/open_spiel/)
- [Melting Pot](https://shimmy.farama.org/contents/meltingpot/)


## Citation

If you use this in your research, please cite:
```
@software{shimmy2022github,
  author = {{Jun Jet Tai, Mark Towers, Elliot Tower} and Jordan Terry},
  title = {Shimmy: Gymnasium and PettingZoo Wrappers for Commonly Used Environments},
  url = {http://github.com/Farama-Foundation/Shimmy},
  version = {0.2.0},
  year = {2022},
}
```

```{toctree}
:hidden:
:caption: Introduction

content/getting_started
content/basic_usage
```

```{toctree}
:hidden:
:caption: Environments
environments/gym
environments/atari
environments/bsuite
environments/dm_lab
environments/dm_control
```

```{toctree}
:hidden:
:caption: Multi-Agent Environments
environments/dm_multi
environments/open_spiel
environments/meltingpot
```

```{toctree}
:hidden:
:caption: Development
Github <https://github.com/Farama-Foundation/shimmy>
```

