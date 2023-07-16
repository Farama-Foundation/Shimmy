---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/shimmy-text.png
:alt: Shimmy Logo
```

```{project-heading}
An API conversion tool for reinforcement learning environments.
```

**Shimmy provides [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) bindings for popular external RL environments.**


```{eval-rst}
+------------------------------------------------+---------------------------------------------+------------------------------------------------+
| .. figure:: /_static/img/dm_control.gif        | .. figure:: /_static/img/dm_soccer.gif      | .. figure::  /_static/img/dm_lab_single.gif    |
|   :alt: DM Control                             |   :alt: DM Soccer                           |   :alt: DM Lab                                 |
|   :height: 180px                               |   :height: 180px                            |   :height: 180px                               |
|   :target: environments/dm_control             |   :target: environments/dm_multi            |   :target: environments/dm_lab                 |
|                                                |                                             |                                                |
|   **DM Control**: 3D physics-based             |   **DM Control Soccer**: Multi-agent        |   **DM Lab**: 3D navigation and                |
|   robotics simulation.                         |   cooperative soccer game.                  |   puzzle-solving.                              |
+------------------------------------------------+---------------------------------------------+------------------------------------------------+
+------------------------------------------------+---------------------------------------------+------------------------------------------------+
| .. figure:: /_static/img/bsuite.png            | .. figure:: /_static/img/ALE.png            | .. figure:: /_static/img/meltingpot.gif        |
|    :alt: Behavior Suite                        |   :alt: Atari Learning Environment          |   :alt: Melting Pot                            |
|    :height: 180px                              |   :height: 180px                            |   :height: 180px                               |
|    :target: environments/bsuite                |   :target: environments/atari               |   :target: environments/meltingpot             |
|                                                |                                             |                                                |
|    **Behavior Suite**: Test suite for          |   **Atari Learning Environment**:           |   **Melting Pot**: Multi-agent social          |
|    evaluating model behavior.                  |   Set of 50+ classic Atari 2600 games.      |   reasoning games.                             |
+------------------------------------------------+---------------------------------------------+------------------------------------------------+
+------------------------------------------------+---------------------------------------------+------------------------------------------------+
| .. figure:: /_static/img/openai_gym.png        | .. figure:: /_static/img/openspiel.png      |                                                |
|    :alt: OpenAI Gym                            |   :alt: OpenSpiel                           |                                                |
|    :height: 180px                              |   :height: 180px                            |                                                |
|    :target: environments/gym                   |   :target: environments/open_spiel          |                                                |
|                                                |                                             |                                                |
|    **OpenAI Gym**: Compatibility support for   |   **OpenSpiel**: Collection of 70+ board    |                                                |
|    Gym V21-V26.                                |   & card game environments.                 |                                                |
+------------------------------------------------+---------------------------------------------+------------------------------------------------+
```

<style>
@media (min-width: 550px) {
    figure img{
        max-height: 180px;
    }
}
@media (max-width: 550px) {
    figure img{
        max-height: 100px;
    }
}
</style>


Environments can be interacted with using a simple, high-level API:

```python
import gymnasium as gym
env = gym.make("dm_control/acrobot-swingup_sparse-v0", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
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
environments/dm_control
environments/dm_lab
environments/bsuite
environments/gym
environments/atari
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
release_notes
```
