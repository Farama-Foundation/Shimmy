---
hide-toc: true
firstpage:
lastpage:
---


# Shimmy is an API conversion tool for popular external reinforcement learning environments to [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/) APIs.

```{eval-rst}
+------------------------------------------------+-------------------------------------------+-----------------------------------------+
| .. figure:: /_static/img/dm_locomotion.png     | .. figure:: /_static/img/dm_soccer.png    | .. figure::  /_static/img/dm_lab.jpg    |
|   :alt: map to buried treasure                 |   :alt: map to buried treasure            |   :alt: map to buried treasure          |
|   :height: 200px                               |   :height: 200px                          |   :height: 200px                        |
|                                                |                                           |                                         |
|   **DM Control**: 3D physics-based             |   **DM Control Soccer**: Multi-agent      |   **DM Lab**: 3D navigation and a       |
|   robotics simulation.                         |   cooperative soccer game.                |   puzzle-solving.                       |
+------------------------------------------------+-------------------------------------------+-----------------------------------------+
+------------------------------------------------+-------------------------------------------+-----------------------------------------+
| .. figure:: /_static/img/bsuite.png            | .. figure:: /_static/img/ALE.png          | .. figure:: /_static/img/meltingpot.gif | 
|    :alt: map to buried treasure                |   :alt: map to buried treasure            |   :alt: map to buried treasure          |
|    :height: 200px                              |   :height: 200px                          |   :height: 200px                        |
|                                                |                                           |                                         |
|    **Behavior Suite**: Test suite for          |   **Atari Learning Environment**:         |   **Melting Pot**: Multi-agent social   |  
|    evaluating model behavior.                  |   Set of 50+ classic Atari 2600 games.    |   reasoning benchmark.                  |
+------------------------------------------------+-------------------------------------------+-----------------------------------------+
+------------------------------------------------+-------------------------------------------+-----------------------------------------+
| .. figure:: /_static/img/openai_gym.png        | .. figure:: /_static/img/openspiel.png    |                                         | 
|    :alt: map to buried treasure                |   :alt: map to buried treasure            |                                         |
|    :height: 200px                              |   :height: 200px                          |                                         |
|                                                |                                           |                                         |
|    **OpenAI Gym**: Compatibility support for   |   **OpenSpiel**: Collection of 70+ board  |                                         |  
|    Gym V21 & V26.                              |   & card game environments.               |                                         |
+------------------------------------------------+-------------------------------------------+-----------------------------------------+
```

## Supported APIs

### Single-agent
- [OpenAI Gym](https://github.com/openai/gym)
- [ALE-py](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [DM Lab](https://github.com/deepmind/lab)
- [Behavior Suite](https://github.com/deepmind/bsuite)
- [DM Control](https://github.com/deepmind/dm_control/)

### Multi-agent
- [DM Control Soccer](https://github.com/deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md)
- [OpenSpiel](https://github.com/deepmind/open_spiel)
- [Melting Pot](https://github.com/deepmind/meltingpot)


## Installation
To install Shimmy from [PyPI](https://pypi.org/):
```
pip install shimmy
```
To install Shimmy and required dependencies for environments, specify them as follows:
```
pip install shimmy[bsuite, atari]
```

### For Developers and Testing
```
pip install shimmy[testing]
```

### All Environments
```
pip install shimmy[all]
```

`gym-v21`, `gym-v26`, `atari`, `bsuite`, `dm-control`, `dm-control-multi-agent`, `openspiel`, `meltingpot`

## Usage

Single-agent [Gymnasium](https://gymnasium.farama.org/) environments can be loaded via `gym.make()`:

```python
import gymnasium as gym
env = gym.make("dm_control/acrobot-swingup_sparse-v0")
```
Multi-agent [PettingZoo](https://pettingzoo.farama.org) environments can be loaded via imported Shimmy wrappers:

```python
from shimmy import MeltingPotCompatibilityV0
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena")
```


## At a glance

This is an example of using Shimmy to convert DM Control environments into a Gymnasium compatible environment:

```python
import gymnasium as gym
from shimmy.registration import DM_CONTROL_SUITE_ENVS

env_ids = [f"dm_control/{'-'.join(item)}-v0" for item in DM_CONTROL_SUITE_ENVS]
print(env_ids)

env = gym.make(env_ids[0])
env_flatten = gym.wrappers.FlattenObservation(env)
print(env_ids[0])
print("===🌎", env.observation_space)
print("===🕹️", env.action_space)
print("---flattened 🌎", env_flatten.observation_space)
print("---flattened 🕹️", env_flatten.action_space)
```
```bash
['dm_control/acrobot-swingup-v0', 'dm_control/acrobot-swingup_sparse-v0', 'dm_control/ball_in_cup-catch-v0', 'dm_control/cartpole-balance-v0', 'dm_control/cartpole-balance_sparse-v0', 'dm_control/cartpole-swingup-v0', 'dm_control/cartpole-swingup_sparse-v0', 'dm_control/cartpole-two_poles-v0', 'dm_control/cartpole-three_poles-v0', 'dm_control/cheetah-run-v0', 'dm_control/dog-stand-v0', 'dm_control/dog-walk-v0', 'dm_control/dog-trot-v0', 'dm_control/dog-run-v0', 'dm_control/dog-fetch-v0', 'dm_control/finger-spin-v0', 'dm_control/finger-turn_easy-v0', 'dm_control/finger-turn_hard-v0', 'dm_control/fish-upright-v0', 'dm_control/fish-swim-v0', 'dm_control/hopper-stand-v0', 'dm_control/hopper-hop-v0', 'dm_control/humanoid-stand-v0', 'dm_control/humanoid-walk-v0', 'dm_control/humanoid-run-v0', 'dm_control/humanoid-run_pure_state-v0', 'dm_control/humanoid_CMU-stand-v0', 'dm_control/humanoid_CMU-run-v0', 'dm_control/lqr-lqr_2_1-v0', 'dm_control/lqr-lqr_6_2-v0', 'dm_control/manipulator-bring_ball-v0', 'dm_control/manipulator-bring_peg-v0', 'dm_control/manipulator-insert_ball-v0', 'dm_control/manipulator-insert_peg-v0', 'dm_control/pendulum-swingup-v0', 'dm_control/point_mass-easy-v0', 'dm_control/point_mass-hard-v0', 'dm_control/quadruped-walk-v0', 'dm_control/quadruped-run-v0', 'dm_control/quadruped-escape-v0', 'dm_control/quadruped-fetch-v0', 'dm_control/reacher-easy-v0', 'dm_control/reacher-hard-v0', 'dm_control/stacker-stack_2-v0', 'dm_control/stacker-stack_4-v0', 'dm_control/swimmer-swimmer6-v0', 'dm_control/swimmer-swimmer15-v0', 'dm_control/walker-stand-v0', 'dm_control/walker-walk-v0', 'dm_control/walker-run-v0']
dm_control/acrobot-swingup-v0
===🌎 Dict('orientations': Box(-inf, inf, (4,), float64), 'velocity': Box(-inf, inf, (2,), float64))
===🕹️ Box(-1.0, 1.0, (1,), float64)
---flattened 🌎 Box(-inf, inf, (6,), float64)
---flattened 🕹️ Box(-1.0, 1.0, (1,), float64)
```

For most usage, we recommend applying the `gym.wrappers.FlattenObservation(env)` wrapper to reduce the `Dict` observation space to a `Box` observation space.

## Citation

If you use this in your research, please cite:
```
@software{shimmy2022github,
  author = {{Jun Jet Tai, Mark Towers} and Elliot Tower and Jordan Terry},
  title = {Shimmy: Gymnasium and PettingZoo Wrappers for Commonly Used Environments},
  url = {http://github.com/Farama-Foundation/Shimmy},
  version = {0.2.0},
  year = {2022},
}
```

```{toctree}
:hidden:
:caption: Environments
contents/index
```

```{toctree}
:hidden:
:caption: Development
Github <https://github.com/Farama-Foundation/shimmy>
```

