---
layout: "contents"
title: Basic Usage
firstpage:
---
# Basic Usage

Shimmy provides API compatibility tools to adapt popular external reinforcement learning environments to work with [Gymnasium](https://github.com/farama-Foundation/gymnasium) and [PettingZoo](https://github.com/farama-Foundation/pettingZoo/).

## Single-agent

Single-agent [Gymnasium](https://gymnasium.farama.org/) environments can be loaded via `gym.make()`:

```python
import gymnasium as gym
env = gym.make("dm_control/acrobot-swingup_sparse-v0")
```

Run the environment:

```python
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

## Multi-agent
Multi-agent [PettingZoo](https://pettingzoo.farama.org) environments can be loaded via Shimmy `Compatibility` wrappers.

### AEC Environments

Load the environment:

```python
from shimmy import OpenSpielCompatibilityV0
env = OpenSpielCompatibilityV0(game_name="backgammon", render_mode="human")
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

### Parallel Environments

Load the environment:

```python
from shimmy import MeltingPotCompatibilityV0
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena")
```

Run the environment:
```python
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```


### Conversion
Environments loaded as [`ParallelEnv`](https://pettingzoo.farama.org/api/parallel/) can be converted to [`AECEnv`](https://pettingzoo.farama.org/api/aec/) using [`parallel_to_aec`](https://pettingzoo.farama.org/api/pz_wrappers/#parallel-to-aec).

Environments loaded as [`AECEnv`](https://pettingzoo.farama.org/api/aec/) can be converted to [`ParallelEnv`](https://pettingzoo.farama.org/api/parallel/) using [`parallel_to_aec`](https://pettingzoo.farama.org/api/pz_wrappers/#parallel-to-aec)
* Note: this conversion makes the following assumptions about the underlying environment:
  1. The environment steps in a cycle, i.e. it steps through every live agent in order.
  2. The environment does not update the observations of the agents except at the end of a cycle.

For more information, see [PettingZoo Wrappers](https://pettingzoo.farama.org/api/pz_wrappers/).









[//]: # (## Initializing Environments)

[//]: # ()
[//]: # (Single-agent [Gymnasium]&#40;https://gymnasium.farama.org/&#41; environments can be loaded via `gym.make&#40;&#41;`:)

[//]: # ()
[//]: # (```python)

[//]: # (import gymnasium as gym)

[//]: # (env = gym.make&#40;"dm_control/acrobot-swingup_sparse-v0"&#41;)

[//]: # (```)

[//]: # (Multi-agent [PettingZoo]&#40;https://pettingzoo.farama.org&#41; environments can be loaded via imported Shimmy wrappers:)

[//]: # ()
[//]: # (```python)

[//]: # (from shimmy import MeltingPotCompatibilityV0)

[//]: # (env = MeltingPotCompatibilityV0&#40;substrate_name="prisoners_dilemma_in_the_matrix__arena"&#41;)

[//]: # (```)

[//]: # ()
[//]: # (## Interacting with the Environment)

[//]: # (Single-agent [Gymnasium]&#40;https://gymnasium.farama.org/&#41; environments can be used as follows:)

[//]: # ()
[//]: # (```python )

[//]: # (observation, info = env.reset&#40;seed=42&#41;)

[//]: # (for _ in range&#40;1000&#41;:)

[//]: # (   action = env.action_space.sample&#40;&#41;  # this is where you would insert your policy)

[//]: # (   observation, reward, terminated, truncated, info = env.step&#40;action&#41;)

[//]: # ()
[//]: # (   if terminated or truncated:)

[//]: # (      observation, info = env.reset&#40;&#41;)

[//]: # (env.close&#40;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (Multi-agent [PettingZoo]&#40;https://pettingzoo.farama.org&#41; environments can be used as follows::)

[//]: # (```python)

[//]: # (observations = env.reset&#40;&#41;)

[//]: # (while env.agents:)

[//]: # (    actions = {agent: env.action_space&#40;agent&#41;.sample&#40;&#41; for agent in env.agents})

[//]: # (    observations, rewards, terminations, truncations, infos = env.step&#40;actions&#41;)

[//]: # (    env.step&#40;actions&#41;)

[//]: # (env.close&#40;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (## At a glance)

[//]: # ()
[//]: # (This is an example of using Shimmy to convert DM Control environments into a Gymnasium compatible environment:)

[//]: # ()
[//]: # (```python)

[//]: # (import gymnasium as gym)

[//]: # (from shimmy.registration import DM_CONTROL_SUITE_ENVS)

[//]: # ()
[//]: # (env_ids = [f"dm_control/{'-'.join&#40;item&#41;}-v0" for item in DM_CONTROL_SUITE_ENVS])

[//]: # (print&#40;env_ids&#41;)

[//]: # ()
[//]: # (env = gym.make&#40;env_ids[0]&#41;)

[//]: # (env_flatten = gym.wrappers.FlattenObservation&#40;env&#41;)

[//]: # (print&#40;env_ids[0]&#41;)

[//]: # (print&#40;"===🌎", env.observation_space&#41;)

[//]: # (print&#40;"===🕹️", env.action_space&#41;)

[//]: # (print&#40;"---flattened 🌎", env_flatten.observation_space&#41;)

[//]: # (print&#40;"---flattened 🕹️", env_flatten.action_space&#41;)

[//]: # (```)

[//]: # (```bash)

[//]: # (['dm_control/acrobot-swingup-v0', 'dm_control/acrobot-swingup_sparse-v0', 'dm_control/ball_in_cup-catch-v0', 'dm_control/cartpole-balance-v0', 'dm_control/cartpole-balance_sparse-v0', 'dm_control/cartpole-swingup-v0', 'dm_control/cartpole-swingup_sparse-v0', 'dm_control/cartpole-two_poles-v0', 'dm_control/cartpole-three_poles-v0', 'dm_control/cheetah-run-v0', 'dm_control/dog-stand-v0', 'dm_control/dog-walk-v0', 'dm_control/dog-trot-v0', 'dm_control/dog-run-v0', 'dm_control/dog-fetch-v0', 'dm_control/finger-spin-v0', 'dm_control/finger-turn_easy-v0', 'dm_control/finger-turn_hard-v0', 'dm_control/fish-upright-v0', 'dm_control/fish-swim-v0', 'dm_control/hopper-stand-v0', 'dm_control/hopper-hop-v0', 'dm_control/humanoid-stand-v0', 'dm_control/humanoid-walk-v0', 'dm_control/humanoid-run-v0', 'dm_control/humanoid-run_pure_state-v0', 'dm_control/humanoid_CMU-stand-v0', 'dm_control/humanoid_CMU-run-v0', 'dm_control/lqr-lqr_2_1-v0', 'dm_control/lqr-lqr_6_2-v0', 'dm_control/manipulator-bring_ball-v0', 'dm_control/manipulator-bring_peg-v0', 'dm_control/manipulator-insert_ball-v0', 'dm_control/manipulator-insert_peg-v0', 'dm_control/pendulum-swingup-v0', 'dm_control/point_mass-easy-v0', 'dm_control/point_mass-hard-v0', 'dm_control/quadruped-walk-v0', 'dm_control/quadruped-run-v0', 'dm_control/quadruped-escape-v0', 'dm_control/quadruped-fetch-v0', 'dm_control/reacher-easy-v0', 'dm_control/reacher-hard-v0', 'dm_control/stacker-stack_2-v0', 'dm_control/stacker-stack_4-v0', 'dm_control/swimmer-swimmer6-v0', 'dm_control/swimmer-swimmer15-v0', 'dm_control/walker-stand-v0', 'dm_control/walker-walk-v0', 'dm_control/walker-run-v0'])

[//]: # (dm_control/acrobot-swingup-v0)

[//]: # (===🌎 Dict&#40;'orientations': Box&#40;-inf, inf, &#40;4,&#41;, float64&#41;, 'velocity': Box&#40;-inf, inf, &#40;2,&#41;, float64&#41;&#41;)

[//]: # (===🕹️ Box&#40;-1.0, 1.0, &#40;1,&#41;, float64&#41;)

[//]: # (---flattened 🌎 Box&#40;-inf, inf, &#40;6,&#41;, float64&#41;)

[//]: # (---flattened 🕹️ Box&#40;-1.0, 1.0, &#40;1,&#41;, float64&#41;)

[//]: # (```)

[//]: # ()
[//]: # (For most usage, we recommend applying the `gym.wrappers.FlattenObservation&#40;env&#41;` wrapper to reduce the `Dict` observation space to a `Box` observation space.)
