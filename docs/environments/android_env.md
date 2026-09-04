# AndroidEnv

## [DeepMind AndroidEnv](https://github.com/google-deepmind/android_env)

[AndroidEnv](https://github.com/google-deepmind/android_env) is a reinforcement learning environment built on top of the Android Emulator, allowing agents to interact with any Android application through touch and gesture inputs.

Shimmy provides a compatibility wrapper that converts an `AndroidEnv` instance into a [Gymnasium](https://gymnasium.farama.org/) environment, preserving the dict-valued observations and actions of the underlying environment.

## Installation
To install `shimmy` and required dependencies:

```
pip install shimmy[android-env]
```

You will additionally need a working Android Emulator and an AndroidEnv task `.textproto` file — see [the AndroidEnv documentation](https://github.com/google-deepmind/android_env/tree/main/docs) for setup instructions.

We also provide a [Dockerfile](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/android_env.Dockerfile) for the Shimmy side of the dependency tree:

```
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/android_env.Dockerfile | docker build -t android_env -f - . && docker run -it android_env
```

## Usage

Load an `AndroidEnv` instance and wrap it:

```python
from android_env import loader
from android_env.components import config_classes
from shimmy import AndroidEnvCompatibilityV0

config = config_classes.AndroidEnvConfig(
    task=config_classes.FilesystemTaskConfig(path="/path/to/task.textproto"),
    simulator=config_classes.EmulatorConfig(
        emulator_launcher=config_classes.EmulatorLauncherConfig(
            emulator_path="~/Android/Sdk/emulator/emulator",
            android_sdk_root="~/Android/Sdk",
            android_avd_home="~/.android/avd",
            avd_name="my_avd",
            run_headless=True,
        ),
        adb_controller=config_classes.AdbControllerConfig(
            adb_path="~/Android/Sdk/platform-tools/adb",
        ),
    ),
)
env = AndroidEnvCompatibilityV0(loader.load(config), render_mode="rgb_array")
```

Run the environment:

```python
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # dict with action_type / touch_position
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

Observations are dicts containing a `pixels` field (the device screen as an RGB array), which is also returned by `env.render()` when `render_mode="rgb_array"`.

AndroidEnv-specific extensions (`task_extras`, `execute_adb_call`, `load_state`, `save_state`, `stats`) are forwarded to the wrapped environment via attribute access.

## Class Description

```{eval-rst}
.. autoclass:: shimmy.android_env_compatibility.AndroidEnvCompatibilityV0
    :members:
    :undoc-members:
```
