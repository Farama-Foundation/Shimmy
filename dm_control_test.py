import numpy as np
from dm_control import suite
from PIL import Image

from shimmy.dm_control_compatibility import dm_control_wrapper

# Load the environment
random_state = np.random.RandomState(42)
env = suite.load("hopper", "stand", task_kwargs={"random": random_state})

# convert the environment
env = dm_control_wrapper(env, render_mode="rgb_array")
env.reset()

frames = []
for _ in range(100):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    frames.append(env.render())
print(len(frames))

frames = [Image.fromarray(frame) for frame in frames]
frames[0].save(
    "array.gif", save_all=True, append_images=frames[1:], duration=50, loop=0
)
