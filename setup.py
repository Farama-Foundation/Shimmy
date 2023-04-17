"""Setups up the Shimmy module."""
from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md", encoding="utf-8") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Gets the shimmy version."""
    path = "shimmy/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


version = get_version()
header_count, long_description = get_description()

extras = {
    "gym-v21": ["gym>=0.21.0", "pyglet==1.5.11"],
    "gym-v26": ["gym>=0.26.2"],
    "atari": ["ale-py~=0.8.1"],
    # "imageio" should be "gymnasium[mujoco]>=0.26" but there are install conflicts
    "dm-control": ["dm-control>=1.0.10", "imageio", "h5py>=3.7.0"],
    "dm-control-multi-agent": [
        "dm-control>=1.0.10",
        "imageio",
        "h5py>=3.7.0",
        "pettingzoo>=1.22.3",
    ],
    "dm-lab": ["dm-env>=1.6"],
    "openspiel": ["open_spiel>=1.2", "pettingzoo>=1.22.3"],
    "meltingpot": ["pettingzoo>=1.22.3"],
    "bsuite": ["bsuite>=0.3.5"],
}
extras["all"] = list({lib for libs in extras.values() for lib in libs})
extras["testing"] = [
    "pytest==7.1.3",
    "pillow>=9.3.0",
    "autorom[accept-rom-license]~=0.6.0",
]

setup(
    name="Shimmy",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="An API conversion tool providing Gymnasium and PettingZoo bindings for popular external reinforcement learning environments.",
    url="https://github.com/Farama-Foundation/Shimmy",
    license_files=("LICENSE.txt",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "game", "RL", "AI"],
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=["numpy>=1.18.0", "gymnasium>=0.27.0"],
    tests_require=extras["testing"],
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    entry_points={
        "gymnasium.envs": ["__root__ = shimmy.registration:register_gymnasium_envs"]
    },
)
