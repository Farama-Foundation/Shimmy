## DM Lab

### Installation

Courtesy to [Danijar Hafner](https://github.com/deepmind/lab/issues/242) for providing this install script.
```bash
#!/bin/sh
set -eu

# Dependencies
apt-get update && apt-get install -y \
    build-essential curl freeglut3 gettext git libffi-dev libglu1-mesa \
    libglu1-mesa-dev libjpeg-dev liblua5.1-0-dev libosmesa6-dev \
    libsdl2-dev lua5.1 pkg-config python-setuptools python3-dev \
    software-properties-common unzip zip zlib1g-dev g++
pip3 install numpy

# Bazel
apt-get install -y apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
apt-get update && apt-get install -y bazel

# Build
git clone https://github.com/deepmind/lab.git
cd lab
echo 'build --cxxopt=-std=c++17' > .bazelrc
bazel build -c opt //python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip3 install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl
cd ..
rm -rf lab
```

### Usage
```python
import deepmind_lab

from shimmy.dm_lab_compatibility import DmLabCompatibilityV0

observations = ["RGBD"]
config = {"width": "640", "height": "480", "botCount": "2"}
renderer = "hardware"

env = deepmind_lab.Lab("lt_chasm", observations, config=config, renderer=renderer)
env = DmLabCompatibilityV0(env)
```

### Class Description
```{eval-rst}
.. autoclass:: shimmy.dm_lab_compatibility.DmLabCompatibilityV0
    :members:
    :undoc-members:
```
