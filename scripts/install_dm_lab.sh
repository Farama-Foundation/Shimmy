#!/bin/sh
set -eu

if [[ "$(uname -s)" == 'Linux' ]]; then
  # Dependencies
  apt-get update && apt-get install -y \
      build-essential curl freeglut3 gettext git libffi-dev libglu1-mesa \
      libglu1-mesa-dev libjpeg-dev liblua5.1-0-dev libosmesa6-dev \
      libsdl2-dev lua5.1 pkg-config python-setuptools python3-dev \
      software-properties-common unzip zip zlib1g-dev g++

  # Bazel
  apt-get install -y apt-transport-https curl gnupg
  curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
  mv bazel.gpg /etc/apt/trusted.gpg.d/
  echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
  apt-get update && apt-get install -y bazel

elif [[ "$(uname -s)" == 'Darwin' ]]; then
  which -s brew
  if [[ $? != 0 ]] ; then
    echo "Homebrew not installed, run: `ruby -e '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)'`"
    exit
  else
    brew install bazel
  fi
else
  echo "Install script does not support windows, Melting Pot and dmlab2d must be installed manually and are not natively supported"
  exit 1
fi


pip3 install numpy

# TODO: fix installation issues on MacOS
# Build
if [ ! -d "lab" ]; then
  git clone https://github.com/deepmind/lab.git
fi
cd lab
echo 'build --cxxopt=-std=c++17' > .bazelrc
bazel build -c opt //python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip3 install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl
cd ..
rm -rf lab
