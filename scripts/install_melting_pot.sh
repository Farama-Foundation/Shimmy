#!/bin/sh
set -eu

# Install bazel (Linux or MacOS)
if [[ "$(uname -s)" == 'Linux' ]]; then
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

# Build
if [ ! -d "meltingpot" ]; then
  git clone https://github.com/deepmind/meltingpot.git
else
  echo "directory already exists"
fi
cd meltingpot
chmod +x *.sh
echo "Installing dmlab2d..."
./install-dmlab2d.sh
echo "Installing Melting Pot..."
./install-meltingpot.sh
cd ..
rm -rf meltingpot
echo "Finished installing Melting Pot"
