# A Dockerfile that sets up a full shimmy install with test dependencies

# if PYTHON_VERSION is not specified as a build argument, set it to 3.10.
ARG PYTHON_VERSION
ARG PYTHON_VERSION=${PYTHON_VERSION:-3.10}
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN pip install --upgrade pip

# Install Shimmy requirements
RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    unzip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

# Include Shimmy in Python path
ENV PYTHONPATH="$PYTHONPATH:/usr/local/shimmy/"

# Install Shimmy
RUN if [ -f "pyproject.toml" ]; then \
        pip install ".[dm-lab, testing]" --no-cache-dir; \
    else \
        pip install -U "shimmy[dm-lab, testing] @ git+https://github.com/Farama-Foundation/Shimmy.git" --no-cache-dir; \
        mkdir -p bin && mv docker_entrypoint bin/docker_entrypoint; \
    fi

# Install DM lab requirements
RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    build-essential curl freeglut3-dev gettext git libffi-dev libglu1-mesa \
    libglu1-mesa-dev libjpeg-dev liblua5.1-0-dev libosmesa6-dev \
    libsdl2-dev lua5.1 pkg-config python3-dev \
    software-properties-common unzip zip zlib1g-dev g++

# Install Bazel
RUN apt-get install -y apt-transport-https curl gnupg  \
    && curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg  \
    && mv bazel.gpg /etc/apt/trusted.gpg.d/  \
    && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && apt-get update && apt-get install -y bazel

# Build DM lab
RUN git clone https://github.com/deepmind/lab.git \
    && cd lab \
    && echo 'build --cxxopt=-std=c++17' > .bazelrc \
    && bazel build -c opt //python/pip_package:build_pip_package  \
    && ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg \
    && pip3 install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl \
    && cd .. \
    && rm -rf lab

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
