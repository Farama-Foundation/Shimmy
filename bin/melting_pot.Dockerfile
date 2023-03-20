ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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

RUN pip install ".[meltingpot, testing]" --no-cache-dir

# Install bazel (used for dmlab2d install)
RUN apt-get install apt-transport-https curl gnupg -y \
    && curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg \
    && mv bazel-archive-keyring.gpg /usr/share/keyrings \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && apt-get update && apt-get install bazel

# Install meltingpot (requires manual installation)
RUN git clone https://github.com/deepmind/meltingpot.git
WORKDIR meltingpot/
RUN ./install-dmlab2d.sh
RUN ./install-meltingpot.sh
RUN ./install-extras.sh

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
