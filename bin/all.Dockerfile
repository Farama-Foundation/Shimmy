# A Dockerfile that sets up a full shimmy install with test dependencies

# if PYTHON_VERSION is not specified as a build argument, set it to 3.10.
ARG PYTHON_VERSION
ARG PYTHON_VERSION=${PYTHON_VERSION:-3.10}
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN pip install --upgrade pip

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    unzip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg cmake \
    curl ca-certificates \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install bazel (via bazelisk) — required to build labmaze from source,
# pulled in transitively by dm_control on Python versions without a prebuilt wheel.
# Pin to bazel 6.5.0: labmaze's build uses WORKSPACE-style @bazel_skylib, which
# bazel 7+ no longer resolves by default (bzlmod is now the default).
RUN curl -fsSL https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
        -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel
ENV USE_BAZEL_VERSION=6.5.0

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

RUN if [ -f "pyproject.toml" ]; then \
        pip install ".[all, testing]" --no-cache-dir; \
    else \
        pip install -U "shimmy[all, testing] @ git+https://github.com/Farama-Foundation/Shimmy.git" --no-cache-dir; \
        mkdir -p bin && mv docker_entrypoint bin/docker_entrypoint; \
    fi

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
