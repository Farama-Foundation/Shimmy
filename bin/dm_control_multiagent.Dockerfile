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
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

RUN if [ -f "pyproject.toml" ]; then \
        pip install ".[dm-control-multi-agent, testing]" --no-cache-dir; \
    else \
        pip install -U "shimmy[dm-control-multi-agent, testing] @ git+https://github.com/Farama-Foundation/Shimmy.git" --no-cache-dir; \
    fi

ENTRYPOINT ["/usr/local/shimmy/docker_entrypoint"]
