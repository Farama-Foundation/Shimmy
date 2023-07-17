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

# Install Shimmy
RUN if [ -f "pyproject.toml" ]; then \
        pip install ".[bsuite, testing]" --no-cache-dir; \
    else \
        pip install -U "shimmy[bsuite, testing] @ git+https://github.com/Farama-Foundation/Shimmy.git" --no-cache-dir; \
        mkdir -p bin && mv docker_entrypoint bin/docker_entrypoint; \
    fi

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
