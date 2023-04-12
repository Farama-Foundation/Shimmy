# A Dockerfile that sets up a full shimmy install with test dependencies
# adapted from https://github.com/deepmind/meltingpot/blob/main/.devcontainer/Dockerfile

# if PYTHON_VERSION is not specified as a build argument, set it to 3.9.
ARG PYTHON_VERSION
ARG PYTHON_VERSION=${PYTHON_VERSION:-3.9}
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
    cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

# Include Shimmy in Python path
ENV PYTHONPATH="$PYTHONPATH:/usr/local/shimmy/"

# Install Shimmy
RUN if [ -f "pyproject.toml" ]; then \
        pip install ".[meltingpot, testing]" --no-cache-dir; \
    else \
        pip install -U "shimmy[meltingpot, testing] @ git+https://github.com/Farama-Foundation/Shimmy.git" --no-cache-dir; \
        mkdir -p bin && mv docker_entrypoint bin/docker_entrypoint; \
    fi

# Install Melting Pot dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    build-essential \
    curl \
    ffmpeg \
    git

# Install lab2d (appropriate version for architecture)
RUN if [ "$(uname -m)" != 'x86_64' ]; then \
        echo "No Lab2d wheel available for $(uname -m) machines." >&2 \
        exit 1; \
    elif [ "$(uname -s)" = 'Linux' ]; then \
        pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl ;\
    else \
        pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-macosx_10_15_x86_64.whl ;\
    fi

# Download Melting Pot assets
RUN mkdir -p /workspaces/meltingpot/meltingpot && \
    curl -SL https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-2.1.0.tar.gz \
    | tar -xz --directory=/workspaces/meltingpot/meltingpot

# Clone Melting Pot repository
RUN git clone https://github.com/deepmind/meltingpot.git
RUN cp -r /meltingpot/ /workspaces/meltingpot/ && rm -R /meltingpot/
WORKDIR /workspaces/meltingpot/meltingpot/

# Install meltingpot dependencies
RUN pip install .

# Set Python path for meltingpot
ENV PYTHONPATH "${PYTHONPATH}:/workspaces/meltingpot/meltingpot/"

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]


