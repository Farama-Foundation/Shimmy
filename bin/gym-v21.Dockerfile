# A Dockerfile that sets up gym v0.21

# if PYTHON_VERSION is not specified as a build argument, set it to 3.10.
ARG PYTHON_VERSION
ARG PYTHON_VERSION=${PYTHON_VERSION:-3.10}
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN pip install --upgrade pip

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

# Install Shimmy
RUN pip install ".[gym-v21, testing]" --no-cache-dir

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
