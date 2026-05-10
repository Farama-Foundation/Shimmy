# A Dockerfile that sets up gym v0.21

# if PYTHON_VERSION is not specified as a build argument, set it to 3.10.
ARG PYTHON_VERSION
ARG PYTHON_VERSION=${PYTHON_VERSION:-3.10}
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

# Install Shimmy.
#   gym 0.21.0's metadata contains an invalid PEP 508 spec ("opencv-python (>=3.)")
#   that pip 24.1+ rejects outright. Pin pip below 24.1 for this image.
RUN pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40.0"
RUN pip install --no-build-isolation ".[gym-v21, testing]" --no-cache-dir
