# A Dockerfile that sets up dm-control

# if PYTHON_VERSION is not specified as a build argument, set it to 3.10.
ARG PYTHON_VERSION
ARG PYTHON_VERSION=${PYTHON_VERSION:-3.10}
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN pip install --upgrade pip

# Install bazel (via bazelisk) — required to build labmaze from source,
# pulled in transitively by dm_control on Python versions without a prebuilt wheel.
# Pin to bazel 5.4.1: labmaze's BUILD references @bazel_tools//platforms (removed in
# bazel 6+) and uses WORKSPACE-style external deps (default-disabled in bazel 7+).
RUN curl -fsSL https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
        -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel
ENV USE_BAZEL_VERSION=5.4.1

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

# Install Shimmy
RUN pip install ".[dm-control, testing]" --no-cache-dir

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
