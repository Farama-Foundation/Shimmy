# from https://github.com/deepmind/meltingpot/blob/main/.devcontainer/Dockerfile
# Use Nvidia Ubuntu 20 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04@sha256:b754c43fe9d62e88862d168c4ab9282618a376dbc54871467870366cacfa456e

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  curl \
  ffmpeg \
  git \
  python3.9 \
  python3.9-dev \
  python3.9-distutils \
  rsync

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install python dev dependencies
RUN pip install \
  jedi \
  pytest \
  pylance \
  toml \
  yapf

# Install lab2d (appropriate version for architecture)
RUN if [ "$(uname -m)" != 'x86_64' ]; then \
    echo "No Lab2d wheel available for $(uname -m) machines." >&2 \
    exit 1; \
  elif [ "$(uname -s)" = 'Linux' ]; then \
    pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl ;\
  else \
    pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-macosx_10_15_x86_64.whl ;\
  fi

# Download assets
RUN mkdir -p /workspaces/meltingpot/meltingpot && \
  curl -SL https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz \
  | tar -xz --directory=/workspaces/meltingpot/meltingpot

# Set Python path
ENV PYTHONPATH="/workspaces/meltingpot"

# Set path for shimmy
COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

RUN pip install ".[openspiel, testing]" --no-cache-dir

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
