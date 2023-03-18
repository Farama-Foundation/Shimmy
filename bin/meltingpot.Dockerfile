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

# Install meltingpot (requires manual installation)
RUN git clone https://github.com/deepmind/meltingpot.git
RUN ./meltingpot/install-dmlab2d.sh
RUN ./meltingpot/install-meltingpot.sh
RUN ./meltingpot/install-extras.sh

ENV PYTHONPATH="/usr/local/shimmy/:/usr/local/shimmy/meltingpot"

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
