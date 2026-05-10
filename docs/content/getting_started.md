---
layout: "contents"
title: Getting Started
firstpage:
---
# Getting Started

## Installation
To install Shimmy from [PyPI](https://pypi.org/):
```
pip install shimmy
```
To install required dependencies for environments, specify them as follows:
```
pip install "shimmy[atari]"
```

Choices: `gym-v21`, `gym-v26`, `atari`, `dm-control`, `dm-control-multi-agent`, `openspiel`

For development and testing:

```
pip install "shimmy[all,testing]"
```


## Docker

[Docker](https://docs.docker.com/get-docker/) can be used for installation and reproducible environment creation on any platform, through virtualized application containers.
We provide [Dockerfiles](https://docs.docker.com/engine/reference/builder/) for each environment, located in [`/bin/`](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/)

### Build a Docker Image

To download environment's Dockerfile and build an image:

`
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/dm_control.Dockerfile | docker build -t dm_control -f - .
`

Or, clone our [GitHub repository](https://github.com/Farama-Foundation/shimmy) and build from the Dockerfile locally:

```
docker build -t dm_control -f bin/dm_control.Dockerfile .
```

### Running a Container

Then, to run the Docker container with an interactive bash terminal:
```
docker run -it dm_control bash
```

Stop the container:
```
docker stop dm_control
```

Remove the stopped container:
```
docker rm dm_control
```

For more information, see [Docker Documentation](https://docs.docker.com/get-started/).
* [Docker Build](https://docs.docker.com/engine/reference/commandline/build/)
* [Docker Run](https://docs.docker.com/engine/reference/commandline/run/)
