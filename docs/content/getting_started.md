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
pip install shimmy[bsuite, atari]
```

Choices: `gym-v21`, `gym-v26`, `atari`, `bsuite`, `dm-control`, `dm-control-multi-agent`, `openspiel`, `meltingpot`

For development and testing:

```
pip install shimmy[all, testing]
```


### Docker

[Docker](https://docs.docker.com/get-docker/) can be used for reproducible environment creation on any platform, through containerized virtualized application containers.
We provide [Dockerfiles](https://docs.docker.com/engine/reference/builder/) for each environment, located in [`/bin/`](https://github.com/Farama-Foundation/Shimmy/blob/main/bin/) 

To download and run a Dockerfile for an environment:

`
curl https://raw.githubusercontent.com/Farama-Foundation/Shimmy/main/bin/dm_lab.Dockerfile | docker build -t dm_lab -f - . && docker run -it dm_lab
`

Or, clone our [GitHub repository](https://github.com/Farama-Foundation/shimmy) and run the Dockerfile locally:

```
docker build -t dm_lab -f bin/dm_lab.Dockerfile && docker run -it dm_lab
```

Stop the container:
``` 
docker stop dm_lab
```

Remove the stopped container:
``` 
docker rm dm_lab
```

For more information, see [Docker Documentation](https://docs.docker.com/get-started/).
