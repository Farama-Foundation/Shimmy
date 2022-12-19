---
title: Installation and Usage
firstpage:
---


## Installation and Usage


To install Shimmy from PyPI:
```
pip install shimmy
```
Out of the box, Shimmy doesn't install any of the dependencies required for the environments it supports.
To install them, you'll have to install the optional extras.
All single agent environments have registration under the Gymnasium API, while all multiagent environments must be wrapped using the corresponding compatibility wrappers.

### For Developers and Testing Only
```
pip install shimmy[testing]
```

### To just install everything
```
pip install shimmy[all, testing]
```
