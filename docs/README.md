# PettingZoo docs

This folder contains the documentation for [Shimmy](https://github.com/Farama-Foundation/Shimmy).

For more information about how to contribute to the documentation go to our [CONTRIBUTING.md](https://github.com/Farama-Foundation/Shimmy/blob/main/CONTRIBUTING.md)
## Build the Documentation

Install the required packages and PettingZoo:

```
pip install -e .
pip install -r docs/requirements.txt
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```
