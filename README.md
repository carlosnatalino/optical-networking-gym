# Optical Networking Gym
An Open-Source Toolkit for Benchmarking Resource Assignment Problems in Optical Networks

## Installation

We recommend installing an environment using Python's *venv* module:

```bash
python -m venv .venv
```

Then activating it:

```bash
source .venv/bin/activate
```

Then, installing the necessary build tools:

```bash
pip install -U pip setuptools Cython
```

Then, it is time to build the package:

```bash
python setup.py build_ext -i
```

## Development

To build and run tests:

```bash
CYTHON_TRACE=1 python setup.py build_ext -i && coverage run -m pytest && coverage report
```
