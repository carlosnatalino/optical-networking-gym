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

Then, install the project:

```bash
pip install -e .
```

Alternatively, you can install with the development and research dependencies:

```bash
pip install -e ".[dev,research]"
```

<!-- Then, installing the necessary build tools:

```bash
pip install -U pip setuptools Cython numpy
```

Then, it is time to build the package:

```bash
python setup.py build_ext -i
```

Finally, we need to install the package: -->


## Development

To install the development dependencies, after the installation steps above, run:

```bash
pip install -e ".[dev]"
```

To build and run tests:

```bash
DEBUG=1 python setup.py clean --all build_ext --force --inplace && coverage run -m pytest && coverage report
```

We recommend the use of VSCode with the extension `ktnrg45.vscode-cython` to enable code completion and highlighting in `.pyx` (Cython) files.

## Research

To install the research dependencies, after the installation steps above, run:

```bash
pip install -e ".[research]"
```
