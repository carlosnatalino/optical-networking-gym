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

When you make changes to pyx files, you must compile the code using:

```bash
python setup.py clean --all build_ext --force --inplace
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

# Contributing

Contributions from the community are welcome.
To start the process, open an issue in GitHub.
Then, we can discuss the functionality, and if the feature you are interested in is of the interest of the maintainers.
After that, we can accept pull requests.

# Maintainers

- Carlos Natalino <carlos.natalino@chalmers.se>

# Citing

To cite this work, use the following reference:

```
@ARTICLE{Natalino_2024_gym,
  author={Natalino, Carlos and Magalhaes, Talles and Arpanaei, Farhad and Lobato, Fabricio R. L. and Costa, Joao C. W. A. and Hernandez, Jose Alberto and Monti, Paolo},
  journal={Journal of Optical Communications and Networking}, 
  title={{Optical Networking Gym}: an open-source toolkit for resource assignment problems in optical networks}, 
  year={2024},
  volume={16},
  number={12},
  pages={G40-G51},
  doi={10.1364/JOCN.532850},
}
```
