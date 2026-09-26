# Optical Networking Gym
An Open-Source Toolkit for Benchmarking Resource Assignment Problems in Optical Networks

## Installation

We recommend [uv](https://docs.astral.sh/uv/) for environment management:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Build the Cython extensions in place and run the tests in one command:

```bash
DEBUG=1 python setup.py build_ext --inplace && coverage run -m pytest && coverage report
```

Alternatively, using Python's built-in *venv* module and pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Optional dependency groups

- `dev`: test and quality tooling (`pytest`, `coverage`, `mypy`, `ruff`, `Cython`, `setuptools`).
- `research`: plotting and notebook tooling (`matplotlib`, `jupyterlab`, `pandas`).

Install both with:

```bash
uv pip install -e ".[dev,research]"
```

## Quickstart

```bash
python examples/quickstart/basic_first_fit.py
```

See `examples/INVENTORY.md` for the full list of examples, and `DEVELOPMENT.md`
for the development workflow (build, tests, lint, type check).

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md).

We recommend the use of VSCode with the extension `ktnrg45.vscode-cython` to enable code completion and highlighting in `.pyx` (Cython) files.

# Contributing

Contributions from the community are welcome.
To start the process, open an issue in GitHub.
Then, we can discuss the functionality, and if the feature you are interested in is of the interest of the maintainers.
After that, we can accept pull requests.

# Maintainers

- Carlos Natalino <carlos.natalino@chalmers.se>
