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

## Memory use

The request-analysis cache is bounded by `ScenarioConfig.request_buffer_limit`
(default `8` analyses, least recently used evicted first; `-1` is unlimited, as
in 0.3.0 and earlier, and `0` disables it). An analysis takes about 0.2–0.3 MB
on nobel-eu with 320 slots and the cache is never hit in a plain step loop, so
an unbounded cache grew by that much at every arrival. Results do not depend on
the limit. The opt-in `capture_traffic_table` and `capture_step_trace` buffers
stay unbounded and grow with every arrival. See
[docs/docs/get_started.md](docs/docs/get_started.md#9-memory-use-of-long-simulations).

## Physical layer

The QoT engine supports a heterogeneous physical layer (per-span fibre loss,
amplifier NF, lumped losses and gain ripple) and runtime updates of the span
parameters during a simulation, e.g. to model network aging
(`env.update_spans([SpanUpdate(link_id, span_index, attenuation_db_per_km=...,
noise_figure_db=...)])`), keeping the traffic state. See
[docs/docs/physical_layer.md](docs/docs/physical_layer.md).

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
