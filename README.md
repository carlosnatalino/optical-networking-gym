# optical_networking_gym_v2

`optical_networking_gym_v2` is the isolated rewrite of the optical networking
environment. The goal is to keep the `v2` runtime independent from the legacy
implementation while making the hot path easier to benchmark, profile, and move
to Cython.

## Scope

- `src/optical_networking_gym_v2`: runtime, contracts, RL helpers, topology,
  QoT, simulation, and Cython kernels
- `tests/`: coverage for contracts, equivalence, simulator behavior, examples,
  and performance harnesses
- `benchmarks/`: focused scripts and comparison artifacts for hot-path analysis
- `examples/`: runnable scripts for smoke tests, traces, and visual inspection

This subproject should not use the legacy runtime as an implementation layer.

## Quick Start

From the outer workspace root:

```powershell
.\.venv\Scripts\python.exe optical_networking_gym_v2\setup.py build_ext --inplace
.\.venv\Scripts\python.exe -m pytest optical_networking_gym_v2\tests -q
```

Create an environment with the canonical preset API:

```python
from optical_networking_gym_v2 import make_env

env = make_env(scenario="ring4_quickstart")
env = make_env(scenario="nobel_eu_baseline", load=400, margin=2.0)
env = make_env(scenario="nobel_eu_baseline", modulations="BPSK,QPSK,16QAM")
```

Use `ScenarioConfig` directly only for advanced custom scenarios. Examples write
standard run folders under `examples/results/<family>/<script-stem>/<timestamp>/`
with `metadata.json`, `episodes.csv`, and `summary.csv` when applicable.

From inside `optical_networking_gym_v2`:

```powershell
..\.venv\Scripts\python.exe setup.py build_ext --inplace
..\.venv\Scripts\python.exe -m pytest tests -q
```

If a `.pyx` file changes and imports look stale, force a clean rebuild:

```powershell
..\.venv\Scripts\python.exe setup.py clean --all build_ext --force --inplace
```

## Common Commands

Run the full v2 suite:

```powershell
.\.venv\Scripts\python.exe -m pytest optical_networking_gym_v2\tests -q
```

Run the benchmark harness:

```powershell
.\.venv\Scripts\python.exe optical_networking_gym_v2\benchmarks\run_hot_path_benchmarks.py
```

Run the integrated comparison harness:

```powershell
.\.venv\Scripts\python.exe optical_networking_gym_v2\benchmarks\run_integrated_benchmarks.py
```

Run the visual smoke test report:

```powershell
.\.venv\Scripts\python.exe optical_networking_gym_v2\examples\env_test.py --steps 6 --no-open
```

The visual report is written to `optical_networking_gym_v2/examples/results/`
and includes:

- smoke-test assertions for reset, action mask, and heuristic validity
- per-step action mask visualization
- per-link spectrum occupancy visualization
- a raw JSON payload next to the HTML report

See `examples/README.md` for the public example families and parallelism
semantics.

## Layout

```text
optical_networking_gym_v2/
  benchmarks/   profiling and comparison scripts
  examples/     runnable demos and visual smoke tests
  src/          v2 package source
  tests/        automated verification
```

## Notes

- Use the repo-local Python 3.11 environment from the outer workspace.
- Cython-generated build outputs under `build/` and local report artifacts under
  `examples/results/` are ignored by `.gitignore`.
- The checked-in source tree under `src/` is the authoritative implementation
  for the `v2` runtime.
