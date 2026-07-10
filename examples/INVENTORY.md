# Examples Inventory

What actually ships under `examples/`:

## Getting started

- `quickstart/basic_first_fit.py`: first-path canonical API example
  (`make_env` + first-fit heuristic over one episode).
- `basic_first_fit.py`: compatibility entry point for the quickstart example.
- `env_test.py`: visual smoke report of a short episode.

## Heuristics

- `heuristics/masked_first_fit.py`, `heuristics/masked_random.py`: policies
  driven by the action mask.
- `heuristics/runtime_first_fit.py`, `heuristics/runtime_random.py`: policies
  driven by the runtime heuristic context.
- `heuristics/load_sweep.py`: sweep episode blocking across loads and policies.
- `heuristics/static_first_fit_trace.py`: replay a captured traffic table and
  write a step trace.
- `load_sweep.py`, `static_first_fit_trace.py`: compatibility wrappers for the
  scripts above.

## Benchmarks and research

- `JOCN_Benchmark_2024/`: reproduction of the JOCN 2024 benchmark scenario
  (`graph_load.py`, `graph_margin.py`, `graph_launch_power.py`, `plots.ipynb`,
  `README.md`).
- `legacy_benchmark/osnr_margin_sweep.py` (+ analysis notebook): OSNR margin
  sweep matching the legacy benchmark setup.

## Analysis and tooling

- `analysis/debug_profiling.py`, `debug_profiling.py`: local profiling helpers.
- `rl/random_policy.py` and `rl/README.md`: RL smoke/example material.

## Outputs

- Generated outputs are written under `examples/**/results/` and are gitignored.
