# Examples Inventory

## keep-public

- `quickstart/basic_first_fit.py`: first-path canonical API example.
- `basic_first_fit.py`: compatibility entry point for the first-fit example.
- `env_test.py`: visual smoke report.
- `SBRT2026/osnr_margin_sweep.py`: publication margin sweep.
- `SBRT2026/trace_disruptions.py`: disruption trace script.
- `SBRT2026/judge_heuristics_load_sweep.py`: SBRT2026 entry point for judge heuristic sweep.

## keep-advanced

- `heuristics/*.py`: advanced heuristic and runtime comparison examples.
- `llm/*.py`: judge research scripts and support tools.
- `legacy_benchmark/*.py`: legacy benchmark comparison examples.
- `analysis/*.py`: local analysis/profiling helpers.
- `static_first_fit_trace.py`: compatibility wrapper for the heuristic trace example.
- `rl/random_policy.py` and `rl/README.md`: RL smoke/example material.

## archive-later

- Historical generated outputs under `examples/**/results/`.
- Locked legacy `SBRT26/results/` contents left in place by Windows; ignored by `.gitignore`.

## remove

- No source scripts were removed in this milestone. Generated outputs are ignored and can be deleted once no process holds them open.
