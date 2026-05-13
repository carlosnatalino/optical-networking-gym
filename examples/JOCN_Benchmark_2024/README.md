# JOCN 2024 Benchmark Reproduction

Run these commands from this package directory with the repository Python environment.

```powershell
..\.venv\Scripts\python.exe examples\JOCN_Benchmark_2024\graph_load.py --topology-id nobel-eu --workers 1
..\.venv\Scripts\python.exe examples\JOCN_Benchmark_2024\graph_margin.py --topology-id nobel-eu --load 210 --workers 1
..\.venv\Scripts\python.exe examples\JOCN_Benchmark_2024\graph_launch_power.py --topology-id nobel-eu --load 210 --workers 1
```

For a quick local smoke run:

```powershell
..\.venv\Scripts\python.exe examples\JOCN_Benchmark_2024\graph_load.py --topology-id ring_4 --loads 10 --episodes-per-point 1 --request-count 8 --workers 1
```

Each script writes a standard run directory under:

```text
examples/results/JOCN_Benchmark_2024/<script-name>/<YYYYMMDD-HHMMSS>/
  metadata.json
  episodes.csv
  summary.csv
```

`plots.ipynb` reads those CSV files and generates the load, margin, and launch-power plots locally.
