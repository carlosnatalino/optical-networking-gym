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

All scripts use the `jocn_benchmark` preset. As in the gym used for the article, its GSNR
includes the cross-channel interference (XCI) of every lightpath established on the route
(`nli_include_interferers=True`, interferers at the channel-under-test PSD), so it depends
on the network load. Results generated before this setting was added to the preset
(ASE and self-channel NLI only) are optimistic: at 210 Erlang the mean GSNR of the accepted
lightpaths is about 1.2 dB lower with XCI, and fewer lightpaths use 64QAM.

## QoT datasets (Section 5.C, Fig. 5)

`generate_dataset.py` builds one AI/ML QoT dataset per provisioning heuristic: after a
warm-up, it processes 100,000 arrivals and saves every accepted lightpath at establishment
(route, spectrum, launch power, modulation format, GSNR with its ASE/NLI split, per-link
ASE/NLI noise, link occupancy and all co-propagating channels), plus the link and span
tables of the network. The defaults are the paper's two most divergent heuristics,
BM-LS-KSP and LB-BM-KSP (`LS-BM-KSP` and `KSP-LB-BM` in the gym), on nobel-eu at
210 Erlang and -4 dBm. It needs the `research` extra (`xarray`, `h5netcdf`, `h5py`).

```bash
python examples/JOCN_Benchmark_2024/generate_dataset.py --workers 2
# quick smoke run
python examples/JOCN_Benchmark_2024/generate_dataset.py --arrivals 2000 --warmup 500
# also save the QoT queries of 10% of the arrivals (see below)
python examples/JOCN_Benchmark_2024/generate_dataset.py --query-fraction 0.1
```

The per-link NLI is split into its self-channel (`hop_sci_nsr`) and cross-channel
(`hop_xci_nsr`) parts, and the file attributes `qot_nli_include_interferers`,
`qot_nli_interferer_psd` and `qot_nli_modulation_correction` state the QoT model.

A dataset of accepted lightpaths is shaped by the heuristic that selected them. With
`--query-fraction f`, a `query` table also records, for a fraction `f` of the recorded
arrivals, every candidate path and modulation format at its first free (first-fit) start
slot with the GSNR the environment computed before the heuristic decided (the population a
QoT estimator serves in operation). It is collected with the `OpticalEnv.on_request_analysed`
hook, from a dedicated random stream, so the traffic and the lightpath table do not change.

It writes one netCDF file per run and per heuristic,
`jocn2024_qot_<topology>_<load>erl_<policy>_seed<seed>_<run_id>.nc` (open with
`xarray.open_dataset`), plus `summary.csv` (blocking, GSNR statistics and
modulation-format shares, the quantities of Fig. 5) and `metadata.json`, to
`results/JOCN_Benchmark_2024/generate_dataset/<run_id>/`.

Each file is self-contained: besides the samples it holds the link, span and
k-shortest-path tables, node coordinates, the full scenario configuration (JSON) and the
original topology file, so `load_topology(dataset)` in `generate_dataset.py` rebuilds the
gym `TopologyModel` from the file alone. Every variable carries `units` and `description`
attributes; integer variables point from one table to another (e.g. `path_id` into `path`,
`hop_link` into `link`, `span_link` into `link`), and `copropagating_*` is a ragged table
whose rows belong to the lightpath `copropagating_lightpath` at hop `copropagating_hop`.
As in the gym, one path record serves both directions of a node pair: `path_reversed`
tells whether a lightpath reads its path backwards, and all per-hop variables are stored
in traversal order, hop 0 being the link that leaves the source.

`dataset_plots.ipynb` reads the files of a run and reproduces Fig. 5(a) and 5(b), shows
how to follow the relations between lightpaths, routes, links, spans and co-propagating
channels, and adds other views (GSNR vs route length, ASE vs NLI, route choice, margins,
link occupancy and a map of link usage). Figures are saved to `<run>/figures/`.
