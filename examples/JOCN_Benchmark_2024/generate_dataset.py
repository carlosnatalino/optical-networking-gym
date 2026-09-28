"""QoT dataset generation (JOCN 2024, Section 5.C and Fig. 5).

Generates one AI/ML QoT dataset per provisioning heuristic, as in the dataset
use case of [Natalino_2024_OpticalNetworkingGym]: the network processes a
stream of dynamic requests, and the information of every *accepted* lightpath
is saved at the moment it is established.

The setup follows the paper's use cases (the ``jocn_benchmark`` preset):
nobel-eu, 5 shortest paths, 320 slots of 12.5 GHz, 80 km spans with
0.2 dB/km and 4.5 dB noise figure, bit rates {10, 40, 100, 400} Gb/s, six
modulation formats (BPSK to 64QAM) and a flat -4 dBm launch power. The GSNR
is the closed-form EGN model with ASE, self-channel interference (SCI) and the
cross-channel interference (XCI) of every lightpath established on the route
(``nli_include_interferers=True``, as in the article), so it depends on the
load. The two default heuristics are the ones with the most divergent blocking in the
paper, BM-LS-KSP and LB-BM-KSP, available in the gym as ``LS-BM-KSP`` and
``KSP-LB-BM``.

Each heuristic runs one continuous stream: ``--warmup`` arrivals bring the
network to steady state and are not recorded, then ``--arrivals`` arrivals
(100,000 by default) are processed and every accepted lightpath becomes a
sample. All heuristics see the same arrival sequence (same seed).

Each dataset is a self-documented netCDF file (xarray, h5netcdf engine) with:

* ``lightpath``: request, route, spectrum, launch power, modulation format and
  the GSNR with its ASE and NLI components at establishment;
* ``lightpath x hop``: the links traversed, in order from the source (hop 0) to
  the destination, their occupancy and each link's ASE and NLI (with its SCI
  and XCI parts) noise-to-signal ratio (per-element noise breakdown);
* ``copropagating``: every channel sharing a traversed link at establishment
  (spectrum, bit rate, modulation format, launch power), as a ragged array
  indexed by ``copropagating_lightpath`` and ``copropagating_hop``;
* ``link`` and ``span``: the physical description of the network (lengths,
  attenuation, amplifier noise figures, lumped losses);
* ``path``: every k-shortest path of the topology (nodes, links, length), with
  the gym's path ids. As in the gym, one record serves both directions of a
  node pair, so a sample's route is ``path_id`` read forwards, or backwards
  when ``path_reversed`` is 1. The gym's physical model is undirected
  (lightpaths are bidirectional), so the GSNR and the per-link noise do not
  depend on the direction;
* ``modulation`` and ``node`` (with coordinates when the topology has them):
  lookup tables;
* ``query`` (only with ``--query-fraction`` > 0): the QoT queries of the
  RMSA. For a sampled fraction of the recorded arrivals, every candidate path
  and modulation format at its first free (first-fit) start slot, with the
  GSNR the environment computed for it before the heuristic decided, from the
  ``OpticalEnv.on_request_analysed`` hook. Unlike the ``lightpath`` table,
  these are not selected by the heuristic, so they are the population a QoT
  estimator serves in operation.

Each file is self-contained: its attributes hold the full scenario
configuration (JSON), the original topology file and the parameters used to
load it, so :func:`load_topology` rebuilds the gym ``TopologyModel`` from the
file alone, plus the run provenance (gym version and commit, seed, time).

Outputs go to ``<output-dir>/JOCN_Benchmark_2024/generate_dataset/<run_id>/``:
one file per run and per heuristic,
``jocn2024_qot_<topology>_<load>erl_<policy>_seed<seed>_<run_id>.nc``, plus
``summary.csv`` (blocking, GSNR and modulation-format statistics of Fig. 5)
and ``metadata.json``. ``dataset_plots.ipynb`` reads the files and draws the
figures.

Requires the ``research`` extra (``xarray``, ``h5netcdf``)::

    uv pip install -e ".[research]"
    python examples/JOCN_Benchmark_2024/generate_dataset.py --workers 2

References are listed at the end of the file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from optical_networking_gym import build_scenario
from optical_networking_gym.contracts import StepTransition
from optical_networking_gym.defaults import resolve_topology
from optical_networking_gym.envs import OpticalEnv
from optical_networking_gym.network.topology import TopologyModel
from optical_networking_gym.utils.experiment_utils import select_policy_action

SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY = "JOCN_Benchmark_2024"
SCRIPT_NAME = "generate_dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_POLICIES = ("LS-BM-KSP", "KSP-LB-BM")
PAPER_NAMES = {"LS-BM-KSP": "BM-LS-KSP", "KSP-LB-BM": "LB-BM-KSP"}
HIGH_GSNR_DB = 20.0  # Fig. 5(a) discusses the share of lightpaths above 20 dB.


@dataclass(frozen=True, slots=True)
class DatasetExperiment:
    """Parameters of one dataset-generation run (one file per policy)."""

    topology_id: str = "nobel-eu"
    load: float = 210.0
    launch_power_dbm: float = -4.0
    arrivals: int = 100_000
    warmup: int = 5_000
    seed: int | None = None
    policies: tuple[str, ...] = DEFAULT_POLICIES
    copropagating: bool = True
    query_fraction: float = 0.0
    output_dir: Path = DEFAULT_OUTPUT_DIR
    workers: int = 1


@dataclass(slots=True)
class _Collected:
    """Samples of one policy, accumulated as Python lists (see ``DatasetEnv``)."""

    lightpath: dict[str, list[Any]] = field(default_factory=dict)
    hop: dict[str, list[Any]] = field(default_factory=dict)
    copropagating: dict[str, list[Any]] = field(default_factory=dict)
    query: dict[str, list[Any]] = field(default_factory=dict)
    arrivals: int = 0
    accepted: int = 0
    query_requests: int = 0

    def add(self, table: dict[str, list[Any]], row: dict[str, Any]) -> None:
        for key, value in row.items():
            table.setdefault(key, []).append(value)


def watt_to_dbm(power_w: float) -> float:
    return 10.0 * math.log10(power_w * 1e3)


class DatasetEnv(OpticalEnv):
    """RMSA environment that records every accepted lightpath after the warm-up.

    The sample is built in :meth:`on_action_applied`, while the network state is
    the one the new lightpath sees at establishment. With ``query_fraction`` >
    0, :meth:`on_request_analysed` also records the QoT queries of a sampled
    fraction of the arrivals (drawn from a dedicated RNG, so the traffic is the
    same with and without queries).
    """

    def __init__(
        self,
        config,
        topology,
        *,
        warmup: int,
        copropagating: bool,
        query_fraction: float = 0.0,
    ) -> None:
        super().__init__(config, topology, episode_length=config.episode_length)
        if not 0.0 <= query_fraction <= 1.0:
            raise ValueError("query_fraction must be in [0, 1]")
        self.warmup = warmup
        self.capture_copropagating = copropagating
        self.query_fraction = float(query_fraction)
        self._query_rng = np.random.default_rng([int(config.seed or 0), 0x5159])
        self.collected = _Collected()
        self._modulation_index = {m.name: i for i, m in enumerate(config.modulations)}

    def on_request_analysed(self, analysis) -> None:
        if self.query_fraction <= 0.0 or analysis.request.request_index < self.warmup:
            return
        if self.query_fraction < 1.0 and self._query_rng.random() >= self.query_fraction:
            return
        self.collected.query_requests += 1
        self._record_queries(analysis)

    def _record_queries(self, analysis) -> None:
        """Every candidate path and format at its first-fit start slot."""
        request = analysis.request
        config = self.simulator.config
        gsnr_db = analysis.gsnr_db_by_start
        launch_power_dbm = analysis.launch_power_dbm
        for path_index, path in enumerate(analysis.paths):
            reversed_route = int(path.node_indices[0] != request.source_id)
            for offset, modulation_index in enumerate(analysis.modulation_indices):
                free_starts = np.flatnonzero(analysis.resource_valid_starts[path_index, offset])
                if free_starts.size == 0:
                    continue
                slot = int(free_starts[0])
                gsnr = float(gsnr_db[path_index, offset, slot])
                if math.isnan(gsnr):
                    continue
                modulation = config.modulations[modulation_index]
                self.collected.add(
                    self.collected.query,
                    {
                        "request_index": request.request_index,
                        "source": request.source_id,
                        "destination": request.destination_id,
                        "bit_rate": request.bit_rate,
                        "path_id": path.id,
                        "path_reversed": reversed_route,
                        "path_k": path.k,
                        "modulation_index": modulation_index,
                        "slot_start": slot,
                        "n_slots": int(analysis.required_slots_by_path_mod[path_index, offset]),
                        "launch_power_dbm": launch_power_dbm,
                        "gsnr_db": gsnr,
                        "gsnr_margin_db": gsnr - modulation.minimum_osnr,
                        "qot_feasible": int(analysis.qot_valid_starts[path_index, offset, slot]),
                    },
                )

    def on_action_applied(self, transition: StepTransition) -> None:
        if transition.request.request_index < self.warmup:
            return
        self.collected.arrivals += 1
        if not transition.accepted:
            return
        self.collected.accepted += 1
        self._record(transition.request.service_id)

    def _record(self, service_id: int) -> None:
        state = self.simulator.state
        assert state is not None
        service = state.active_services_by_id[service_id]
        request = service.request
        path = service.path
        modulation = service.modulation
        assert modulation is not None
        breakdown = self.simulator.qot_engine.service_noise_breakdown(state, service_id)
        index = len(self.collected.lightpath.get("request_index", []))
        links = self.simulator.topology.links
        n_spans = sum(len(links[link_id].spans) for link_id in path.link_ids)
        # The gym keeps one PathRecord per route, in one direction, for both
        # directions of a node pair; hops are stored from the source instead.
        reversed_route = path.node_indices[0] != request.source_id
        if reversed_route and path.node_indices[-1] != request.source_id:
            raise ValueError(f"route of service {service_id} does not start or end at its source")
        traversal = tuple(reversed(path.link_ids)) if reversed_route else tuple(path.link_ids)
        noise_by_link = {
            link_id: (float(ase), float(nli), float(sci), float(xci))
            for link_id, ase, nli, sci, xci in zip(
                breakdown.link_ids,
                breakdown.link_ase_nsr,
                breakdown.link_nli_nsr,
                breakdown.link_sci_nsr,
                breakdown.link_xci_nsr,
            )
        }
        self.collected.add(
            self.collected.lightpath,
            {
                "request_index": request.request_index,
                "arrival_time": request.arrival_time,
                "holding_time": request.holding_time,
                "source": request.source_id,
                "destination": request.destination_id,
                "bit_rate": request.bit_rate,
                "path_id": path.id,
                "path_reversed": int(reversed_route),
                "path_k": path.k,
                "hops": path.hops,
                "n_spans": n_spans,
                "route_length_km": path.length_km,
                "modulation_index": self._modulation_index[modulation.name],
                "slot_start": service.service_slot_start,
                "n_slots": service.service_num_slots,
                "center_frequency_hz": service.center_frequency,
                "bandwidth_hz": service.bandwidth,
                "launch_power_dbm": watt_to_dbm(service.launch_power),
                "gsnr_db": service.osnr,
                "snr_ase_db": service.ase,
                "snr_nli_db": service.nli,
                "gsnr_margin_db": service.osnr - modulation.minimum_osnr,
                "active_lightpaths": len(state.active_services_by_id),
            },
        )
        for hop, link_id in enumerate(traversal):
            others = state.link_active_service_ids[link_id] - {service_id}
            self.collected.add(
                self.collected.hop,
                {
                    "lightpath": index,
                    "hop": hop,
                    "link_id": link_id,
                    "link_ase_nsr": noise_by_link[link_id][0],
                    "link_nli_nsr": noise_by_link[link_id][1],
                    "link_sci_nsr": noise_by_link[link_id][2],
                    "link_xci_nsr": noise_by_link[link_id][3],
                    "link_occupancy": float(np.mean(state.slot_allocation[link_id] != -1)),
                    "link_copropagating": len(others),
                },
            )
            if not self.capture_copropagating:
                continue
            for other_id in sorted(others):
                other = state.active_services_by_id[other_id]
                assert other.modulation is not None
                self.collected.add(
                    self.collected.copropagating,
                    {
                        "lightpath": index,
                        "hop": hop,
                        "slot_start": other.service_slot_start,
                        "n_slots": other.service_num_slots,
                        "bit_rate": other.request.bit_rate,
                        "modulation_index": self._modulation_index[other.modulation.name],
                        "launch_power_dbm": watt_to_dbm(other.launch_power),
                    },
                )


def build_env(experiment: DatasetExperiment) -> DatasetEnv:
    """The ``jocn_benchmark`` preset with one continuous episode (warm-up + arrivals)."""
    overrides: dict[str, Any] = {
        "scenario_id": f"jocn_dataset_{experiment.topology_id}_load_{experiment.load:g}",
        "topology_id": experiment.topology_id,
        "episode_length": experiment.warmup + experiment.arrivals,
        "load": float(experiment.load),
        "launch_power_dbm": float(experiment.launch_power_dbm),
    }
    if experiment.seed is not None:
        overrides["seed"] = int(experiment.seed)
    config = build_scenario("jocn_benchmark", **overrides)
    topology = TopologyModel.from_file(
        resolve_topology(config.topology_id),
        topology_id=config.topology_id,
        k_paths=config.k_paths,
        max_span_length_km=config.max_span_length_km,
        default_attenuation_db_per_km=config.default_attenuation_db_per_km,
        default_noise_figure_db=config.default_noise_figure_db,
    )
    return DatasetEnv(
        config,
        topology,
        warmup=experiment.warmup,
        copropagating=experiment.copropagating,
        query_fraction=experiment.query_fraction,
    )


def collect(experiment: DatasetExperiment, policy: str) -> tuple[DatasetEnv, _Collected]:
    """Run one continuous stream with ``policy`` and return the collected samples."""
    env = build_env(experiment)
    _, info = env.reset(seed=env.simulator.config.seed)
    while True:
        action = select_policy_action(policy, env, info)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return env, env.collected


def _array(values: list[Any], dtype: Any) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def build_dataset(env: DatasetEnv, collected: _Collected, policy: str, attrs: dict[str, Any]):
    """Assemble the samples of one policy into a documented ``xarray.Dataset``."""
    import xarray as xr

    config = env.simulator.config
    topology = env.simulator.topology
    lp = collected.lightpath
    n = len(lp.get("request_index", []))
    lp_dim = ("lightpath",)

    def lp_var(name: str, dtype: Any, units: str, description: str):
        return xr.Variable(lp_dim, _array(lp.get(name, []), dtype), {"units": units, "description": description})

    data_vars = {
        "request_index": lp_var("request_index", np.int64, "1", "Arrival index of the request in the stream (warm-up included)."),
        "arrival_time": lp_var("arrival_time", np.float64, "time unit", "Arrival time of the request."),
        "holding_time": lp_var("holding_time", np.float64, "time unit", "Holding time of the lightpath."),
        "source": lp_var("source", np.int16, "1", "Source node (index into `node`)."),
        "destination": lp_var("destination", np.int16, "1", "Destination node (index into `node`)."),
        "bit_rate": lp_var("bit_rate", np.int16, "Gb/s", "Requested bit rate."),
        "path_id": lp_var("path_id", np.int32, "1", "Chosen route (index into `path`, the gym's PathRecord.id; shared by both directions of a node pair)."),
        "path_reversed": lp_var("path_reversed", np.int8, "1", "1 if the lightpath traverses `path_nodes`/`path_links` of `path_id` backwards (source is the path's last node)."),
        "path_k": lp_var("path_k", np.int8, "1", "Rank of the chosen route among the k shortest paths (0 = shortest)."),
        "hops": lp_var("hops", np.int8, "1", "Number of links of the route."),
        "n_spans": lp_var("n_spans", np.int16, "1", "Number of amplified fibre spans of the route."),
        "route_length_km": lp_var("route_length_km", np.float64, "km", "Length of the chosen route."),
        "modulation_index": lp_var("modulation_index", np.int8, "1", "Modulation format (index into `modulation`)."),
        "slot_start": lp_var("slot_start", np.int16, "1", "First frequency slot of the lightpath."),
        "n_slots": lp_var("n_slots", np.int16, "1", "Number of frequency slots (without guard band)."),
        "center_frequency_hz": lp_var("center_frequency_hz", np.float64, "Hz", "Centre frequency."),
        "bandwidth_hz": lp_var("bandwidth_hz", np.float64, "Hz", "Occupied bandwidth."),
        "launch_power_dbm": lp_var("launch_power_dbm", np.float32, "dBm", "Launch power per channel."),
        "gsnr_db": lp_var("gsnr_db", np.float64, "dB", "GSNR at establishment (ASE + NLI: SCI and the XCI of the lightpaths established on the route; closed-form EGN model)."),
        "snr_ase_db": lp_var("snr_ase_db", np.float64, "dB", "SNR considering only ASE noise."),
        "snr_nli_db": lp_var("snr_nli_db", np.float64, "dB", "SNR considering only nonlinear interference."),
        "gsnr_margin_db": lp_var("gsnr_margin_db", np.float64, "dB", "GSNR minus the threshold of the chosen modulation format."),
        "active_lightpaths": lp_var("active_lightpaths", np.int32, "1", "Lightpaths in the network at establishment, this one included."),
    }

    # Per-hop link information, padded to the longest route of the dataset.
    max_hops = int(max(lp.get("hops", [0]) or [0]))
    hop_rows = collected.hop
    hop_lp = _array(hop_rows.get("lightpath", []), np.int64)
    hop_pos = _array(hop_rows.get("hop", []), np.int64)

    def hop_var(name: str, dtype: Any, fill: Any, units: str, description: str):
        # Padding beyond the route: NaN for floats, -1 for integers (documented
        # sentinel, not a _FillValue, so integer columns stay integer on read).
        values = np.full((n, max_hops), fill, dtype=dtype)
        values[hop_lp, hop_pos] = _array(hop_rows.get(name, []), dtype)
        return xr.Variable(("lightpath", "hop"), values, {"units": units, "description": description})

    data_vars |= {
        "hop_link": hop_var("link_id", np.int16, -1, "1", "Link traversed at each hop, from the source (hop 0) to the destination (index into `link`); -1 beyond the route."),
        "hop_ase_nsr": hop_var("link_ase_nsr", np.float64, np.nan, "1", "ASE noise-to-signal ratio (linear) of the link."),
        "hop_nli_nsr": hop_var("link_nli_nsr", np.float64, np.nan, "1", "NLI noise-to-signal ratio (linear) of the link."),
        "hop_sci_nsr": hop_var("link_sci_nsr", np.float64, np.nan, "1", "Self-channel part of the NLI of the link (linear NSR)."),
        "hop_xci_nsr": hop_var("link_xci_nsr", np.float64, np.nan, "1", "Cross-channel part of the NLI of the link (linear NSR; hop_sci_nsr + hop_xci_nsr = hop_nli_nsr)."),
        "hop_occupancy": hop_var("link_occupancy", np.float32, np.nan, "1", "Fraction of occupied slots on the link at establishment."),
        "hop_copropagating": hop_var("link_copropagating", np.int16, -1, "1", "Other lightpaths on the link at establishment."),
    }

    # Co-propagating channels as a ragged array.
    cp = collected.copropagating

    def cp_var(name: str, dtype: Any, units: str, description: str):
        return xr.Variable(("copropagating",), _array(cp.get(name, []), dtype), {"units": units, "description": description})

    if env.capture_copropagating:
        data_vars |= {
            "copropagating_lightpath": cp_var("lightpath", np.int32, "1", "Sample (index along `lightpath`) the channel co-propagates with."),
            "copropagating_hop": cp_var("hop", np.int8, "1", "Hop of that sample on which the channel is present."),
            "copropagating_slot_start": cp_var("slot_start", np.int16, "1", "First frequency slot of the channel."),
            "copropagating_n_slots": cp_var("n_slots", np.int16, "1", "Number of frequency slots of the channel."),
            "copropagating_bit_rate": cp_var("bit_rate", np.int16, "Gb/s", "Bit rate of the channel."),
            "copropagating_modulation_index": cp_var("modulation_index", np.int8, "1", "Modulation format of the channel (index into `modulation`)."),
            "copropagating_launch_power_dbm": cp_var("launch_power_dbm", np.float32, "dBm", "Launch power of the channel."),
        }

    # QoT queries of the RMSA (first-fit slot of every candidate path and format).
    queries = collected.query

    def query_var(name: str, dtype: Any, units: str, description: str):
        return xr.Variable(("query",), _array(queries.get(name, []), dtype), {"units": units, "description": description})

    if env.query_fraction > 0.0:
        data_vars |= {
            "query_request_index": query_var("request_index", np.int64, "1", "Arrival index of the request (joins `request_index` of the lightpath table)."),
            "query_source": query_var("source", np.int16, "1", "Source node (index into `node`)."),
            "query_destination": query_var("destination", np.int16, "1", "Destination node (index into `node`)."),
            "query_bit_rate": query_var("bit_rate", np.int16, "Gb/s", "Requested bit rate."),
            "query_path_id": query_var("path_id", np.int32, "1", "Candidate route (index into `path`)."),
            "query_path_reversed": query_var("path_reversed", np.int8, "1", "1 if the route is traversed backwards (source is the path's last node)."),
            "query_path_k": query_var("path_k", np.int8, "1", "Rank of the route among the k shortest paths (0 = shortest)."),
            "query_modulation_index": query_var("modulation_index", np.int8, "1", "Candidate modulation format (index into `modulation`)."),
            "query_slot_start": query_var("slot_start", np.int16, "1", "First free (first-fit) start slot of the candidate."),
            "query_n_slots": query_var("n_slots", np.int16, "1", "Number of frequency slots of the candidate."),
            "query_launch_power_dbm": query_var("launch_power_dbm", np.float32, "dBm", "Launch power per channel."),
            "query_gsnr_db": query_var("gsnr_db", np.float64, "dB", "GSNR computed by the environment before the decision (float32 precision)."),
            "query_gsnr_margin_db": query_var("gsnr_margin_db", np.float64, "dB", "GSNR minus the threshold of the candidate modulation format."),
            "query_qot_feasible": query_var("qot_feasible", np.int8, "1", "1 if the candidate meets the GSNR threshold (plus the scenario margin)."),
        }

    # Static description of the network.
    links = topology.links
    spans = [(link.id, span) for link in links for span in link.spans]
    data_vars |= {
        "link_source": xr.Variable(("link",), np.array([link.source_index for link in links], np.int16), {"description": "Source node of the link."}),
        "link_target": xr.Variable(("link",), np.array([link.target_index for link in links], np.int16), {"description": "Target node of the link."}),
        "link_length_km": xr.Variable(("link",), np.array([link.length_km for link in links]), {"units": "km", "description": "Link length."}),
        "link_n_spans": xr.Variable(("link",), np.array([len(link.spans) for link in links], np.int16), {"description": "Number of spans."}),
        "span_link": xr.Variable(("span",), np.array([i for i, _ in spans], np.int16), {"description": "Link of the span (index into `link`)."}),
        "span_length_km": xr.Variable(("span",), np.array([s.length_km for _, s in spans]), {"units": "km", "description": "Fibre length."}),
        "span_attenuation_db_per_km": xr.Variable(("span",), np.array([s.attenuation_db_per_km for _, s in spans]), {"units": "dB/km", "description": "Fibre attenuation."}),
        "span_noise_figure_db": xr.Variable(("span",), np.array([s.noise_figure_db for _, s in spans]), {"units": "dB", "description": "Noise figure of the amplifier after the span."}),
        "span_input_loss_db": xr.Variable(("span",), np.array([s.input_loss_db for _, s in spans]), {"units": "dB", "description": "Lumped loss before the fibre."}),
        "span_output_loss_db": xr.Variable(("span",), np.array([s.output_loss_db for _, s in spans]), {"units": "dB", "description": "Lumped loss after the fibre."}),
        "modulation_spectral_efficiency": xr.Variable(("modulation",), np.array([m.spectral_efficiency for m in config.modulations], np.int8), {"units": "b/s/Hz/pol", "description": "Spectral efficiency."}),
        "modulation_gsnr_threshold_db": xr.Variable(("modulation",), np.array([m.minimum_osnr for m in config.modulations]), {"units": "dB", "description": "Minimum GSNR of the modulation format."}),
    }
    for extra in (_path_table(topology), _node_coordinates(topology, config)):
        clashes = set(extra) & set(data_vars)
        if clashes:
            raise ValueError(f"dataset variable names defined twice: {sorted(clashes)}")
        data_vars |= extra
    coords = {
        "modulation": ("modulation", [m.name for m in config.modulations]),
        "node": ("node", list(topology.node_names)),
    }
    dataset = xr.Dataset(data_vars, coords=coords)
    topology_file = Path(resolve_topology(config.topology_id))
    dataset.attrs = {
        "title": f"QoT dataset of the accepted lightpaths, heuristic {policy}",
        "policy": policy,
        "policy_paper_name": PAPER_NAMES.get(policy, policy),
        "frequency_start_hz": float(config.frequency_start),
        "frequency_slot_bandwidth_hz": float(config.frequency_slot_bandwidth),
        "num_spectrum_resources": int(config.num_spectrum_resources),
        "arrivals_recorded": collected.arrivals,
        "lightpaths_accepted": collected.accepted,
        "blocking_ratio": 1.0 - collected.accepted / collected.arrivals if collected.arrivals else float("nan"),
        "ragged_arrays": "copropagating_* rows belong to lightpath copropagating_lightpath at hop copropagating_hop",
        # QoT model settings (also in scenario_config_json), stated explicitly.
        "qot_constraint": str(config.qot_constraint),
        "qot_nli_include_interferers": int(
            config.measure_disruptions if config.nli_include_interferers is None else config.nli_include_interferers
        ),
        "qot_nli_interferer_psd": str(config.nli_interferer_psd),
        "qot_nli_modulation_correction": str(config.nli_modulation_correction),
        "qot_direction": "undirected: GSNR of the route in its canonical direction (lower node index first), for both directions",
        "qot_query_requests": collected.query_requests,
        # Everything needed to rebuild the gym objects (see ``load_topology``).
        "scenario_config_json": scenario_config_json(config),
        "topology_id": config.topology_id,
        "topology_file_name": topology_file.name,
        "topology_file_content": topology_file.read_text(encoding="utf-8", errors="replace"),
        "topology_k_paths": int(config.k_paths),
        "topology_max_span_length_km": float(config.max_span_length_km),
        "topology_default_attenuation_db_per_km": float(config.default_attenuation_db_per_km),
        "topology_default_noise_figure_db": float(config.default_noise_figure_db),
        "references": REFERENCES,
        **attrs,
    }
    return dataset


def _path_table(topology: TopologyModel) -> dict[str, Any]:
    """All k-shortest paths of the topology (``path`` dimension), padded with -1."""
    import xarray as xr

    paths = sorted(topology.paths, key=lambda record: record.id)
    if [record.id for record in paths] != list(range(len(paths))):
        raise ValueError("path ids are expected to be 0..n-1")
    max_hops = max(record.hops for record in paths)
    nodes = np.full((len(paths), max_hops + 1), -1, dtype=np.int16)
    links = np.full((len(paths), max_hops), -1, dtype=np.int16)
    for record in paths:
        nodes[record.id, : len(record.node_indices)] = record.node_indices
        links[record.id, : len(record.link_ids)] = record.link_ids
    return {
        "path_source": xr.Variable(("path",), nodes[:, 0], {"description": "First node of the path."}),
        "path_destination": xr.Variable(("path",), np.array([r.node_indices[-1] for r in paths], np.int16), {"description": "Last node of the path."}),
        "path_rank": xr.Variable(("path",), np.array([r.k for r in paths], np.int8), {"description": "Rank among the k shortest paths of the node pair (0 = shortest)."}),
        "path_hops": xr.Variable(("path",), np.array([r.hops for r in paths], np.int8), {"description": "Number of links."}),
        "path_length_km": xr.Variable(("path",), np.array([r.length_km for r in paths]), {"units": "km", "description": "Path length."}),
        "path_nodes": xr.Variable(("path", "path_node_position"), nodes, {"description": "Nodes of the path in order (index into `node`); -1 beyond the path."}),
        "path_links": xr.Variable(("path", "path_hop"), links, {"description": "Links of the path in order (index into `link`); -1 beyond the path."}),
    }


def _node_coordinates(topology: TopologyModel, config) -> dict[str, Any]:
    """Node coordinates from an SNDlib XML topology (NaN for other formats)."""
    import xml.etree.ElementTree as ET

    import xarray as xr

    x = np.full(len(topology.node_names), np.nan)
    y = np.full(len(topology.node_names), np.nan)
    coordinates_type = "unknown"
    topology_file = Path(resolve_topology(config.topology_id))
    if topology_file.suffix == ".xml":
        root = ET.parse(topology_file).getroot()
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "nodes":
                coordinates_type = element.get("coordinatesType", coordinates_type)
            if tag != "node" or element.get("id") not in topology.node_index_by_name:
                continue
            index = topology.node_index_by_name[element.get("id")]
            for child in element.iter():
                name = child.tag.rsplit("}", 1)[-1]
                if name in ("x", "y") and child.text:
                    (x if name == "x" else y)[index] = float(child.text)
    units = "degree" if coordinates_type == "geographical" else "1"
    return {
        "node_x": xr.Variable(("node",), x, {"units": units, "description": f"Node x coordinate ({coordinates_type}; longitude if geographical)."}),
        "node_y": xr.Variable(("node",), y, {"units": units, "description": f"Node y coordinate ({coordinates_type}; latitude if geographical)."}),
    }


def scenario_config_json(config) -> str:
    """The full ``ScenarioConfig`` as JSON (enums as values, paths as strings)."""
    from dataclasses import asdict as dataclass_asdict
    from enum import Enum

    def default(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        return repr(value)

    return json.dumps(dataclass_asdict(config), default=default, sort_keys=True)


def load_topology(dataset) -> TopologyModel:
    """Rebuild the gym ``TopologyModel`` from a dataset alone (embedded topology file)."""
    import tempfile

    attrs = dataset.attrs
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / str(attrs["topology_file_name"])
        path.write_text(str(attrs["topology_file_content"]), encoding="utf-8")
        return TopologyModel.from_file(
            path,
            topology_id=str(attrs["topology_id"]),
            k_paths=int(attrs["topology_k_paths"]),
            max_span_length_km=float(attrs["topology_max_span_length_km"]),
            default_attenuation_db_per_km=float(attrs["topology_default_attenuation_db_per_km"]),
            default_noise_figure_db=float(attrs["topology_default_noise_figure_db"]),
        )


def dataset_file_name(experiment: DatasetExperiment, policy: str, seed: int, stamp: str) -> str:
    """Self-describing file name: one file per run and per heuristic."""
    return f"jocn2024_qot_{experiment.topology_id}_{experiment.load:g}erl_{policy}_seed{seed}_{stamp}.nc"


def summarize(dataset, policy: str) -> dict[str, Any]:
    """Blocking, GSNR and modulation-format statistics of one dataset (Fig. 5)."""
    gsnr = dataset["gsnr_db"].values
    names = dataset["modulation"].values
    counts = np.bincount(dataset["modulation_index"].values.astype(np.int64), minlength=len(names))
    share = counts / max(1, counts.sum())
    row: dict[str, Any] = {
        "policy": policy,
        "policy_paper_name": PAPER_NAMES.get(policy, policy),
        "arrivals": int(dataset.attrs["arrivals_recorded"]),
        "accepted": int(dataset.attrs["lightpaths_accepted"]),
        "blocking_ratio": float(dataset.attrs["blocking_ratio"]),
        "gsnr_mean_db": float(np.mean(gsnr)) if gsnr.size else float("nan"),
        "gsnr_median_db": float(np.median(gsnr)) if gsnr.size else float("nan"),
        f"share_gsnr_above_{HIGH_GSNR_DB:g}db": float(np.mean(gsnr > HIGH_GSNR_DB)) if gsnr.size else float("nan"),
    }
    row |= {f"share_{name}": float(value) for name, value in zip(names, share)}
    return row


def _provenance() -> dict[str, Any]:
    try:
        gym_version = version("optical-networking-gym")
    except PackageNotFoundError:
        gym_version = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "gym_version": gym_version,
        "gym_git_commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": f"examples/{FAMILY}/{SCRIPT_NAME}.py",
    }


def run_policy(experiment: DatasetExperiment, policy: str, run_dir: Path) -> dict[str, Any]:
    """Collect, write the dataset of ``policy`` and return its summary row."""
    env, collected = collect(experiment, policy)
    # netCDF attributes hold strings and numbers only (no bool or None).
    experiment_attrs = {
        f"experiment_{key}": int(value) if isinstance(value, bool) else value
        for key, value in asdict(experiment).items()
        if key not in ("output_dir", "workers", "policies", "seed")
    }
    experiment_attrs["experiment_seed"] = int(env.simulator.config.seed)
    run_attrs = {"run_id": run_dir.name} | experiment_attrs | _provenance()
    dataset = build_dataset(env, collected, policy, run_attrs)
    path = run_dir / dataset_file_name(experiment, policy, int(env.simulator.config.seed), run_dir.name)
    encoding = {name: {"zlib": True, "complevel": 4} for name in dataset.data_vars}
    dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)
    return summarize(dataset, policy) | {"file": path.name}


def run(experiment: DatasetExperiment, *, now: datetime | None = None) -> Path:
    """Generate one dataset per policy; return the run directory."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(experiment.output_dir) / FAMILY / SCRIPT_NAME / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(experiment.workers, len(experiment.policies)))
    if workers == 1:
        rows = [run_policy(experiment, policy, run_dir) for policy in experiment.policies]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_policy, experiment, policy, run_dir) for policy in experiment.policies]
            rows = [future.result() for future in futures]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "family": FAMILY,
        "script": SCRIPT_NAME,
        "experiment": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(experiment).items()},
        "files": [row["file"] for row in rows],
    } | _provenance()
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JOCN 2024 QoT dataset generation (Fig. 5).")
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--load", type=float, default=210.0, help="Offered load in Erlang.")
    parser.add_argument("--launch-power-dbm", type=float, default=-4.0)
    parser.add_argument("--arrivals", type=int, default=100_000, help="Recorded arrivals per policy.")
    parser.add_argument("--warmup", type=int, default=5_000, help="Arrivals before recording starts.")
    parser.add_argument("--seed", type=int, default=None, help="Traffic seed (default: preset seed).")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    parser.add_argument("--no-copropagating", action="store_true", help="Skip the co-propagating table.")
    parser.add_argument(
        "--query-fraction",
        type=float,
        default=0.0,
        help="Fraction of the recorded arrivals whose QoT queries are saved (0 = no query table).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1, help="Policies generated in parallel.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = run(
        DatasetExperiment(
            topology_id=args.topology_id,
            load=args.load,
            launch_power_dbm=args.launch_power_dbm,
            arrivals=args.arrivals,
            warmup=args.warmup,
            seed=args.seed,
            policies=tuple(args.policies),
            copropagating=not args.no_copropagating,
            query_fraction=args.query_fraction,
            output_dir=args.output_dir,
            workers=args.workers,
        )
    )
    print(f"JOCN datasets saved to: {run_dir}")


REFERENCES = (
    "[Natalino_2024_OpticalNetworkingGym] C. Natalino, T. Magalhaes, F. Arpanaei et al., "
    '"Optical Networking Gym: An Open-Source Toolkit for Resource Assignment Problems in '
    'Optical Networks," J. Opt. Commun. Netw., vol. 16, no. 12, pp. G40-G51, Dec. 2024, '
    "doi: 10.1364/JOCN.532850."
)

if __name__ == "__main__":
    main()

# References
# [Natalino_2024_OpticalNetworkingGym] C. Natalino, T. Magalhaes, F. Arpanaei et al.,
#     "Optical Networking Gym: An Open-Source Toolkit for Resource Assignment Problems
#     in Optical Networks," J. Opt. Commun. Netw., vol. 16, no. 12, pp. G40-G51,
#     Dec. 2024, doi: 10.1364/JOCN.532850.
