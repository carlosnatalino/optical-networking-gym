from __future__ import annotations

import cProfile
from collections.abc import Iterable
import io
from pathlib import Path
import pstats
import statistics
import time
from typing import SupportsFloat, SupportsInt, cast

import numpy as np

from optical_networking_gym.contracts.enums import TrafficMode
from optical_networking_gym.contracts.modulation import Modulation
from optical_networking_gym.contracts.traffic import TrafficRecord, TrafficTable
from optical_networking_gym.defaults import BUILTIN_TOPOLOGY_DIR, resolve_topology
from optical_networking_gym.network.topology import TopologyModel
from optical_networking_gym.config.scenario import ScenarioConfig
from optical_networking_gym.runtime.simulator import Simulator
from optical_networking_gym.runtime.traffic_model import TrafficModel


TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR


def _as_float(value: object) -> float:
    return float(cast(SupportsFloat, value))


def _as_int(value: object) -> int:
    return int(cast(SupportsInt, value))


def _durations_summary_us(durations_ns: list[int]) -> tuple[float, float]:
    if not durations_ns:
        return 0.0, 0.0
    durations_us = [duration / 1_000.0 for duration in durations_ns]
    return float(statistics.fmean(durations_us)), float(np.percentile(durations_us, 95))


def _topology_path(topology_id: str) -> Path:
    return resolve_topology(topology_id)


def _v2_modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def _build_topology(topology_id: str, *, k_paths: int) -> TopologyModel:
    return TopologyModel.from_file(_topology_path(topology_id), topology_id=topology_id, k_paths=k_paths)


def _build_dynamic_config(
    *,
    topology_id: str,
    k_paths: int,
    num_spectrum_resources: int,
    seed: int,
    load: float,
    mean_holding_time: float,
) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=f"{topology_id}_integrated_dynamic",
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        traffic_mode=TrafficMode.DYNAMIC,
        traffic_source={
            "bit_rates": (40,),
            "bit_rate_probabilities": (1.0,),
            "load": load,
            "mean_holding_time": mean_holding_time,
        },
        modulations=_v2_modulations(),
        modulations_to_consider=2,
        seed=seed,
    )


def _build_static_config(
    *,
    topology_id: str,
    k_paths: int,
    num_spectrum_resources: int,
    seed: int,
    table: TrafficTable,
    records: tuple[TrafficRecord, ...],
) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=f"{topology_id}_integrated_static",
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        traffic_mode=TrafficMode.STATIC,
        traffic_source={"table": table, "records": records},
        modulations=_v2_modulations(),
        modulations_to_consider=2,
        seed=seed,
    )


def _capture_dynamic_table(
    *,
    topology_id: str,
    k_paths: int,
    num_spectrum_resources: int,
    request_count: int,
    seed: int,
    load: float,
    mean_holding_time: float,
) -> tuple[TrafficTable, tuple[TrafficRecord, ...]]:
    topology = _build_topology(topology_id, k_paths=k_paths)
    config = _build_dynamic_config(
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
    )
    model = TrafficModel(config, topology, capture_table=True)
    for _ in range(request_count):
        model.next_request()
    return model.export_table()


def _first_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask[:-1])
    return int(valid[0]) if valid.size > 0 else int(mask.shape[0] - 1)


def _run_v2_episode(
    *,
    static_config: ScenarioConfig,
    topology: TopologyModel,
    records: tuple[TrafficRecord, ...],
) -> dict[str, object]:
    simulator = Simulator(static_config, topology, episode_length=len(records))

    reset_start = time.perf_counter_ns()
    _, info = simulator.reset(seed=static_config.seed)
    reset_elapsed = time.perf_counter_ns() - reset_start

    step_durations: list[int] = []
    statuses: list[str] = []
    osnrs: list[float] = []
    active_counts: list[int] = []
    masks: list[np.ndarray] = []
    slot_snapshots: list[np.ndarray] = []

    for _ in records:
        if "mask" not in info:
            raise RuntimeError("v2 reset/step must return the next mask")
        current_mask = np.asarray(info["mask"], dtype=np.uint8)
        masks.append(current_mask.copy())
        action = _first_valid_action(current_mask)

        step_start = time.perf_counter_ns()
        _, _, _, _, info = simulator.step(action)
        step_elapsed = time.perf_counter_ns() - step_start
        step_durations.append(step_elapsed)

        statuses.append(str(info["status"]))
        osnrs.append(_as_float(info.get("osnr", 0.0)))
        if simulator.state is not None:
            active_counts.append(len(simulator.state.active_services_by_id))
            slot_snapshots.append((simulator.state.slot_allocation == -1).astype(np.int32).copy())
        else:
            active_counts.append(0)
            slot_snapshots.append(np.empty((0, 0), dtype=np.int32))

    step_mean_us, step_p95_us = _durations_summary_us(step_durations)
    return {
        "reset_ns": reset_elapsed,
        "step_durations_ns": step_durations,
        "step_mean_us": step_mean_us,
        "step_p95_us": step_p95_us,
        "statuses": tuple(statuses),
        "osnrs": tuple(osnrs),
        "active_counts": tuple(active_counts),
        "masks": tuple(masks),
        "slot_snapshots": tuple(slot_snapshots),
        "episode_services_accepted": info["episode_services_accepted"],
    }


def benchmark_simulator_episode(
    *,
    topology_id: str = "ring_4",
    k_paths: int = 2,
    num_spectrum_resources: int = 24,
    request_count: int = 32,
    seed: int = 7,
    load: float = 10.0,
    mean_holding_time: float = 100.0,
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, object]:
    table, records = _capture_dynamic_table(
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        request_count=request_count,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
    )
    topology = _build_topology(topology_id, k_paths=k_paths)
    static_config = _build_static_config(
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        seed=seed,
        table=table,
        records=records,
    )

    reset_durations: list[int] = []
    step_durations: list[int] = []
    episode_durations: list[int] = []
    accepted = 0

    for repeat_index in range(repeats + warmup):
        start_ns = time.perf_counter_ns()
        result = _run_v2_episode(static_config=static_config, topology=topology, records=records)
        episode_elapsed = time.perf_counter_ns() - start_ns
        if repeat_index < warmup:
            continue
        reset_durations.append(_as_int(result["reset_ns"]))
        step_durations.extend(
            _as_int(duration) for duration in cast(Iterable[object], result["step_durations_ns"])
        )
        episode_durations.append(episode_elapsed)
        accepted = _as_int(result["episode_services_accepted"])

    reset_mean_us, reset_p95_us = _durations_summary_us(reset_durations)
    step_mean_us, step_p95_us = _durations_summary_us(step_durations)
    episode_mean_us, episode_p95_us = _durations_summary_us(episode_durations)
    return {
        "component": "SimulatorEpisode",
        "topology_id": topology_id,
        "request_count": request_count,
        "repeats": repeats,
        "warmup": warmup,
        "reset_mean_us": reset_mean_us,
        "reset_p95_us": reset_p95_us,
        "step_mean_us": step_mean_us,
        "step_p95_us": step_p95_us,
        "episode_mean_us": episode_mean_us,
        "episode_p95_us": episode_p95_us,
        "episode_services_accepted": accepted,
    }


def profile_simulator_episode(
    *,
    topology_id: str = "ring_4",
    k_paths: int = 2,
    num_spectrum_resources: int = 24,
    request_count: int = 32,
    seed: int = 7,
    load: float = 10.0,
    mean_holding_time: float = 100.0,
    top_n: int = 15,
) -> dict[str, object]:
    table, records = _capture_dynamic_table(
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        request_count=request_count,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
    )
    topology = _build_topology(topology_id, k_paths=k_paths)
    static_config = _build_static_config(
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        seed=seed,
        table=table,
        records=records,
    )

    def _run_episode() -> None:
        simulator = Simulator(static_config, topology, episode_length=len(records))
        _, info = simulator.reset(seed=seed)
        for _ in records:
            _, _, terminated, _, info = simulator.step(_first_valid_action(np.asarray(info["mask"], dtype=np.uint8)))
            if terminated:
                break

    profiler = cProfile.Profile()
    start_ns = time.perf_counter_ns()
    profiler.enable()
    _run_episode()
    profiler.disable()
    elapsed_ns = time.perf_counter_ns() - start_ns

    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    # pstats.Stats populates `stats` (and honours `stream`) dynamically; the
    # attributes are not declared in typeshed, hence getattr/setattr.
    stats_table = cast(
        "dict[tuple[str, int, str], tuple[int, int, float, float, object]]",
        getattr(stats, "stats"),
    )
    entries: list[dict[str, object]] = []
    for function_descriptor, function_stats in stats_table.items():
        primitive_calls, total_calls, total_time, cumulative_time, _ = function_stats
        file_name, line_number, function_name = function_descriptor
        entries.append(
            {
                "function": function_name,
                "file": str(file_name),
                "line": int(line_number),
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "total_time_s": float(total_time),
                "cumulative_time_s": float(cumulative_time),
            }
        )
    entries.sort(key=lambda entry: _as_float(entry["cumulative_time_s"]), reverse=True)

    stream = io.StringIO()
    setattr(stats, "stream", stream)
    stats.print_stats(top_n)

    return {
        "component": "SimulatorProfile",
        "topology_id": topology_id,
        "request_count": request_count,
        "elapsed_ms": elapsed_ns / 1_000_000.0,
        "top_entries": entries[:top_n],
        "rendered_stats": stream.getvalue(),
    }


__all__ = [
    "benchmark_simulator_episode",
    "profile_simulator_episode",
]
