from __future__ import annotations

import cProfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
import heapq
import io
import os
from pathlib import Path
import pstats
import statistics
import time

import numpy as np

import optical_networking_gym.core.osnr as legacy_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv, Service
from optical_networking_gym.topology import Modulation as LegacyModulation, get_topology

from optical_networking_gym_v2.contracts.enums import TrafficMode
from optical_networking_gym_v2.contracts.modulation import Modulation
from optical_networking_gym_v2.contracts.traffic import TrafficRecord, TrafficTable
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.runtime.simulator import Simulator
from optical_networking_gym_v2.runtime.traffic_model import TrafficModel


PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOPOLOGY_DIR = PROJECT_ROOT / "examples" / "topologies"


def _durations_summary_us(durations_ns: list[int]) -> tuple[float, float]:
    if not durations_ns:
        return 0.0, 0.0
    durations_us = [duration / 1_000.0 for duration in durations_ns]
    return float(statistics.fmean(durations_us)), float(np.percentile(durations_us, 95))


@contextmanager
def _suppress_legacy_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def _topology_path(topology_id: str) -> Path:
    for suffix in (".txt", ".xml"):
        candidate = TOPOLOGY_DIR / f"{topology_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve topology file for {topology_id!r}")


def _v2_modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def _legacy_modulations() -> tuple[LegacyModulation, ...]:
    return (
        LegacyModulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        LegacyModulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
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


def _build_legacy_env(
    *,
    topology_id: str,
    num_spectrum_resources: int,
    k_paths: int,
    seed: int,
    load: float,
    mean_holding_time: float,
    episode_length: int,
) -> QRMSAEnv:
    topology = get_topology(
        str(_topology_path(topology_id)),
        topology_name=topology_id,
        modulations=_legacy_modulations(),
        max_span_length=100.0,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=k_paths,
    )
    return QRMSAEnv(
        topology=topology,
        num_spectrum_resources=num_spectrum_resources,
        episode_length=episode_length,
        load=load,
        mean_service_holding_time=mean_holding_time,
        bit_rate_selection="discrete",
        bit_rates=(40,),
        bit_rate_probabilities=(1.0,),
        bandwidth=num_spectrum_resources * 12.5e9,
        seed=seed,
        reset=True,
        gen_observation=False,
        measure_disruptions=False,
        k_paths=k_paths,
    )


def _build_legacy_service(request: TrafficRecord, legacy_env: QRMSAEnv) -> Service:
    source_name = legacy_env.topology.graph["node_indices"][request.source_id]
    destination_name = legacy_env.topology.graph["node_indices"][request.destination_id]
    return Service(
        service_id=request.service_id,
        source=source_name,
        source_id=request.source_id,
        destination=destination_name,
        destination_id=str(request.destination_id),
        arrival_time=request.arrival_time,
        holding_time=request.holding_time,
        bit_rate=request.bit_rate,
    )


def _window_mask(available_slots: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0 or window_size > available_slots.shape[0]:
        return np.zeros((0,), dtype=np.uint8)
    slot_view = np.asarray(available_slots, dtype=np.int8)
    kernel = np.ones(window_size, dtype=np.int8)
    window_sums = np.convolve(slot_view, kernel, mode="valid")
    return (window_sums == window_size).astype(np.uint8)


def _clear_legacy_observation_caches(legacy_env: QRMSAEnv) -> None:
    for attribute_name in (
        "available_slots_cache",
        "available_slots_signature_cache",
        "block_info_cache",
        "osnr_matrix_cache",
        "fragmentation_cache",
        "slot_window_cache",
    ):
        cache = getattr(legacy_env, attribute_name, None)
        if cache is not None:
            cache.clear()


def _legacy_available_slots(legacy_env: QRMSAEnv, route: object) -> np.ndarray:
    node_list = route.node_list
    link_indices = [
        legacy_env.topology[node_list[index]][node_list[index + 1]]["index"]
        for index in range(len(node_list) - 1)
    ]
    matrix = np.asarray(legacy_env.topology.graph["available_slots"][link_indices, :], dtype=np.int32)
    if matrix.shape[0] == 1:
        return matrix[0].copy()
    return np.min(matrix, axis=0).astype(np.int32, copy=False)


def _build_legacy_action_mask(legacy_env: QRMSAEnv) -> np.ndarray:
    current_service = legacy_env.current_service
    if current_service is None:
        raise ValueError("legacy_env.current_service is required")

    legacy_env.get_max_modulation_index()
    paths_info: list[tuple[object, np.ndarray]] = []
    for path_index, route in enumerate(
        legacy_env.k_shortest_paths[current_service.source, current_service.destination]
    ):
        if path_index >= legacy_env.k_paths:
            break
        paths_info.append((route, _legacy_available_slots(legacy_env, route)))

    total_actions = legacy_env.k_paths * legacy_env.modulations_to_consider * legacy_env.num_spectrum_resources
    action_mask = np.zeros(total_actions + 1, dtype=np.uint8)
    path_modulations_cache: dict[int, list[object]] = {}
    path_window_masks: dict[tuple[int, int], np.ndarray] = {}

    for action_index in range(total_actions):
        path_index = action_index // (
            legacy_env.modulations_to_consider * legacy_env.num_spectrum_resources
        )
        modulation_and_slot = action_index % (
            legacy_env.modulations_to_consider * legacy_env.num_spectrum_resources
        )
        modulation_offset = modulation_and_slot // legacy_env.num_spectrum_resources
        initial_slot = modulation_and_slot % legacy_env.num_spectrum_resources

        if path_index >= len(paths_info):
            continue
        route, available_slots = paths_info[path_index]
        if available_slots[initial_slot] == 0:
            continue

        if path_index not in path_modulations_cache:
            start_index = max(
                0,
                legacy_env.max_modulation_idx - (legacy_env.modulations_to_consider - 1),
            )
            path_modulations_cache[path_index] = list(
                reversed(
                    legacy_env.modulations[start_index : legacy_env.max_modulation_idx + 1][
                        : legacy_env.modulations_to_consider
                    ]
                )
            )
        modulation_list = path_modulations_cache[path_index]
        if modulation_offset >= len(modulation_list):
            continue
        modulation = modulation_list[modulation_offset]
        number_slots = legacy_env.get_number_slots(current_service, modulation)

        if initial_slot + number_slots > legacy_env.num_spectrum_resources:
            continue

        base_key = (path_index, number_slots)
        if base_key not in path_window_masks:
            path_window_masks[base_key] = _window_mask(available_slots, number_slots)
        base_mask = path_window_masks[base_key]
        if initial_slot >= base_mask.shape[0] or base_mask[initial_slot] == 0:
            continue

        guard_needed = (initial_slot + number_slots) < legacy_env.num_spectrum_resources
        if guard_needed:
            guard_window = number_slots + 1
            guard_key = (path_index, guard_window)
            if guard_key not in path_window_masks:
                path_window_masks[guard_key] = _window_mask(available_slots, guard_window)
            guard_mask = path_window_masks[guard_key]
            if initial_slot >= guard_mask.shape[0] or guard_mask[initial_slot] == 0:
                continue

        current_service.path = route
        current_service.initial_slot = initial_slot
        current_service.number_slots = number_slots
        current_service.center_frequency = (
            legacy_env.frequency_start
            + legacy_env.frequency_slot_bandwidth * (initial_slot + number_slots / 2)
        )
        current_service.bandwidth = legacy_env.frequency_slot_bandwidth * number_slots
        current_service.launch_power = legacy_env.launch_power

        if legacy_env.qot_constraint == "DIST":
            qot_acceptable = route.length <= modulation.maximum_length
        else:
            osnr, _, _ = legacy_osnr.calculate_osnr(legacy_env, current_service, legacy_env.qot_constraint)
            qot_acceptable = osnr >= modulation.minimum_osnr + legacy_env.margin

        current_service.path = None
        current_service.initial_slot = -1
        current_service.number_slots = 0
        current_service.center_frequency = 0.0
        current_service.bandwidth = 0.0
        current_service.launch_power = 0.0

        if qot_acceptable:
            action_mask[action_index] = 1

    action_mask[-1] = 1
    return action_mask


def _legacy_prepare_request(
    legacy_env: QRMSAEnv,
    release_events: list[tuple[float, int, object]],
    record: TrafficRecord,
) -> None:
    while release_events:
        release_time, _, service = heapq.heappop(release_events)
        if release_time <= record.arrival_time:
            legacy_env.current_time = release_time
            legacy_env._release_path(service)
            continue
        heapq.heappush(release_events, (release_time, service.service_id, service))
        break

    _clear_legacy_observation_caches(legacy_env)
    legacy_env.current_time = record.arrival_time
    legacy_env.current_service = _build_legacy_service(record, legacy_env)


def _legacy_step_current(
    legacy_env: QRMSAEnv,
    release_events: list[tuple[float, int, object]],
    counters: dict[str, int],
    action: int,
) -> dict[str, float | str | bool]:
    service = legacy_env.current_service
    if service is None:
        raise RuntimeError("legacy current service is required")

    counters["processed"] += 1

    status = "rejected_by_agent"
    osnr = 0.0
    if action == legacy_env.action_space.n - 1:
        service.accepted = False
        service.blocked_due_to_resources = False
        service.blocked_due_to_osnr = False
        legacy_env.bl_reject += 1
        return {"status": status, "accepted": False, "osnr": osnr}

    route_index, modulation_index, initial_slot = legacy_env.encoded_decimal_to_array(
        action,
        [legacy_env.k_paths, legacy_env.modulations_to_consider, legacy_env.num_spectrum_resources],
    )
    modulation = legacy_env.modulations[modulation_index]
    path = legacy_env.k_shortest_paths[service.source, service.destination][route_index]
    number_slots = legacy_env.get_number_slots(service=service, modulation=modulation)

    if not legacy_env.is_path_free(path=path, initial_slot=initial_slot, number_slots=number_slots):
        service.accepted = False
        service.blocked_due_to_resources = True
        service.blocked_due_to_osnr = False
        legacy_env.bl_resource += 1
        return {"status": "blocked_resources", "accepted": False, "osnr": osnr}

    service.path = path
    service.initial_slot = initial_slot
    service.number_slots = number_slots
    service.center_frequency = (
        legacy_env.frequency_start
        + (legacy_env.frequency_slot_bandwidth * initial_slot)
        + (legacy_env.frequency_slot_bandwidth * (number_slots / 2.0))
    )
    service.bandwidth = legacy_env.frequency_slot_bandwidth * number_slots
    service.launch_power = legacy_env.launch_power

    if legacy_env.qot_constraint == "DIST":
        qot_acceptable = path.length <= modulation.maximum_length
    else:
        osnr, ase, nli = legacy_osnr.calculate_osnr(legacy_env, service, legacy_env.qot_constraint)
        qot_acceptable = osnr >= modulation.minimum_osnr + legacy_env.margin

    if not qot_acceptable:
        service.accepted = False
        service.blocked_due_to_resources = False
        service.blocked_due_to_osnr = True
        service.path = None
        service.initial_slot = -1
        service.number_slots = 0
        service.center_frequency = 0.0
        service.bandwidth = 0.0
        service.launch_power = 0.0
        legacy_env.bl_osnr += 1
        return {"status": "blocked_qot", "accepted": False, "osnr": osnr}

    service.accepted = True
    service.blocked_due_to_resources = False
    service.blocked_due_to_osnr = False
    service.OSNR = osnr
    service.ASE = ase
    service.NLI = nli
    service.current_modulation = modulation
    legacy_env._provision_path(path, initial_slot, number_slots)
    _clear_legacy_observation_caches(legacy_env)
    counters["accepted"] += 1
    heapq.heappush(
        release_events,
        (service.arrival_time + service.holding_time, service.service_id, service),
    )
    return {"status": "accepted", "accepted": True, "osnr": osnr}


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
        osnrs.append(float(info.get("osnr", 0.0)))
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


def _run_legacy_replay_episode(
    *,
    topology_id: str,
    num_spectrum_resources: int,
    k_paths: int,
    seed: int,
    load: float,
    mean_holding_time: float,
    records: tuple[TrafficRecord, ...],
) -> dict[str, object]:
    with _suppress_legacy_output():
        env_start = time.perf_counter_ns()
        legacy_env = _build_legacy_env(
            topology_id=topology_id,
            num_spectrum_resources=num_spectrum_resources,
            k_paths=k_paths,
            seed=seed,
            load=load,
            mean_holding_time=mean_holding_time,
            episode_length=len(records),
        )
        setup_elapsed = time.perf_counter_ns() - env_start

        legacy_env.current_time = 0.0
        legacy_env.current_service = None
        release_events: list[tuple[float, int, object]] = []
        counters = {"accepted": 0, "processed": 0}

        apply_durations: list[int] = []
        request_cycle_durations: list[int] = []
        statuses: list[str] = []
        osnrs: list[float] = []
        active_counts: list[int] = []
        masks: list[np.ndarray] = []
        slot_snapshots: list[np.ndarray] = []

        for record in records:
            cycle_start = time.perf_counter_ns()
            _legacy_prepare_request(legacy_env, release_events, record)
            current_mask = _build_legacy_action_mask(legacy_env)
            masks.append(current_mask.copy())
            action = _first_valid_action(current_mask)

            apply_start = time.perf_counter_ns()
            outcome = _legacy_step_current(legacy_env, release_events, counters, action)
            apply_elapsed = time.perf_counter_ns() - apply_start
            cycle_elapsed = time.perf_counter_ns() - cycle_start
            apply_durations.append(apply_elapsed)
            request_cycle_durations.append(cycle_elapsed)

            statuses.append(str(outcome["status"]))
            osnrs.append(float(outcome["osnr"]))
            active_counts.append(len(legacy_env.topology.graph["running_services"]))
            slot_snapshots.append(np.asarray(legacy_env.topology.graph["available_slots"], dtype=np.int32).copy())

    apply_mean_us, apply_p95_us = _durations_summary_us(apply_durations)
    request_cycle_mean_us, request_cycle_p95_us = _durations_summary_us(request_cycle_durations)
    return {
        "setup_ns": setup_elapsed,
        "apply_durations_ns": apply_durations,
        "request_cycle_durations_ns": request_cycle_durations,
        "apply_mean_us": apply_mean_us,
        "apply_p95_us": apply_p95_us,
        "request_cycle_mean_us": request_cycle_mean_us,
        "request_cycle_p95_us": request_cycle_p95_us,
        "statuses": tuple(statuses),
        "osnrs": tuple(osnrs),
        "active_counts": tuple(active_counts),
        "masks": tuple(masks),
        "slot_snapshots": tuple(slot_snapshots),
        "episode_services_accepted": counters["accepted"],
    }


def compare_simulator_episode_with_legacy(
    *,
    topology_id: str = "ring_4",
    k_paths: int = 2,
    num_spectrum_resources: int = 24,
    request_count: int = 8,
    seed: int = 7,
    load: float = 10.0,
    mean_holding_time: float = 100.0,
    osnr_tolerance: float = 1e-9,
    include_records: bool = False,
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
    v2 = _run_v2_episode(static_config=static_config, topology=topology, records=records)
    legacy = _run_legacy_replay_episode(
        topology_id=topology_id,
        num_spectrum_resources=num_spectrum_resources,
        k_paths=k_paths,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
        records=records,
    )

    mask_matches = all(
        np.array_equal(v2_mask, legacy_mask)
        for v2_mask, legacy_mask in zip(v2["masks"], legacy["masks"], strict=True)
    )
    slot_matches = np.array_equal(v2["slot_snapshots"][-1], legacy["slot_snapshots"][-1])
    active_count_matches = v2["active_counts"][-1] == legacy["active_counts"][-1]
    status_matches = tuple(v2["statuses"]) == tuple(legacy["statuses"])

    osnr_matches = True
    for status, v2_osnr, legacy_osnr_value in zip(
        v2["statuses"],
        v2["osnrs"],
        legacy["osnrs"],
        strict=True,
    ):
        if status != "accepted":
            continue
        if abs(float(v2_osnr) - float(legacy_osnr_value)) > osnr_tolerance:
            osnr_matches = False
            break

    result = {
        "topology_id": topology_id,
        "request_count": request_count,
        "reverse_request_count": sum(1 for record in records if record.source_id > record.destination_id),
        "status_sequence": tuple(v2["statuses"]),
        "legacy_status_sequence": tuple(legacy["statuses"]),
        "status_matches": status_matches,
        "mask_matches": mask_matches,
        "slot_matches": slot_matches,
        "active_count_matches": active_count_matches,
        "osnr_matches": osnr_matches,
        "episode_services_accepted_matches": (
            v2["episode_services_accepted"] == legacy["episode_services_accepted"]
        ),
    }
    if include_records:
        result["records"] = tuple(asdict(record) for record in records)
    return result


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
        reset_durations.append(int(result["reset_ns"]))
        step_durations.extend(int(duration) for duration in result["step_durations_ns"])
        episode_durations.append(episode_elapsed)
        accepted = int(result["episode_services_accepted"])

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


def benchmark_integrated_episode_vs_legacy(
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

    v2_reset_durations: list[int] = []
    v2_step_durations: list[int] = []
    v2_episode_durations: list[int] = []
    legacy_setup_durations: list[int] = []
    legacy_apply_durations: list[int] = []
    legacy_request_cycle_durations: list[int] = []
    legacy_episode_durations: list[int] = []

    for repeat_index in range(repeats + warmup):
        start_ns = time.perf_counter_ns()
        v2_result = _run_v2_episode(static_config=static_config, topology=topology, records=records)
        v2_episode_elapsed = time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        legacy_result = _run_legacy_replay_episode(
            topology_id=topology_id,
            num_spectrum_resources=num_spectrum_resources,
            k_paths=k_paths,
            seed=seed,
            load=load,
            mean_holding_time=mean_holding_time,
            records=records,
        )
        legacy_episode_elapsed = time.perf_counter_ns() - start_ns

        if repeat_index < warmup:
            continue
        v2_reset_durations.append(int(v2_result["reset_ns"]))
        v2_step_durations.extend(int(duration) for duration in v2_result["step_durations_ns"])
        v2_episode_durations.append(v2_episode_elapsed)

        legacy_setup_durations.append(int(legacy_result["setup_ns"]))
        legacy_apply_durations.extend(int(duration) for duration in legacy_result["apply_durations_ns"])
        legacy_request_cycle_durations.extend(
            int(duration) for duration in legacy_result["request_cycle_durations_ns"]
        )
        legacy_episode_durations.append(legacy_episode_elapsed)

    comparison = compare_simulator_episode_with_legacy(
        topology_id=topology_id,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        request_count=request_count,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
    )

    v2_reset_mean_us, v2_reset_p95_us = _durations_summary_us(v2_reset_durations)
    v2_step_mean_us, v2_step_p95_us = _durations_summary_us(v2_step_durations)
    v2_episode_mean_us, v2_episode_p95_us = _durations_summary_us(v2_episode_durations)
    legacy_setup_mean_us, legacy_setup_p95_us = _durations_summary_us(legacy_setup_durations)
    legacy_apply_mean_us, legacy_apply_p95_us = _durations_summary_us(legacy_apply_durations)
    legacy_step_mean_us, legacy_step_p95_us = _durations_summary_us(legacy_request_cycle_durations)
    legacy_episode_mean_us, legacy_episode_p95_us = _durations_summary_us(legacy_episode_durations)

    return {
        "component": "IntegratedEpisodeReplay",
        "topology_id": topology_id,
        "request_count": request_count,
        "repeats": repeats,
        "warmup": warmup,
        "v2_reset_mean_us": v2_reset_mean_us,
        "v2_reset_p95_us": v2_reset_p95_us,
        "v2_step_mean_us": v2_step_mean_us,
        "v2_step_p95_us": v2_step_p95_us,
        "v2_episode_mean_us": v2_episode_mean_us,
        "v2_episode_p95_us": v2_episode_p95_us,
        "legacy_setup_mean_us": legacy_setup_mean_us,
        "legacy_setup_p95_us": legacy_setup_p95_us,
        "legacy_apply_mean_us": legacy_apply_mean_us,
        "legacy_apply_p95_us": legacy_apply_p95_us,
        "legacy_step_mean_us": legacy_step_mean_us,
        "legacy_step_p95_us": legacy_step_p95_us,
        "legacy_episode_mean_us": legacy_episode_mean_us,
        "legacy_episode_p95_us": legacy_episode_p95_us,
        "v2_step_speedup_vs_legacy": (
            legacy_step_mean_us / v2_step_mean_us if v2_step_mean_us > 0 else 0.0
        ),
        "v2_episode_speedup_vs_legacy": (
            legacy_episode_mean_us / v2_episode_mean_us if v2_episode_mean_us > 0 else 0.0
        ),
        "legacy_step_definition": "full request cycle including prepare, mask build, and apply",
        "parity": comparison,
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
    entries: list[dict[str, object]] = []
    for function_descriptor, function_stats in stats.stats.items():
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
    entries.sort(key=lambda entry: entry["cumulative_time_s"], reverse=True)

    stream = io.StringIO()
    stats.stream = stream
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
    "benchmark_integrated_episode_vs_legacy",
    "compare_simulator_episode_with_legacy",
    "profile_simulator_episode",
]
