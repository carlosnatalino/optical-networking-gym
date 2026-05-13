from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

import optical_networking_gym.core.osnr as legacy_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv, Service
from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym_v2 import (
    ActionMask,
    Modulation as V2Modulation,
    QoTEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
    benchmark_action_mask,
    benchmark_allocation,
    benchmark_qot_engine,
    benchmark_runtime_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "examples" / "topologies" / "ring_4.txt"


def _durations_summary_us(durations_ns: list[int]) -> tuple[float, float]:
    if not durations_ns:
        return 0.0, 0.0
    durations_us = [duration / 1_000.0 for duration in durations_ns]
    return float(statistics.fmean(durations_us)), float(np.percentile(durations_us, 95))


def _build_legacy_env() -> QRMSAEnv:
    topology = get_topology(
        str(RING_4_PATH),
        topology_name="ring_4",
        modulations=(
            Modulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17),
            Modulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
        ),
        max_span_length=100.0,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=2,
    )
    return QRMSAEnv(
        topology=topology,
        num_spectrum_resources=24,
        episode_length=10,
        load=10.0,
        mean_service_holding_time=100.0,
        bit_rate_selection="discrete",
        bit_rates=(40,),
        bit_rate_probabilities=(1.0,),
        bandwidth=24 * 12.5e9,
        seed=7,
        reset=True,
        gen_observation=False,
        k_paths=2,
    )


def _build_legacy_service(service_id: int, legacy_env: QRMSAEnv) -> Service:
    return Service(
        service_id=service_id,
        source="1",
        source_id=0,
        destination="3",
        destination_id="2",
        arrival_time=1.0 + service_id,
        holding_time=10.0,
        bit_rate=40.0,
    )


def _build_request(service_id: int) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=2,
        bit_rate=40,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def benchmark_runtime_state_vs_legacy(
    *,
    iterations: int = 1_000,
    warmup: int = 100,
) -> dict[str, float | int | str]:
    v2 = benchmark_runtime_state(iterations=iterations, warmup=warmup)
    legacy_env = _build_legacy_env()
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]

    provision_durations: list[int] = []
    release_durations: list[int] = []
    cycle_durations: list[int] = []
    for index in range(iterations + warmup):
        legacy_service = _build_legacy_service(index, legacy_env)
        legacy_env.current_service = legacy_service

        cycle_start = time.perf_counter_ns()
        start_ns = time.perf_counter_ns()
        legacy_env._provision_path(legacy_path, 4, 3)
        legacy_env._add_release(legacy_service)
        provision_elapsed = time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        legacy_env._release_path(legacy_service)
        release_elapsed = time.perf_counter_ns() - start_ns
        cycle_elapsed = time.perf_counter_ns() - cycle_start

        if index < warmup:
            continue
        provision_durations.append(provision_elapsed)
        release_durations.append(release_elapsed)
        cycle_durations.append(cycle_elapsed)

    legacy_provision_mean_us, _ = _durations_summary_us(provision_durations)
    legacy_release_mean_us, _ = _durations_summary_us(release_durations)
    legacy_cycle_mean_us, legacy_cycle_p95_us = _durations_summary_us(cycle_durations)
    return {
        "component": "RuntimeState",
        "iterations": iterations,
        "warmup": warmup,
        "v2_provision_mean_us": v2["provision_mean_us"],
        "v2_release_mean_us": v2["release_mean_us"],
        "v2_cycle_mean_us": v2["cycle_mean_us"],
        "v2_cycle_p95_us": v2["cycle_p95_us"],
        "legacy_provision_mean_us": legacy_provision_mean_us,
        "legacy_release_mean_us": legacy_release_mean_us,
        "legacy_cycle_mean_us": legacy_cycle_mean_us,
        "legacy_cycle_p95_us": legacy_cycle_p95_us,
        "speedup_vs_legacy": (legacy_cycle_mean_us / v2["cycle_mean_us"]) if v2["cycle_mean_us"] > 0 else 0.0,
    }


def benchmark_allocation_vs_legacy(
    *,
    iterations: int = 5_000,
    warmup: int = 500,
) -> dict[str, float | int | str]:
    v2 = benchmark_allocation(iterations=iterations, warmup=warmup)

    legacy_env = _build_legacy_env()
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    for index in range(len(legacy_path.node_list) - 1):
        link_index = legacy_env.topology[legacy_path.node_list[index]][legacy_path.node_list[index + 1]]["index"]
        legacy_env.topology.graph["available_slots"][link_index, 2:5] = 0
        legacy_env.topology.graph["available_slots"][link_index, 9:11] = 0
        legacy_env.topology.graph["available_slots"][link_index, 15:18] = 0

    legacy_durations: list[int] = []
    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        legacy_available = legacy_env.get_available_slots(legacy_path)
        legacy_candidates = legacy_env._get_candidates(legacy_available, 2, 24)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        legacy_durations.append(elapsed)
        assert len(legacy_candidates) == v2["candidate_count"]

    legacy_mean_us, legacy_p95_us = _durations_summary_us(legacy_durations)
    return {
        "component": "Allocation",
        "iterations": iterations,
        "warmup": warmup,
        "candidate_count": v2["candidate_count"],
        "v2_mean_us": v2["mean_us"],
        "v2_p95_us": v2["p95_us"],
        "legacy_mean_us": legacy_mean_us,
        "legacy_p95_us": legacy_p95_us,
        "speedup_vs_legacy": (legacy_mean_us / v2["mean_us"]) if v2["mean_us"] > 0 else 0.0,
    }


def benchmark_qot_engine_vs_legacy(
    *,
    iterations: int = 1_000,
    warmup: int = 100,
) -> dict[str, float | int | str]:
    v2 = benchmark_qot_engine(iterations=iterations, warmup=warmup)

    topology = TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)
    config = ScenarioConfig(
        scenario_id="benchmark_qot_vs_legacy",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
    )
    state = RuntimeState(config, topology)
    engine = QoTEngine(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = V2Modulation(
        name="QPSK",
        maximum_length=200_000.0,
        spectral_efficiency=2,
        minimum_osnr=6.72,
        inband_xt=-17.0,
    )

    first_request = ServiceRequest(
        request_index=10_000,
        service_id=10_000,
        source_id=0,
        destination_id=2,
        bit_rate=40,
        arrival_time=1.0,
        holding_time=10.0,
    )
    second_request = ServiceRequest(
        request_index=10_001,
        service_id=10_001,
        source_id=0,
        destination_id=2,
        bit_rate=40,
        arrival_time=2.0,
        holding_time=10.0,
    )
    first = engine.build_candidate(first_request, path, modulation, service_slot_start=2, service_num_slots=2)
    state.apply_provision(
        request=first.request,
        path=path,
        service_slot_start=first.service_slot_start,
        service_num_slots=first.service_num_slots,
        occupied_slot_start=first.service_slot_start,
        occupied_slot_end_exclusive=first.service_slot_start + first.service_num_slots + 1,
        modulation=first.modulation,
        center_frequency=first.center_frequency,
        bandwidth=first.bandwidth,
        launch_power=first.launch_power,
    )
    second = engine.build_candidate(second_request, path, modulation, service_slot_start=6, service_num_slots=2)

    legacy_env = _build_legacy_env()
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_modulation = Modulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17)
    legacy_first = _build_legacy_service(first_request.service_id, legacy_env)
    legacy_first.path = legacy_path
    legacy_first.initial_slot = first.service_slot_start
    legacy_first.number_slots = first.service_num_slots
    legacy_first.center_frequency = first.center_frequency
    legacy_first.bandwidth = first.bandwidth
    legacy_first.launch_power = first.launch_power
    legacy_first.current_modulation = legacy_modulation
    legacy_env.current_service = legacy_first
    legacy_env._provision_path(legacy_path, first.service_slot_start, first.service_num_slots)

    legacy_second = _build_legacy_service(second_request.service_id, legacy_env)
    legacy_second.path = legacy_path
    legacy_second.initial_slot = second.service_slot_start
    legacy_second.number_slots = second.service_num_slots
    legacy_second.center_frequency = second.center_frequency
    legacy_second.bandwidth = second.bandwidth
    legacy_second.launch_power = second.launch_power
    legacy_second.current_modulation = legacy_modulation

    evaluate_durations: list[int] = []
    refresh_durations: list[int] = []
    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        legacy_candidate = legacy_osnr.calculate_osnr(legacy_env, legacy_second, "ASE+NLI")
        evaluate_elapsed = time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        legacy_refresh = legacy_osnr.calculate_osnr(legacy_env, legacy_first, "ASE+NLI")
        refresh_elapsed = time.perf_counter_ns() - start_ns

        if index < warmup:
            continue
        assert legacy_candidate[0] != 0.0
        assert legacy_refresh[0] != 0.0
        evaluate_durations.append(evaluate_elapsed)
        refresh_durations.append(refresh_elapsed)

    legacy_eval_mean_us, legacy_eval_p95_us = _durations_summary_us(evaluate_durations)
    legacy_refresh_mean_us, legacy_refresh_p95_us = _durations_summary_us(refresh_durations)
    return {
        "component": "QoTEngine",
        "iterations": iterations,
        "warmup": warmup,
        "v2_evaluate_candidate_mean_us": v2["evaluate_candidate_mean_us"],
        "v2_evaluate_candidate_p95_us": v2["evaluate_candidate_p95_us"],
        "v2_refresh_service_mean_us": v2["refresh_service_mean_us"],
        "v2_refresh_service_p95_us": v2["refresh_service_p95_us"],
        "legacy_evaluate_candidate_mean_us": legacy_eval_mean_us,
        "legacy_evaluate_candidate_p95_us": legacy_eval_p95_us,
        "legacy_refresh_service_mean_us": legacy_refresh_mean_us,
        "legacy_refresh_service_p95_us": legacy_refresh_p95_us,
        "evaluate_speedup_vs_legacy": (
            legacy_eval_mean_us / v2["evaluate_candidate_mean_us"]
            if v2["evaluate_candidate_mean_us"] > 0
            else 0.0
        ),
        "refresh_speedup_vs_legacy": (
            legacy_refresh_mean_us / v2["refresh_service_mean_us"]
            if v2["refresh_service_mean_us"] > 0
            else 0.0
        ),
    }


def _build_legacy_action_mask(legacy_env: QRMSAEnv) -> np.ndarray:
    current_service = legacy_env.current_service
    if current_service is None:
        raise ValueError("legacy_env.current_service is required")

    legacy_env.get_max_modulation_index()
    paths_info: list[tuple[object, np.ndarray]] = []
    for path_index, route in enumerate(legacy_env.k_shortest_paths[current_service.source, current_service.destination]):
        if path_index >= legacy_env.k_paths:
            break
        paths_info.append((route, legacy_env.get_available_slots(route)))

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
            start_index = max(0, legacy_env.max_modulation_idx - (legacy_env.modulations_to_consider - 1))
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
            legacy_env.frequency_start + legacy_env.frequency_slot_bandwidth * (initial_slot + number_slots / 2)
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


def _window_mask(available_slots: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0 or window_size > available_slots.shape[0]:
        return np.zeros((0,), dtype=np.uint8)
    slot_view = np.asarray(available_slots, dtype=np.int8)
    kernel = np.ones(window_size, dtype=np.int8)
    window_sums = np.convolve(slot_view, kernel, mode="valid")
    return (window_sums == window_size).astype(np.uint8)


def benchmark_action_mask_vs_legacy(
    *,
    iterations: int = 250,
    warmup: int = 25,
) -> dict[str, float | int | str]:
    v2 = benchmark_action_mask(iterations=iterations, warmup=warmup)

    topology = TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)
    config = ScenarioConfig(
        scenario_id="benchmark_action_mask_vs_legacy",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            V2Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            V2Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )
    state = RuntimeState(config, topology)
    engine = QoTEngine(config, topology)
    builder = ActionMask(config, topology, engine)
    path = topology.get_paths("1", "3")[0]
    first_request = _build_request(30_000)
    first = engine.build_candidate(first_request, path, config.modulations[0], service_slot_start=2, service_num_slots=2)
    state.apply_provision(
        request=first.request,
        path=path,
        service_slot_start=first.service_slot_start,
        service_num_slots=first.service_num_slots,
        occupied_slot_start=first.service_slot_start,
        occupied_slot_end_exclusive=first.service_slot_start + first.service_num_slots + 1,
        modulation=first.modulation,
        center_frequency=first.center_frequency,
        bandwidth=first.bandwidth,
        launch_power=first.launch_power,
    )
    second_request = _build_request(30_001)

    legacy_env = _build_legacy_env()
    legacy_first = _build_legacy_service(first_request.service_id, legacy_env)
    legacy_env.current_service = legacy_first
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_first.path = legacy_path
    legacy_first.initial_slot = first.service_slot_start
    legacy_first.number_slots = first.service_num_slots
    legacy_first.current_modulation = legacy_env.modulations[0]
    legacy_first.center_frequency = first.center_frequency
    legacy_first.bandwidth = first.bandwidth
    legacy_first.launch_power = first.launch_power
    legacy_env._provision_path(legacy_path, first.service_slot_start, first.service_num_slots)
    legacy_env.current_service = _build_legacy_service(second_request.service_id, legacy_env)

    legacy_durations: list[int] = []
    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        legacy_mask = _build_legacy_action_mask(legacy_env)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        legacy_durations.append(elapsed)
        assert int(legacy_mask[:-1].sum()) == v2["valid_actions"]

    legacy_mean_us, legacy_p95_us = _durations_summary_us(legacy_durations)
    return {
        "component": "ActionMask",
        "iterations": iterations,
        "warmup": warmup,
        "valid_actions": v2["valid_actions"],
        "v2_warm_mean_us": v2["warm_mean_us"],
        "v2_warm_p95_us": v2["warm_p95_us"],
        "v2_cold_mean_us": v2["cold_mean_us"],
        "v2_cold_p95_us": v2["cold_p95_us"],
        "legacy_mean_us": legacy_mean_us,
        "legacy_p95_us": legacy_p95_us,
        "warm_speedup_vs_legacy": (legacy_mean_us / v2["warm_mean_us"]) if v2["warm_mean_us"] > 0 else 0.0,
        "cold_speedup_vs_legacy": (legacy_mean_us / v2["cold_mean_us"]) if v2["cold_mean_us"] > 0 else 0.0,
    }


def run_hot_path_benchmarks(
    *,
    runtime_iterations: int = 1_000,
    runtime_warmup: int = 100,
    allocation_iterations: int = 5_000,
    allocation_warmup: int = 500,
    qot_iterations: int = 1_000,
    qot_warmup: int = 100,
    action_mask_iterations: int = 250,
    action_mask_warmup: int = 25,
) -> dict[str, object]:
    return {
        "scenario": "ring_4__24_slots",
        "runtime_state": benchmark_runtime_state_vs_legacy(
            iterations=runtime_iterations,
            warmup=runtime_warmup,
        ),
        "allocation": benchmark_allocation_vs_legacy(
            iterations=allocation_iterations,
            warmup=allocation_warmup,
        ),
        "qot_engine": benchmark_qot_engine_vs_legacy(
            iterations=qot_iterations,
            warmup=qot_warmup,
        ),
        "action_mask": benchmark_action_mask_vs_legacy(
            iterations=action_mask_iterations,
            warmup=action_mask_warmup,
        ),
    }


__all__ = [
    "benchmark_action_mask_vs_legacy",
    "benchmark_allocation_vs_legacy",
    "benchmark_qot_engine_vs_legacy",
    "benchmark_runtime_state_vs_legacy",
    "run_hot_path_benchmarks",
]
