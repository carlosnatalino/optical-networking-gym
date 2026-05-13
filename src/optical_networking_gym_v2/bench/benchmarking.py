from __future__ import annotations

from pathlib import Path
import statistics
import time

import numpy as np

from optical_networking_gym_v2.contracts.allocation import Allocation
from optical_networking_gym_v2.contracts.modulation import Modulation
from optical_networking_gym_v2.contracts.reward import CandidateRewardMetrics, RewardInput
from optical_networking_gym_v2.contracts.step import StepTransition
from optical_networking_gym_v2.contracts.traffic import ServiceRequest
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState
from optical_networking_gym_v2.runtime.step_info import StepInfo
from optical_networking_gym_v2.network.allocation import (
    available_slots_for_path,
    build_first_fit_allocation,
    candidate_starts,
)
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.optical.qot_engine import QoTEngine
from optical_networking_gym_v2.features.action_mask import ActionMask
from optical_networking_gym_v2.features.observation import Observation
from optical_networking_gym_v2.rl.reward_function import RewardFunction
from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysisEngine
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.stats.statistics import Statistics


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RING_4_PATH = PROJECT_ROOT / "examples" / "topologies" / "ring_4.txt"


def _durations_summary_us(durations_ns: list[int]) -> tuple[float, float]:
    if not durations_ns:
        return 0.0, 0.0
    durations_us = [duration / 1_000.0 for duration in durations_ns]
    return float(statistics.fmean(durations_us)), float(np.percentile(durations_us, 95))


def _build_v2_topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _build_v2_config(scenario_id: str) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=scenario_id,
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
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


def _build_modulation() -> Modulation:
    return Modulation(
        name="QPSK",
        maximum_length=200_000.0,
        spectral_efficiency=2,
        minimum_osnr=6.72,
        inband_xt=-17.0,
    )


def benchmark_runtime_state(*, iterations: int = 1_000, warmup: int = 100) -> dict[str, float | int | str]:
    topology = _build_v2_topology()
    state = RuntimeState(_build_v2_config("benchmark_runtime"), topology)
    path = topology.get_paths("1", "3")[0]
    requests = [_build_request(index) for index in range(iterations + warmup)]

    provision_durations: list[int] = []
    release_durations: list[int] = []
    cycle_durations: list[int] = []

    for index, request in enumerate(requests):
        cycle_start = time.perf_counter_ns()
        start_ns = time.perf_counter_ns()
        state.apply_provision(
            request=request,
            path=path,
            service_slot_start=4,
            service_num_slots=3,
            occupied_slot_start=4,
            occupied_slot_end_exclusive=8,
        )
        provision_elapsed = time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        state.apply_release(request.service_id)
        release_elapsed = time.perf_counter_ns() - start_ns
        cycle_elapsed = time.perf_counter_ns() - cycle_start

        if index < warmup:
            continue
        provision_durations.append(provision_elapsed)
        release_durations.append(release_elapsed)
        cycle_durations.append(cycle_elapsed)

    provision_mean_us, _ = _durations_summary_us(provision_durations)
    release_mean_us, _ = _durations_summary_us(release_durations)
    cycle_mean_us, cycle_p95_us = _durations_summary_us(cycle_durations)
    return {
        "component": "RuntimeState",
        "iterations": iterations,
        "warmup": warmup,
        "provision_mean_us": provision_mean_us,
        "release_mean_us": release_mean_us,
        "cycle_mean_us": cycle_mean_us,
        "cycle_p95_us": cycle_p95_us,
    }


def benchmark_allocation(*, iterations: int = 5_000, warmup: int = 500) -> dict[str, float | int | str]:
    topology = _build_v2_topology()
    state = RuntimeState(_build_v2_config("benchmark_allocation"), topology)
    path = topology.get_paths("1", "3")[0]

    for link_id in path.link_ids:
        state.slot_allocation[link_id, 2:5] = 101
        state.slot_allocation[link_id, 9:11] = 202
        state.slot_allocation[link_id, 15:18] = 303

    durations: list[int] = []
    candidate_count = 0

    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        available = available_slots_for_path(state, path)
        candidates = candidate_starts(available, required_slots=2, total_slots=24)
        build_first_fit_allocation(
            state,
            path=path,
            path_index=0,
            modulation_index=0,
            service_num_slots=2,
        )
        elapsed = time.perf_counter_ns() - start_ns

        if index < warmup:
            continue
        candidate_count = len(candidates)
        durations.append(elapsed)

    mean_us, p95_us = _durations_summary_us(durations)
    return {
        "component": "Allocation",
        "iterations": iterations,
        "warmup": warmup,
        "candidate_count": candidate_count,
        "mean_us": mean_us,
        "p95_us": p95_us,
    }


def benchmark_qot_engine(*, iterations: int = 1_000, warmup: int = 100) -> dict[str, float | int | str]:
    topology = _build_v2_topology()
    config = _build_v2_config("benchmark_qot")
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = _build_modulation()

    first = engine.build_candidate(
        request=_build_request(10_000),
        path=path,
        modulation=modulation,
        service_slot_start=2,
        service_num_slots=2,
    )
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
    second = engine.build_candidate(
        request=_build_request(10_001),
        path=path,
        modulation=modulation,
        service_slot_start=6,
        service_num_slots=2,
    )

    evaluate_durations: list[int] = []
    refresh_durations: list[int] = []
    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        result = engine.evaluate_candidate(state, second)
        evaluate_elapsed = time.perf_counter_ns() - start_ns

        start_ns = time.perf_counter_ns()
        update = engine.recompute_service(state, first.request.service_id)
        refresh_elapsed = time.perf_counter_ns() - start_ns

        if index < warmup:
            continue
        assert result.osnr != 0.0
        assert update.service_id == first.request.service_id
        evaluate_durations.append(evaluate_elapsed)
        refresh_durations.append(refresh_elapsed)

    evaluate_mean_us, evaluate_p95_us = _durations_summary_us(evaluate_durations)
    refresh_mean_us, refresh_p95_us = _durations_summary_us(refresh_durations)
    return {
        "component": "QoTEngine",
        "iterations": iterations,
        "warmup": warmup,
        "evaluate_candidate_mean_us": evaluate_mean_us,
        "evaluate_candidate_p95_us": evaluate_p95_us,
        "refresh_service_mean_us": refresh_mean_us,
        "refresh_service_p95_us": refresh_p95_us,
    }


def benchmark_action_mask(*, iterations: int = 250, warmup: int = 25) -> dict[str, float | int | str]:
    topology = _build_v2_topology()
    config = ScenarioConfig(
        scenario_id="benchmark_action_mask",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )
    engine = QoTEngine(config, topology)
    builder = ActionMask(config, topology, engine)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]

    first = engine.build_candidate(
        request=_build_request(20_000),
        path=path,
        modulation=config.modulations[0],
        service_slot_start=2,
        service_num_slots=2,
    )
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
    second_request = _build_request(20_001)

    warm_durations: list[int] = []
    cold_durations: list[int] = []
    valid_actions = 0
    builder.build(state, second_request)

    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        mask = builder.build(state, second_request)
        elapsed = time.perf_counter_ns() - start_ns

        if index < warmup:
            continue
        warm_durations.append(elapsed)
        valid_actions = int(mask[:-1].sum())

    for index in range(iterations + warmup):
        builder.analysis_engine.clear_cache()
        start_ns = time.perf_counter_ns()
        mask = builder.build(state, second_request)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        cold_durations.append(elapsed)
        valid_actions = int(mask[:-1].sum())

    warm_mean_us, warm_p95_us = _durations_summary_us(warm_durations)
    cold_mean_us, cold_p95_us = _durations_summary_us(cold_durations)
    return {
        "component": "ActionMask",
        "iterations": iterations,
        "warmup": warmup,
        "valid_actions": valid_actions,
        "warm_mean_us": warm_mean_us,
        "warm_p95_us": warm_p95_us,
        "cold_mean_us": cold_mean_us,
        "cold_p95_us": cold_p95_us,
    }


def benchmark_observation(*, iterations: int = 100, warmup: int = 10) -> dict[str, float | int | str]:
    topology = _build_v2_topology()
    config = ScenarioConfig(
        scenario_id="benchmark_observation",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    observation = Observation(config, topology, analysis_engine)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]

    first = qot_engine.build_candidate(
        request=_build_request(30_000),
        path=path,
        modulation=config.modulations[0],
        service_slot_start=2,
        service_num_slots=2,
    )
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
    request = _build_request(30_001)

    warm_durations: list[int] = []
    cold_durations: list[int] = []
    feature_size = 0
    observation.build_snapshot(state, request)

    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        snapshot = observation.build_snapshot(state, request)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        warm_durations.append(elapsed)
        feature_size = int(snapshot.flat.size)

    for index in range(iterations + warmup):
        analysis_engine.clear_cache()
        start_ns = time.perf_counter_ns()
        snapshot = observation.build_snapshot(state, request)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        cold_durations.append(elapsed)
        feature_size = int(snapshot.flat.size)

    warm_mean_us, warm_p95_us = _durations_summary_us(warm_durations)
    cold_mean_us, cold_p95_us = _durations_summary_us(cold_durations)
    return {
        "component": "Observation",
        "iterations": iterations,
        "warmup": warmup,
        "feature_size": feature_size,
        "warm_mean_us": warm_mean_us,
        "warm_p95_us": warm_p95_us,
        "cold_mean_us": cold_mean_us,
        "cold_p95_us": cold_p95_us,
    }


def benchmark_request_analysis(*, iterations: int = 100, warmup: int = 10) -> dict[str, float | int | str]:
    topology = _build_v2_topology()
    config = ScenarioConfig(
        scenario_id="benchmark_request_analysis",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    first = qot_engine.build_candidate(
        request=_build_request(35_000),
        path=path,
        modulation=config.modulations[0],
        service_slot_start=2,
        service_num_slots=2,
    )
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
    request = _build_request(35_001)

    warm_durations: list[int] = []
    cold_durations: list[int] = []
    candidate_count = 0

    analysis_engine.build(state, request)
    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        analysis = analysis_engine.build(state, request)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        warm_durations.append(elapsed)
        candidate_count = int(np.count_nonzero(analysis.resource_valid_starts))

    for index in range(iterations + warmup):
        analysis_engine.clear_cache()
        start_ns = time.perf_counter_ns()
        analysis = analysis_engine.build(state, request)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        cold_durations.append(elapsed)
        candidate_count = int(np.count_nonzero(analysis.resource_valid_starts))

    warm_mean_us, warm_p95_us = _durations_summary_us(warm_durations)
    cold_mean_us, cold_p95_us = _durations_summary_us(cold_durations)
    return {
        "component": "RequestAnalysis",
        "iterations": iterations,
        "warmup": warmup,
        "candidate_count": candidate_count,
        "warm_mean_us": warm_mean_us,
        "warm_p95_us": warm_p95_us,
        "cold_mean_us": cold_mean_us,
        "cold_p95_us": cold_p95_us,
    }


def benchmark_statistics_step_info(*, iterations: int = 1_000, warmup: int = 100) -> dict[str, float | int | str]:
    config = ScenarioConfig(
        scenario_id="benchmark_statistics",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )
    step_info = StepInfo(config)
    transition = StepTransition.accept(
        request=_build_request(40_000),
        allocation=build_first_fit_allocation(
            RuntimeState(config, _build_v2_topology()),
            path=_build_v2_topology().get_paths("1", "3")[0],
            path_index=0,
            modulation_index=0,
            service_num_slots=2,
        ),
        modulation_spectral_efficiency=2,
        osnr=15.0,
        osnr_requirement=6.72,
        disrupted_services=1,
    )

    durations: list[int] = []
    for index in range(iterations + warmup):
        local_statistics = Statistics(config)
        start_ns = time.perf_counter_ns()
        local_statistics.record_transition(transition)
        info = step_info.build(local_statistics.snapshot(), transition)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        assert info["accepted"] is True
        durations.append(elapsed)

    mean_us, p95_us = _durations_summary_us(durations)
    return {
        "component": "StatisticsStepInfo",
        "iterations": iterations,
        "warmup": warmup,
        "mean_us": mean_us,
        "p95_us": p95_us,
    }


def benchmark_reward_function(*, iterations: int = 5_000, warmup: int = 500) -> dict[str, float | int | str]:
    topology = _build_v2_topology()
    config = ScenarioConfig(
        scenario_id="benchmark_reward",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )
    reward_function = RewardFunction(config, topology)
    transition = StepTransition.accept(
        request=_build_request(50_000),
        allocation=Allocation.accept(
            path_index=0,
            modulation_index=1,
            service_slot_start=3,
            service_num_slots=2,
            occupied_slot_start=3,
            occupied_slot_end_exclusive=6,
        ),
        modulation_spectral_efficiency=4,
    )
    reward_input = RewardInput(
        transition=transition,
        statistics=Statistics(config).snapshot(),
        selected_candidate_metrics=CandidateRewardMetrics(
            osnr_margin=1.5,
            nli_share=0.25,
            worst_link_nli_share=0.5,
            fragmentation_damage_num_blocks=0.3,
            fragmentation_damage_largest_block=0.2,
        ),
        has_valid_non_reject_action=True,
    )

    durations: list[int] = []
    reward_value = 0.0
    for index in range(iterations + warmup):
        start_ns = time.perf_counter_ns()
        reward_value, breakdown = reward_function.evaluate(reward_input)
        elapsed = time.perf_counter_ns() - start_ns
        if index < warmup:
            continue
        assert breakdown.profile == "balanced"
        durations.append(elapsed)

    mean_us, p95_us = _durations_summary_us(durations)
    return {
        "component": "RewardFunction",
        "iterations": iterations,
        "warmup": warmup,
        "reward": reward_value,
        "mean_us": mean_us,
        "p95_us": p95_us,
    }


__all__ = [
    "benchmark_action_mask",
    "benchmark_allocation",
    "benchmark_observation",
    "benchmark_qot_engine",
    "benchmark_request_analysis",
    "benchmark_reward_function",
    "benchmark_statistics_step_info",
    "benchmark_runtime_state",
]
