from __future__ import annotations

import pytest

import optical_networking_gym.core.osnr as legacy_osnr
from optical_networking_gym.topology import Modulation as LegacyModulation

from optical_networking_gym_v2 import Modulation, QoTEngine, RuntimeState

from .helpers import (
    build_legacy_qrmsa_env,
    build_legacy_service,
    build_ring4_config,
    build_ring4_topology_v2,
    build_service_request,
)


def _v2_modulation() -> Modulation:
    return Modulation(
        name="QPSK",
        maximum_length=200_000.0,
        spectral_efficiency=2,
        minimum_osnr=6.72,
        inband_xt=-17.0,
    )


def _legacy_modulation() -> LegacyModulation:
    return LegacyModulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17)


def test_qot_engine_matches_legacy_for_isolated_candidate() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    config = build_ring4_config(scenario_id="qot_equivalence")
    state = RuntimeState(config, topology)
    engine = QoTEngine(config, topology)
    path = topology.get_paths("1", "3")[0]
    request = build_service_request(service_id=21)
    modulation = _v2_modulation()

    candidate = engine.build_candidate(
        request=request,
        path=path,
        modulation=modulation,
        service_slot_start=4,
        service_num_slots=3,
    )
    result = engine.evaluate_candidate(state, candidate)

    legacy_env = build_legacy_qrmsa_env(k_paths=2)
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_service = build_legacy_service(request, legacy_env)
    legacy_service.path = legacy_path
    legacy_service.initial_slot = 4
    legacy_service.number_slots = 3
    legacy_service.center_frequency = candidate.center_frequency
    legacy_service.bandwidth = candidate.bandwidth
    legacy_service.launch_power = candidate.launch_power
    legacy_service.current_modulation = _legacy_modulation()

    legacy_gsnr, legacy_ase, legacy_nli = legacy_osnr.calculate_osnr(legacy_env, legacy_service, "ASE+NLI")

    assert result.osnr == pytest.approx(legacy_gsnr, abs=1e-6)
    assert result.ase == pytest.approx(legacy_ase, abs=1e-6)
    assert result.nli == pytest.approx(legacy_nli, abs=1e-6)


def test_qot_engine_matches_legacy_for_candidate_with_interference() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    config = build_ring4_config(scenario_id="qot_equivalence_interference")
    state = RuntimeState(config, topology)
    engine = QoTEngine(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = _v2_modulation()

    first_request = build_service_request(service_id=30)
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

    second_request = build_service_request(service_id=31)
    second = engine.build_candidate(second_request, path, modulation, service_slot_start=6, service_num_slots=2)
    result = engine.evaluate_candidate(state, second)

    legacy_env = build_legacy_qrmsa_env(k_paths=2)
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_first = build_legacy_service(first_request, legacy_env)
    legacy_first.path = legacy_path
    legacy_first.initial_slot = first.service_slot_start
    legacy_first.number_slots = first.service_num_slots
    legacy_first.center_frequency = first.center_frequency
    legacy_first.bandwidth = first.bandwidth
    legacy_first.launch_power = first.launch_power
    legacy_first.current_modulation = _legacy_modulation()
    legacy_env.current_service = legacy_first
    legacy_env._provision_path(legacy_path, first.service_slot_start, first.service_num_slots)

    legacy_second = build_legacy_service(second_request, legacy_env)
    legacy_second.path = legacy_path
    legacy_second.initial_slot = second.service_slot_start
    legacy_second.number_slots = second.service_num_slots
    legacy_second.center_frequency = second.center_frequency
    legacy_second.bandwidth = second.bandwidth
    legacy_second.launch_power = second.launch_power
    legacy_second.current_modulation = _legacy_modulation()

    legacy_gsnr, legacy_ase, legacy_nli = legacy_osnr.calculate_osnr(legacy_env, legacy_second, "ASE+NLI")

    assert result.osnr == pytest.approx(legacy_gsnr, abs=1e-6)
    assert result.ase == pytest.approx(legacy_ase, abs=1e-6)
    assert result.nli == pytest.approx(legacy_nli, abs=1e-6)


def test_qot_engine_recomputes_impacted_service_like_legacy() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    config = build_ring4_config(scenario_id="qot_equivalence_refresh")
    state = RuntimeState(config, topology)
    engine = QoTEngine(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = _v2_modulation()

    first_request = build_service_request(service_id=40)
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

    second_request = build_service_request(service_id=41)
    second = engine.build_candidate(second_request, path, modulation, service_slot_start=6, service_num_slots=2)
    state.apply_provision(
        request=second.request,
        path=path,
        service_slot_start=second.service_slot_start,
        service_num_slots=second.service_num_slots,
        occupied_slot_start=second.service_slot_start,
        occupied_slot_end_exclusive=second.service_slot_start + second.service_num_slots + 1,
        modulation=second.modulation,
        center_frequency=second.center_frequency,
        bandwidth=second.bandwidth,
        launch_power=second.launch_power,
    )

    update = engine.recompute_service(state, first_request.service_id)

    legacy_env = build_legacy_qrmsa_env(k_paths=2)
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_first = build_legacy_service(first_request, legacy_env)
    legacy_first.path = legacy_path
    legacy_first.initial_slot = first.service_slot_start
    legacy_first.number_slots = first.service_num_slots
    legacy_first.center_frequency = first.center_frequency
    legacy_first.bandwidth = first.bandwidth
    legacy_first.launch_power = first.launch_power
    legacy_first.current_modulation = _legacy_modulation()
    legacy_env.current_service = legacy_first
    legacy_env._provision_path(legacy_path, first.service_slot_start, first.service_num_slots)

    legacy_second = build_legacy_service(second_request, legacy_env)
    legacy_second.path = legacy_path
    legacy_second.initial_slot = second.service_slot_start
    legacy_second.number_slots = second.service_num_slots
    legacy_second.center_frequency = second.center_frequency
    legacy_second.bandwidth = second.bandwidth
    legacy_second.launch_power = second.launch_power
    legacy_second.current_modulation = _legacy_modulation()
    legacy_env.current_service = legacy_second
    legacy_env._provision_path(legacy_path, second.service_slot_start, second.service_num_slots)

    legacy_gsnr, legacy_ase, legacy_nli = legacy_osnr.calculate_osnr(legacy_env, legacy_first, "ASE+NLI")

    assert update.osnr == pytest.approx(legacy_gsnr, abs=1e-6)
    assert update.ase == pytest.approx(legacy_ase, abs=1e-6)
    assert update.nli == pytest.approx(legacy_nli, abs=1e-6)
