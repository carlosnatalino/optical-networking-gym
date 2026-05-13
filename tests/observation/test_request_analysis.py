from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym_v2 import Modulation, QoTEngine, RequestAnalysisEngine, RuntimeState, ScenarioConfig, ServiceRequest, TopologyModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="request_analysis",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )


def _request(service_id: int = 0) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=2,
        bit_rate=40,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def test_request_analysis_reuses_cached_object_for_same_state_and_request() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    engine = RequestAnalysisEngine(config, topology, QoTEngine(config, topology))
    request = _request(41)

    first = engine.build(state, request)
    second = engine.build(state, request)

    assert first is second
    assert engine.cache_misses == 1
    assert engine.cache_hits == 1


def test_request_analysis_invalidates_on_state_change() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    engine = RequestAnalysisEngine(config, topology, qot_engine)
    request = _request(42)
    path = topology.get_paths("1", "3")[0]
    candidate = qot_engine.build_candidate(
        request=request,
        path=path,
        modulation=config.modulations[0],
        service_slot_start=2,
        service_num_slots=2,
    )

    before = engine.build(state, request)
    state.apply_provision(
        request=candidate.request,
        path=path,
        service_slot_start=candidate.service_slot_start,
        service_num_slots=candidate.service_num_slots,
        occupied_slot_start=candidate.service_slot_start,
        occupied_slot_end_exclusive=candidate.service_slot_start + candidate.service_num_slots + 1,
        modulation=candidate.modulation,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
        launch_power=candidate.launch_power,
    )
    after = engine.build(state, _request(43))

    assert before is not after
    assert engine.cache_misses == 2


def test_request_analysis_uses_distinct_runtime_state_identity_for_cache() -> None:
    topology = _topology()
    config = _config()
    qot_engine = QoTEngine(config, topology)
    engine = RequestAnalysisEngine(config, topology, qot_engine)
    first_state = RuntimeState(config, topology)
    second_state = RuntimeState(config, topology)
    request = _request(45)

    first = engine.build(first_state, request)
    second = engine.build(second_state, request)

    assert first is not second
    assert engine.cache_misses == 2
    assert first.state_id != second.state_id


def test_request_analysis_global_features_remain_stable_after_state_mutation() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    engine = RequestAnalysisEngine(config, topology, qot_engine)
    request = _request(46)
    analysis = engine.build(state, request)

    path = topology.get_paths("1", "3")[0]
    provision_request = _request(47)
    candidate = qot_engine.build_candidate(
        request=provision_request,
        path=path,
        modulation=config.modulations[0],
        service_slot_start=0,
        service_num_slots=2,
    )
    state.apply_provision(
        request=provision_request,
        path=path,
        service_slot_start=candidate.service_slot_start,
        service_num_slots=candidate.service_num_slots,
        occupied_slot_start=candidate.service_slot_start,
        occupied_slot_end_exclusive=candidate.service_slot_start + candidate.service_num_slots + 1,
        modulation=candidate.modulation,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
        launch_power=candidate.launch_power,
    )

    free_slots_index = engine.global_feature_names.index("free_slots_ratio")
    active_services_index = engine.global_feature_names.index("active_services_norm")

    assert analysis.global_features[free_slots_index] == pytest.approx(1.0)
    assert analysis.global_features[active_services_index] == pytest.approx(0.0)


def test_request_analysis_exposes_path_slot_flags_consistent_with_mask_logic() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    for link_id in path.link_ids:
        state.slot_allocation[link_id, 0:4] = 999

    engine = RequestAnalysisEngine(config, topology, QoTEngine(config, topology))
    analysis = engine.build(state, _request(44))

    assert analysis.path_slot_features.shape == (2, 24, 9)
    assert np.all(analysis.path_slot_features[0, 0:4, 0] == 0.0)
    assert np.all(analysis.path_slot_features[0, 0:4, 5] == 0.0)


def test_request_analysis_can_build_lean_and_inspection_variants_separately() -> None:
    topology = _topology()
    config = ScenarioConfig(
        scenario_id="request_analysis_lean",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=_config().modulations,
        modulations_to_consider=2,
        enable_observation=False,
    )
    engine = RequestAnalysisEngine(config, topology, QoTEngine(config, topology))
    state = RuntimeState(config, topology)
    request = _request(48)

    lean = engine.build(state, request)
    rich = engine.build(state, request, include_inspection=True)

    assert lean is not rich
    assert lean.inspection is None
    assert rich.inspection is not None
    assert engine.cache_misses == 2


def test_request_analysis_requires_modulations() -> None:
    topology = _topology()
    config = ScenarioConfig(
        scenario_id="analysis_invalid",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
    )

    with pytest.raises(ValueError, match="modulations"):
        RequestAnalysisEngine(config, topology, QoTEngine(config, topology))
