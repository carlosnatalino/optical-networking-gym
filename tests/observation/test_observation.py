from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym_v2 import (
    ActionMask,
    MaskMode,
    Modulation,
    Observation,
    QoTEngine,
    RequestAnalysisEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RING_4_PATH = PROJECT_ROOT / "optical_networking_gym_v2" / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def _config(*, mask_mode: MaskMode = MaskMode.RESOURCE_AND_QOT, margin: float = 0.0) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=f"observation_{mask_mode.value}",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        mask_mode=mask_mode,
        margin=margin,
        modulations=_modulations(),
        modulations_to_consider=2,
    )


def _request(service_id: int = 0, *, bit_rate: float = 40.0) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=2,
        bit_rate=bit_rate,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def _occupy_pattern(state: RuntimeState, path_link_ids: tuple[int, ...], spans: tuple[tuple[int, int], ...]) -> None:
    for link_id in path_link_ids:
        for start, end in spans:
            state.slot_allocation[link_id, start:end] = 77


def test_observation_schema_is_stable_and_excludes_holding_time() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    observation = Observation(config, topology, analysis_engine)

    snapshot = observation.build_snapshot(state, _request(5))

    assert snapshot.flat.dtype == np.float32
    assert snapshot.request.shape == (1,)
    assert snapshot.global_features.shape == (8,)
    assert snapshot.path.shape == (2, 13)
    assert snapshot.path_mod.shape == (2, 2, 10)
    assert snapshot.path_slot.shape == (2, 24, 9)
    assert snapshot.flat.shape == (observation.schema.total_size,)
    assert "holding_time" not in " ".join(observation.schema.feature_names)
    assert observation.schema.request_feature_names == ("bit_rate_norm",)


def test_observation_fragmentation_features_reflect_more_fragmented_paths() -> None:
    topology = _topology()
    config = _config()
    path = topology.get_paths("1", "3")[0]
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    observation = Observation(config, topology, analysis_engine)

    contiguous_state = RuntimeState(config, topology)
    fragmented_state = RuntimeState(config, topology)

    _occupy_pattern(contiguous_state, path.link_ids, ((4, 10),))
    _occupy_pattern(fragmented_state, path.link_ids, ((2, 4), (8, 10), (14, 16)))

    contiguous = observation.build_snapshot(contiguous_state, _request(11))
    fragmented = observation.build_snapshot(fragmented_state, _request(11))

    largest_index = observation.schema.path_feature_index("path_common_largest_block_ratio")
    blocks_index = observation.schema.path_feature_index("path_common_num_blocks_norm")
    entropy_index = observation.schema.path_feature_index("path_common_entropy")
    cuts_index = observation.schema.path_feature_index("path_route_cuts_norm")

    assert fragmented.path[0, blocks_index] > contiguous.path[0, blocks_index]
    assert fragmented.path[0, largest_index] < contiguous.path[0, largest_index]
    assert fragmented.path[0, entropy_index] > contiguous.path[0, entropy_index]
    assert fragmented.path[0, cuts_index] > contiguous.path[0, cuts_index]


def test_observation_tracks_resource_and_qot_candidate_separation() -> None:
    topology = _topology()
    config = _config(margin=100.0)
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    observation = Observation(config, topology, analysis_engine)

    snapshot = observation.build_snapshot(state, _request(21))

    resource_index = observation.schema.path_mod_feature_index("resource_candidate_ratio")
    qot_index = observation.schema.path_mod_feature_index("qot_candidate_ratio")

    assert snapshot.path_mod[0, 0, resource_index] > 0.0
    assert snapshot.path_mod[0, 0, qot_index] == 0.0


def test_observation_and_action_mask_share_request_analysis_cache() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    observation = Observation(config, topology, analysis_engine)
    action_mask = ActionMask(config, topology, qot_engine, analysis_engine=analysis_engine)
    request = _request(31)

    mask = action_mask.build(state, request)
    snapshot = observation.build_snapshot(state, request)

    assert analysis_engine.cache_misses == 1
    assert analysis_engine.cache_hits >= 1
    assert np.array_equal(mask[:-1], snapshot.analysis.action_mask)
    assert mask[-1] == 1


def test_observation_build_with_analysis_matches_snapshot() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    observation = Observation(config, topology, analysis_engine)
    request = _request(32)

    flat, analysis = observation.build_with_analysis(state, request)
    snapshot = observation.build_snapshot(state, request)

    assert analysis is snapshot.analysis
    assert flat.dtype == np.float32
    assert np.array_equal(flat, snapshot.flat)


def test_observation_build_returns_empty_array_when_output_is_disabled() -> None:
    topology = _topology()
    config = replace(_config(), enable_observation=False)
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    observation = Observation(config, topology, analysis_engine)

    flat, analysis = observation.build_with_analysis(state, _request(33))
    snapshot = observation.build_snapshot(state, _request(33))

    assert flat.shape == (0,)
    assert flat.dtype == np.float32
    assert analysis.inspection is None
    assert snapshot.flat.shape == (observation.schema.total_size,)
    assert snapshot.analysis.inspection is not None
    assert analysis_engine.cache_misses == 2


def test_observation_requires_modulations() -> None:
    topology = _topology()
    config = ScenarioConfig(
        scenario_id="observation_invalid",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
    )

    with pytest.raises(ValueError, match="modulations"):
        Observation(config, topology, qot_engine=QoTEngine(config, topology))
