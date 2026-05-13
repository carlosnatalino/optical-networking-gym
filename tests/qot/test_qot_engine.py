from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym_v2 import Modulation, QoTEngine, RuntimeState, ScenarioConfig, ServiceRequest, TopologyModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _config(*, measure_disruptions: bool = False) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="qot_ring_4",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        measure_disruptions=measure_disruptions,
    )


def _modulation() -> Modulation:
    return Modulation(
        name="QPSK",
        maximum_length=200_000.0,
        spectral_efficiency=2,
        minimum_osnr=6.72,
        inband_xt=-17.0,
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


def test_build_candidate_uses_configured_frequency_and_power() -> None:
    topology = _topology()
    config = _config()
    engine = QoTEngine(config, topology)
    path = topology.get_paths("1", "3")[0]

    candidate = engine.build_candidate(
        request=_request(3),
        path=path,
        modulation=_modulation(),
        service_slot_start=4,
        service_num_slots=3,
    )

    assert candidate.service_id == 3
    assert candidate.bandwidth == pytest.approx(config.frequency_slot_bandwidth * 3)
    assert candidate.center_frequency == pytest.approx(
        config.frequency_start + config.frequency_slot_bandwidth * 4 + config.frequency_slot_bandwidth * 1.5
    )
    assert candidate.launch_power == pytest.approx(10 ** ((config.launch_power_dbm - 30.0) / 10.0))


def test_impacted_service_ids_collects_services_from_shared_links() -> None:
    topology = _topology()
    config = _config()
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = _modulation()

    first = engine.build_candidate(_request(1), path, modulation, service_slot_start=2, service_num_slots=2)
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

    impacted = engine.impacted_service_ids(state, path, exclude_service_id=99)

    assert impacted == (1,)


def test_recompute_service_invalidates_cached_link_descriptors_on_state_change() -> None:
    topology = _topology()
    config = _config(measure_disruptions=True)
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = _modulation()

    first = engine.build_candidate(_request(10), path, modulation, service_slot_start=2, service_num_slots=2)
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
    initial = engine.recompute_service(state, first.request.service_id)

    second = engine.build_candidate(_request(11), path, modulation, service_slot_start=6, service_num_slots=2)
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
    impacted = engine.recompute_service(state, first.request.service_id)

    state.apply_release(second.request.service_id)
    restored = engine.recompute_service(state, first.request.service_id)

    assert impacted.osnr < initial.osnr
    assert restored.osnr == pytest.approx(initial.osnr, abs=1e-9)


def test_summarize_candidate_starts_matches_scalar_candidate_summaries() -> None:
    topology = _topology()
    config = _config()
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = _modulation()

    first = engine.build_candidate(_request(20), path, modulation, service_slot_start=2, service_num_slots=2)
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

    starts = [0, 5, 8, 12]
    batch = engine.summarize_candidate_starts(
        state=state,
        service_id=21,
        path=path,
        modulation=modulation,
        service_num_slots=2,
        candidate_starts=starts,
    )

    assert batch.meets_threshold.shape == (len(starts),)
    assert batch.osnr_margin.shape == (len(starts),)
    assert batch.nli_share.shape == (len(starts),)
    assert batch.worst_link_nli_share.shape == (len(starts),)

    for index, initial_slot in enumerate(starts):
        scalar = engine.summarize_candidate_at(
            state=state,
            service_id=21,
            path=path,
            modulation=modulation,
            service_slot_start=initial_slot,
            service_num_slots=2,
        )
        assert bool(batch.meets_threshold[index]) is scalar.meets_threshold
        assert float(batch.osnr_margin[index]) == pytest.approx(scalar.osnr_margin, abs=1e-9)
        assert float(batch.nli_share[index]) == pytest.approx(scalar.nli_share, abs=1e-9)
        assert float(batch.worst_link_nli_share[index]) == pytest.approx(
            scalar.worst_link_nli_share,
            abs=1e-9,
        )


def test_prepared_candidate_summary_inputs_match_public_batch_api() -> None:
    topology = _topology()
    config = _config()
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = _modulation()

    first = engine.build_candidate(_request(30), path, modulation, service_slot_start=2, service_num_slots=2)
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

    starts = np.array([0, 5, 8, 12], dtype=np.int32)
    prepared = engine._prepare_candidate_summary_inputs(state, path)

    prepared_batch = engine._summarize_candidate_starts_prepared(
        prepared_inputs=prepared,
        service_id=31,
        service_num_slots=2,
        candidate_starts=starts,
        threshold=modulation.minimum_osnr + config.margin,
    )
    public_batch = engine.summarize_candidate_starts(
        state=state,
        service_id=31,
        path=path,
        modulation=modulation,
        service_num_slots=2,
        candidate_starts=starts,
    )

    assert np.array_equal(prepared_batch.meets_threshold, public_batch.meets_threshold)
    assert np.allclose(prepared_batch.osnr_margin, public_batch.osnr_margin)
    assert np.allclose(prepared_batch.nli_share, public_batch.nli_share)
    assert np.allclose(prepared_batch.worst_link_nli_share, public_batch.worst_link_nli_share)
