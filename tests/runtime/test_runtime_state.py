from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import RuntimeState, ScenarioConfig, ServiceRequest, TopologyModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="runtime_ring_4",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
    )


def _request(service_id: int = 0) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=2,
        bit_rate=40,
        arrival_time=3.0 + service_id,
        holding_time=5.0,
    )


def test_runtime_state_initializes_slot_allocation_and_versions() -> None:
    state = RuntimeState(_config(), _topology())

    assert state.slot_allocation.shape == (4, 24)
    assert state.slot_allocation.dtype.name == "int32"
    assert (state.slot_allocation == -1).all()
    assert state.state_id >= 0
    assert state.global_state_version == 0
    assert state.current_request is None
    assert state.active_services_by_id == {}


def test_runtime_state_assigns_monotonic_state_ids() -> None:
    topology = _topology()
    config = _config()

    first = RuntimeState(config, topology)
    second = RuntimeState(config, topology)

    assert second.state_id > first.state_id


def test_runtime_state_tracks_current_request() -> None:
    state = RuntimeState(_config(), _topology())
    request = _request(3)

    state.set_current_request(request)

    assert state.current_request == request


def test_apply_provision_updates_canonical_state_and_invariants() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    request = _request(0)
    path = topology.get_paths("1", "3")[0]

    active_service = state.apply_provision(
        request=request,
        path=path,
        service_slot_start=4,
        service_num_slots=3,
        occupied_slot_start=4,
        occupied_slot_end_exclusive=8,
    )

    assert active_service.service_id == 0
    assert set(state.active_services_by_id) == {0}
    assert state.service_link_indices[0] == path.link_ids
    assert state.service_slot_ranges[0] == (4, 7)
    assert state.service_occupied_ranges[0] == (4, 8)
    assert state.global_state_version == 1
    assert tuple(state.link_versions[link_id] for link_id in path.link_ids) == (1, 1)
    for link_id in path.link_ids:
        assert 0 in state.link_active_service_ids[link_id]
        assert (state.slot_allocation[link_id, 4:8] == 0).all()
    assert state.release_queue_snapshot() == ((request.release_time, request.service_id),)
    state.validate_invariants()


def test_apply_provision_accepts_canonical_path_for_reverse_request() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    reverse_request = ServiceRequest(
        request_index=9,
        service_id=9,
        source_id=2,
        destination_id=0,
        bit_rate=40,
        arrival_time=12.0,
        holding_time=5.0,
    )
    canonical_path = topology.get_paths("1", "3")[0]

    active_service = state.apply_provision(
        request=reverse_request,
        path=canonical_path,
        service_slot_start=4,
        service_num_slots=3,
        occupied_slot_start=4,
        occupied_slot_end_exclusive=8,
    )

    assert active_service.request == reverse_request
    assert active_service.path == canonical_path
    state.validate_invariants()


def test_apply_release_clears_slots_and_queue() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    request = _request(1)
    path = topology.get_paths("1", "3")[0]

    state.apply_provision(
        request=request,
        path=path,
        service_slot_start=2,
        service_num_slots=2,
        occupied_slot_start=2,
        occupied_slot_end_exclusive=5,
    )
    released = state.apply_release(request.service_id)

    assert released.service_id == request.service_id
    assert state.active_services_by_id == {}
    assert state.service_link_indices == {}
    assert state.service_slot_ranges == {}
    assert state.service_occupied_ranges == {}
    assert state.release_queue_snapshot() == ()
    for link_id in path.link_ids:
        assert request.service_id not in state.link_active_service_ids[link_id]
        assert (state.slot_allocation[link_id, 2:5] == -1).all()
    state.validate_invariants()


def test_apply_release_uses_lazy_deletion_for_release_heap() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    path = topology.get_paths("1", "3")[0]
    first = _request(10)
    second = _request(11)

    state.apply_provision(
        request=first,
        path=path,
        service_slot_start=2,
        service_num_slots=2,
        occupied_slot_start=2,
        occupied_slot_end_exclusive=5,
    )
    state.apply_provision(
        request=second,
        path=path,
        service_slot_start=6,
        service_num_slots=2,
        occupied_slot_start=6,
        occupied_slot_end_exclusive=9,
    )

    state.apply_release(first.service_id)

    assert len(state.release_queue) == 2
    assert state.release_queue_snapshot() == ((second.release_time, second.service_id),)

    released = state.advance_time_and_release_due_services(second.release_time)

    assert [service.service_id for service in released] == [second.service_id]
    assert state.release_queue_snapshot() == ()
    state.validate_invariants()


def test_advance_time_releases_due_services() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    request = _request(2)
    path = topology.get_paths("1", "3")[0]

    state.apply_provision(
        request=request,
        path=path,
        service_slot_start=1,
        service_num_slots=2,
        occupied_slot_start=1,
        occupied_slot_end_exclusive=4,
    )
    released = state.advance_time_and_release_due_services(request.release_time)

    assert state.current_time == pytest.approx(request.release_time)
    assert [service.service_id for service in released] == [request.service_id]
    assert state.active_services_by_id == {}
    state.validate_invariants()


def test_apply_qot_updates_changes_only_targeted_service() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    request = _request(4)
    path = topology.get_paths("1", "3")[0]

    state.apply_provision(
        request=request,
        path=path,
        service_slot_start=6,
        service_num_slots=2,
        occupied_slot_start=6,
        occupied_slot_end_exclusive=9,
    )
    state.apply_qot_updates({4: {"osnr": 17.3, "ase": 20.7, "nli": 18.8}})

    service = state.active_services_by_id[4]
    assert service.osnr == pytest.approx(17.3)
    assert service.ase == pytest.approx(20.7)
    assert service.nli == pytest.approx(18.8)


def test_apply_disruption_moves_service_to_limbo_and_clears_runtime_state() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    request = _request(5)
    path = topology.get_paths("1", "3")[0]

    state.apply_provision(
        request=request,
        path=path,
        service_slot_start=6,
        service_num_slots=2,
        occupied_slot_start=6,
        occupied_slot_end_exclusive=9,
    )
    state.apply_qot_updates({5: {"osnr": 17.3, "ase": 20.7, "nli": 18.8}})

    disrupted = state.apply_disruption(5, terminal=True)

    assert disrupted.service_id == 5
    assert disrupted.is_disrupted is True
    assert disrupted.disruption_count == 1
    assert 5 not in state.active_services_by_id
    assert state.disrupted_services_by_id[5] is disrupted
    assert 5 not in state.service_qot_by_id
    assert state.release_queue_snapshot() == ()
    for link_id in path.link_ids:
        assert 5 not in state.link_active_service_ids[link_id]
        assert (state.slot_allocation[link_id, 6:9] == -1).all()
    state.validate_invariants()


def test_provision_rejects_slot_collision() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    path = topology.get_paths("1", "3")[0]

    state.apply_provision(
        request=_request(0),
        path=path,
        service_slot_start=3,
        service_num_slots=2,
        occupied_slot_start=3,
        occupied_slot_end_exclusive=6,
    )

    with pytest.raises(ValueError, match="occupied"):
        state.apply_provision(
            request=_request(1),
            path=path,
            service_slot_start=4,
            service_num_slots=2,
            occupied_slot_start=4,
            occupied_slot_end_exclusive=7,
        )
