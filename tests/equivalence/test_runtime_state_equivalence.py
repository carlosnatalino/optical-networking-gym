from __future__ import annotations

import numpy as np

from optical_networking_gym_v2 import RuntimeState

from .helpers import (
    build_legacy_qrmsa_env,
    build_legacy_service,
    build_ring4_config,
    build_ring4_topology_v2,
    build_service_request,
)


def test_runtime_state_provision_matches_legacy_slot_occupancy() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    state = RuntimeState(build_ring4_config(scenario_id="runtime_equivalence"), topology)
    path = topology.get_paths("1", "3")[0]
    request = build_service_request(service_id=10)

    state.apply_provision(
        request=request,
        path=path,
        service_slot_start=4,
        service_num_slots=3,
        occupied_slot_start=4,
        occupied_slot_end_exclusive=8,
    )

    legacy_env = build_legacy_qrmsa_env(k_paths=2)
    legacy_service = build_legacy_service(request, legacy_env)
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_env.current_service = legacy_service
    legacy_env._provision_path(legacy_path, 4, 3)
    legacy_env._add_release(legacy_service)

    assert np.array_equal(state.slot_allocation == -1, legacy_env.topology.graph["available_slots"] == 1)
    assert state.release_queue_snapshot() == ((request.release_time, request.service_id),)
    assert sorted(state.active_services_by_id) == [request.service_id]
    assert [service.service_id for service in legacy_env.topology.graph["running_services"]] == [request.service_id]


def test_runtime_state_release_matches_legacy_slot_cleanup() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    state = RuntimeState(build_ring4_config(scenario_id="runtime_equivalence_release"), topology)
    path = topology.get_paths("1", "3")[0]
    request = build_service_request(service_id=11)

    active = state.apply_provision(
        request=request,
        path=path,
        service_slot_start=2,
        service_num_slots=2,
        occupied_slot_start=2,
        occupied_slot_end_exclusive=5,
    )
    state.apply_release(request.service_id)

    legacy_env = build_legacy_qrmsa_env(k_paths=2)
    legacy_service = build_legacy_service(request, legacy_env)
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_env.current_service = legacy_service
    legacy_env._provision_path(legacy_path, 2, 2)
    legacy_env._release_path(legacy_service)

    assert active.service_id == request.service_id
    assert np.array_equal(state.slot_allocation == -1, legacy_env.topology.graph["available_slots"] == 1)
    assert state.release_queue_snapshot() == ()
    assert state.active_services_by_id == {}
    assert legacy_env.topology.graph["running_services"] == []
