from __future__ import annotations

from pathlib import Path

import numpy as np

from optical_networking_gym_v2 import RuntimeState, ScenarioConfig, ServiceRequest, TopologyModel
from optical_networking_gym_v2.network import (
    available_slots_for_path,
    build_first_fit_allocation,
    candidate_starts,
    compute_required_slots,
    occupied_slot_range,
    path_is_free,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RING_4_PATH = PROJECT_ROOT / "optical_networking_gym_v2" / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="allocation_ring_4",
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
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def test_available_slots_for_path_intersects_all_links() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    path = topology.get_paths("1", "3")[0]

    state.slot_allocation[path.link_ids[0], 3:6] = 101
    state.slot_allocation[path.link_ids[1], 8:10] = 202

    available = available_slots_for_path(state, path)

    assert available.dtype == np.bool_
    assert available.shape == (24,)
    assert not available[3]
    assert not available[4]
    assert not available[5]
    assert not available[8]
    assert not available[9]
    assert available[2]
    assert available[6]
    assert available[10]


def test_candidate_starts_reserve_trailing_guard_band_inside_spectrum() -> None:
    available = np.array([False, True, True, True, True, True, False, False], dtype=np.bool_)

    candidates = candidate_starts(available, required_slots=3, total_slots=available.size)

    assert candidates == (1, 2)


def test_candidate_starts_allow_exact_fit_when_block_reaches_spectrum_end() -> None:
    available = np.array([False, False, False, False, True, True, True], dtype=np.bool_)

    candidates = candidate_starts(available, required_slots=3, total_slots=available.size)

    assert candidates == (4,)


def test_occupied_slot_range_matches_v1_right_guard_band_rule() -> None:
    assert occupied_slot_range(service_slot_start=4, service_num_slots=3, total_slots=12) == (4, 8)
    assert occupied_slot_range(service_slot_start=9, service_num_slots=3, total_slots=12) == (9, 12)


def test_path_is_free_checks_guard_band_on_every_link() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    path = topology.get_paths("1", "3")[0]

    state.slot_allocation[path.link_ids[1], 6] = 900

    assert not path_is_free(state, path, service_slot_start=4, service_num_slots=2)
    assert path_is_free(state, path, service_slot_start=1, service_num_slots=2)


def test_compute_required_slots_uses_ceil_like_legacy_get_number_slots() -> None:
    assert compute_required_slots(bit_rate=100, spectral_efficiency=4.0, channel_width=12.5) == 2
    assert compute_required_slots(bit_rate=75, spectral_efficiency=4.0, channel_width=12.5) == 2


def test_build_first_fit_allocation_returns_accepted_contract() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    path = topology.get_paths("1", "3")[0]

    state.slot_allocation[path.link_ids[0], 0:2] = 33
    state.slot_allocation[path.link_ids[1], 0:2] = 33

    allocation = build_first_fit_allocation(
        state,
        path=path,
        path_index=0,
        modulation_index=1,
        service_num_slots=2,
    )

    assert allocation.accepted is True
    assert allocation.path_index == 0
    assert allocation.modulation_index == 1
    assert allocation.service_slot_start == 2
    assert allocation.service_num_slots == 2
    assert allocation.occupied_slot_start == 2
    assert allocation.occupied_slot_end_exclusive == 5


def test_build_first_fit_allocation_returns_blocked_resources_when_no_candidate_exists() -> None:
    topology = _topology()
    state = RuntimeState(_config(), topology)
    path = topology.get_paths("1", "3")[0]

    for link_id in path.link_ids:
        state.slot_allocation[link_id, :] = 55

    allocation = build_first_fit_allocation(
        state,
        path=path,
        path_index=0,
        modulation_index=0,
        service_num_slots=2,
    )

    assert allocation.accepted is False
    assert allocation.status.value == "blocked_resources"
