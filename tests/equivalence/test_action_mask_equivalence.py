from __future__ import annotations

import numpy as np

from optical_networking_gym_v2 import ActionMask, QoTEngine, RuntimeState

from .helpers import (
    build_action_mask_modulations_v2,
    build_legacy_action_mask,
    build_legacy_qrmsa_env,
    build_legacy_service,
    build_ring4_config,
    build_ring4_topology_v2,
    build_service_request,
)


def test_action_mask_matches_legacy_on_empty_state() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    config = build_ring4_config(
        scenario_id="mask_equivalence_empty",
        modulations=build_action_mask_modulations_v2(),
        modulations_to_consider=2,
    )
    state = RuntimeState(config, topology)
    request = build_service_request(service_id=51)
    builder = ActionMask(config, topology, QoTEngine(config, topology))

    legacy_env = build_legacy_qrmsa_env(k_paths=2, gen_observation=False, reset=True)
    legacy_env.current_service = build_legacy_service(request, legacy_env)
    legacy_mask = build_legacy_action_mask(legacy_env)

    mask = builder.build(state, request)

    assert np.array_equal(mask, legacy_mask)


def test_action_mask_matches_legacy_after_existing_provision() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    config = build_ring4_config(
        scenario_id="mask_equivalence_loaded",
        modulations=build_action_mask_modulations_v2(),
        modulations_to_consider=2,
    )
    state = RuntimeState(config, topology)
    builder = ActionMask(config, topology, QoTEngine(config, topology))
    path = topology.get_paths("1", "3")[0]

    first_request = build_service_request(service_id=60)
    first_candidate = builder.qot_engine.build_candidate(
        request=first_request,
        path=path,
        modulation=config.modulations[0],
        service_slot_start=2,
        service_num_slots=2,
    )
    state.apply_provision(
        request=first_candidate.request,
        path=path,
        service_slot_start=first_candidate.service_slot_start,
        service_num_slots=first_candidate.service_num_slots,
        occupied_slot_start=first_candidate.service_slot_start,
        occupied_slot_end_exclusive=first_candidate.service_slot_start + first_candidate.service_num_slots + 1,
        modulation=first_candidate.modulation,
        center_frequency=first_candidate.center_frequency,
        bandwidth=first_candidate.bandwidth,
        launch_power=first_candidate.launch_power,
    )

    second_request = build_service_request(service_id=61)
    mask = builder.build(state, second_request)

    legacy_env = build_legacy_qrmsa_env(k_paths=2, gen_observation=False, reset=True)
    legacy_first = build_legacy_service(first_request, legacy_env)
    legacy_env.current_service = legacy_first
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    legacy_first.path = legacy_path
    legacy_first.initial_slot = 2
    legacy_first.number_slots = 2
    legacy_first.current_modulation = legacy_env.modulations[0]
    legacy_first.center_frequency = first_candidate.center_frequency
    legacy_first.bandwidth = first_candidate.bandwidth
    legacy_first.launch_power = first_candidate.launch_power
    legacy_env._provision_path(legacy_path, 2, 2)
    legacy_env.current_service = build_legacy_service(second_request, legacy_env)
    legacy_mask = build_legacy_action_mask(legacy_env)

    assert np.array_equal(mask, legacy_mask)
