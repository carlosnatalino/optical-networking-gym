from __future__ import annotations

import numpy as np

from optical_networking_gym_v2.network import available_slots_for_path, candidate_starts

from .helpers import build_legacy_qrmsa_env, build_ring4_config, build_ring4_topology_v2
from optical_networking_gym_v2 import RuntimeState


def test_available_slots_and_candidates_match_legacy_qrmsa_helpers() -> None:
    topology = build_ring4_topology_v2(k_paths=2)
    state = RuntimeState(build_ring4_config(scenario_id="allocation_equivalence"), topology)
    path = topology.get_paths("1", "3")[0]

    for link_id in path.link_ids:
        state.slot_allocation[link_id, 3:6] = 71
    state.slot_allocation[path.link_ids[0], 10:12] = 72

    legacy_env = build_legacy_qrmsa_env(k_paths=2)
    legacy_path = legacy_env.k_shortest_paths["1", "3"][0]
    for index in range(len(legacy_path.node_list) - 1):
        link_index = legacy_env.topology[legacy_path.node_list[index]][legacy_path.node_list[index + 1]]["index"]
        legacy_env.topology.graph["available_slots"][link_index, 3:6] = 0
    first_link_index = legacy_env.topology[legacy_path.node_list[0]][legacy_path.node_list[1]]["index"]
    legacy_env.topology.graph["available_slots"][first_link_index, 10:12] = 0

    new_available = available_slots_for_path(state, path)
    legacy_available = legacy_env.get_available_slots(legacy_path).astype(bool)

    assert np.array_equal(new_available, legacy_available)
    assert candidate_starts(new_available, required_slots=2, total_slots=24) == tuple(
        legacy_env._get_candidates(legacy_available.astype(np.int32), 2, 24)
    )
