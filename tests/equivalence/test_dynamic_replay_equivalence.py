from __future__ import annotations

from optical_networking_gym_v2.bench import compare_simulator_episode_with_legacy


def test_simulator_matches_legacy_for_captured_dynamic_replay_episode() -> None:
    result = compare_simulator_episode_with_legacy(request_count=8, seed=7)

    assert result["topology_id"] == "ring_4"
    assert result["request_count"] == 8
    assert result["status_matches"] is True
    assert result["mask_matches"] is True
    assert result["slot_matches"] is True
    assert result["active_count_matches"] is True
    assert result["osnr_matches"] is True
    assert result["episode_services_accepted_matches"] is True
