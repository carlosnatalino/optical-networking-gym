from __future__ import annotations

import numpy as np

from optical_networking_gym_v2 import (
    Allocation,
    Modulation,
    RewardBreakdown,
    RewardProfile,
    ScenarioConfig,
    ServiceRequest,
    Statistics,
    Status,
    StepInfo,
    StepTransition,
)


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="step_info",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )


def _request(service_id: int, *, bit_rate: int = 40) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=2,
        bit_rate=bit_rate,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def test_step_info_builds_legacy_compatible_rates_and_debug_fields() -> None:
    config = _config()
    statistics = Statistics(config)
    transition = StepTransition.accept(
        request=_request(1, bit_rate=40),
        allocation=Allocation.accept(
            path_index=1,
            modulation_index=0,
            service_slot_start=3,
            service_num_slots=2,
            occupied_slot_start=3,
            occupied_slot_end_exclusive=6,
        ),
        modulation_spectral_efficiency=2,
        osnr=15.0,
        osnr_requirement=6.72,
        disrupted_services=1,
        fragmentation_shannon_entropy=0.4,
        fragmentation_route_cuts=3.0,
        fragmentation_route_rss=0.75,
        mask=np.array([1, 0, 1], dtype=np.uint8),
    )
    statistics.record_transition(transition)

    info = StepInfo(config).build(statistics.snapshot(), transition)

    assert info["accepted"] is True
    assert info["status"] == "accepted"
    assert info["episode_services_accepted"] == 1
    assert info["services_served"] == 1
    assert info["episode_services_served"] == 1
    assert info["service_blocking_rate"] == 0.0
    assert info["episode_service_blocking_rate"] == 0.0
    assert info["service_served_rate"] == 1.0
    assert info["episode_service_served_rate"] == 1.0
    assert info["bit_rate_blocking_rate"] == 0.0
    assert info["episode_bit_rate_blocking_rate"] == 0.0
    assert info["disrupted_services"] == 1.0
    assert info["episode_disrupted_services"] == 1.0
    assert info["osnr"] == 15.0
    assert info["osnr_req"] == 6.72
    assert info["chosen_path_index"] == 1
    assert info["chosen_slot"] == 3
    assert info["chosen_modulation_index"] == 0
    assert info["fragmentation_shannon_entropy"] == 0.4
    assert info["fragmentation_route_cuts"] == 3.0
    assert info["fragmentation_route_rss"] == 0.75
    assert np.array_equal(info["mask"], np.array([1, 0, 1], dtype=np.uint8))
    assert info["modulation_2"] == 1
    assert info["modulation_4"] == 0


def test_step_info_adds_terminal_block_counters() -> None:
    config = _config()
    statistics = Statistics(config)
    statistics.record_transition(
        StepTransition.accept(
            request=_request(1),
            allocation=Allocation.accept(
                path_index=0,
                modulation_index=0,
                service_slot_start=2,
                service_num_slots=2,
                occupied_slot_start=2,
                occupied_slot_end_exclusive=5,
            ),
            modulation_spectral_efficiency=2,
        )
    )
    statistics.record_transition(
        StepTransition(
            request=_request(2),
            allocation=Allocation.reject(Status.BLOCKED_RESOURCES),
        )
    )
    statistics.record_transition(
        StepTransition(
            request=_request(3),
            allocation=Allocation.reject(Status.BLOCKED_QOT),
        )
    )
    transition = StepTransition(
        request=_request(4),
        allocation=Allocation.reject(Status.REJECTED_BY_AGENT),
    )
    statistics.record_transition(transition)
    statistics.record_dropped_qot(1)

    info = StepInfo(config).build(statistics.snapshot(), transition, terminated=True)

    assert info["accepted"] is False
    assert info["status"] == "rejected_by_agent"
    assert info["services_served"] == 0
    assert info["episode_services_served"] == 0
    assert info["service_served_rate"] == 0.0
    assert info["episode_service_served_rate"] == 0.0
    assert info["blocked_due_to_resources"] == 1
    assert info["blocked_due_to_osnr"] == 1
    assert info["disrupted_or_dropped_services"] == 1
    assert info["rejected"] == 1


def test_step_info_includes_reward_breakdown_when_provided() -> None:
    config = _config()
    statistics = Statistics(config)
    transition = StepTransition(
        request=_request(5),
        allocation=Allocation.reject(Status.REJECTED_BY_AGENT),
    )
    statistics.record_transition(transition)
    breakdown = RewardBreakdown(
        profile=RewardProfile.BALANCED.value,
        raw_reward=-0.15,
        clipped_reward=-0.15,
        accept_component=0.0,
        spectral_efficiency_bonus=0.0,
        fragmentation_penalty=0.0,
        physical_penalty=0.0,
        reject_penalty=0.15,
    )

    info = StepInfo(config).build(
        statistics.snapshot(),
        transition,
        reward=-0.15,
        reward_breakdown=breakdown,
    )

    assert info["reward"] == -0.15
    assert info["reward_profile"] == "balanced"
    assert info["reward_raw"] == -0.15
    assert info["reward_clipped"] == -0.15
    assert info["reward_reject_penalty"] == 0.15


def test_step_info_accepts_live_statistics_without_snapshot() -> None:
    config = _config()
    statistics = Statistics(config)
    transition = StepTransition.accept(
        request=_request(6),
        allocation=Allocation.accept(
            path_index=0,
            modulation_index=0,
            service_slot_start=2,
            service_num_slots=2,
            occupied_slot_start=2,
            occupied_slot_end_exclusive=5,
        ),
        modulation_spectral_efficiency=2,
    )
    statistics.record_transition(transition)

    info = StepInfo(config).build(statistics, transition)

    assert info["accepted"] is True
    assert info["services_processed"] == 1
    assert info["episode_services_accepted"] == 1
    assert info["services_served"] == 1
    assert info["service_blocking_rate"] == 0.0
    assert info["service_served_rate"] == 1.0
    assert info["modulation_2"] == 1
