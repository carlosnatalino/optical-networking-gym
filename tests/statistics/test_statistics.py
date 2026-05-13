from __future__ import annotations

import pytest

from optical_networking_gym_v2 import (
    Allocation,
    Modulation,
    ScenarioConfig,
    ServiceRequest,
    Statistics,
    Status,
    StepTransition,
)


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="statistics",
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


def test_statistics_accumulates_counters_rates_and_histogram() -> None:
    statistics = Statistics(_config())

    statistics.record_transition(
        StepTransition.accept(
            request=_request(1, bit_rate=40),
            allocation=Allocation.accept(
                path_index=0,
                modulation_index=0,
                service_slot_start=2,
                service_num_slots=2,
                occupied_slot_start=2,
                occupied_slot_end_exclusive=5,
            ),
            modulation_spectral_efficiency=2,
            disrupted_services=1,
            osnr=15.0,
            osnr_requirement=6.72,
        )
    )
    statistics.record_transition(
        StepTransition(
            request=_request(2, bit_rate=100),
            allocation=Allocation.reject(Status.BLOCKED_RESOURCES),
        )
    )
    statistics.record_transition(
        StepTransition(
            request=_request(3, bit_rate=200),
            allocation=Allocation.reject(Status.BLOCKED_QOT),
        )
    )
    statistics.record_transition(
        StepTransition(
            request=_request(4, bit_rate=80),
            allocation=Allocation.reject(Status.REJECTED_BY_AGENT),
        )
    )

    snapshot = statistics.snapshot()

    assert snapshot.services_processed == 4
    assert snapshot.services_accepted == 1
    assert snapshot.services_blocked_resources == 1
    assert snapshot.services_blocked_qot == 1
    assert snapshot.services_dropped_qot == 0
    assert snapshot.services_rejected_by_agent == 1
    assert snapshot.services_served == 1
    assert snapshot.service_blocking_rate == pytest.approx(0.75)
    assert snapshot.service_served_rate == pytest.approx(0.25)
    assert snapshot.bit_rate_blocking_rate == pytest.approx((420 - 40) / 420)
    assert snapshot.disrupted_services_rate == pytest.approx(1.0)
    assert snapshot.episode_modulation_histogram == ((2, 1), (4, 0))


def test_statistics_tracks_dropped_qot_separately_from_admission_blocks() -> None:
    statistics = Statistics(_config())
    statistics.record_transition(
        StepTransition.accept(
            request=_request(30, bit_rate=40),
            allocation=Allocation.accept(
                path_index=0,
                modulation_index=0,
                service_slot_start=2,
                service_num_slots=2,
                occupied_slot_start=2,
                occupied_slot_end_exclusive=5,
            ),
            modulation_spectral_efficiency=2,
            disrupted_services=1,
            osnr=15.0,
            osnr_requirement=6.72,
        )
    )
    statistics.record_dropped_qot(1)

    snapshot = statistics.snapshot()

    assert snapshot.services_blocked_qot == 0
    assert snapshot.services_dropped_qot == 1
    assert snapshot.services_served == 0
    assert snapshot.episode_services_blocked_qot == 0
    assert snapshot.episode_services_dropped_qot == 1
    assert snapshot.episode_services_served == 0
    assert snapshot.disrupted_services == 1
    assert snapshot.service_served_rate == pytest.approx(0.0)
    assert snapshot.episode_service_served_rate == pytest.approx(0.0)


def test_statistics_reset_episode_preserves_totals_and_clears_episode_counters() -> None:
    statistics = Statistics(_config())
    statistics.record_transition(
        StepTransition.accept(
            request=_request(10, bit_rate=40),
            allocation=Allocation.accept(
                path_index=0,
                modulation_index=0,
                service_slot_start=1,
                service_num_slots=2,
                occupied_slot_start=1,
                occupied_slot_end_exclusive=4,
            ),
            modulation_spectral_efficiency=4,
        )
    )

    statistics.reset_episode()
    snapshot = statistics.snapshot()

    assert snapshot.services_processed == 1
    assert snapshot.services_accepted == 1
    assert snapshot.episode_services_processed == 0
    assert snapshot.episode_services_accepted == 0
    assert snapshot.services_served == 1
    assert snapshot.episode_services_served == 0
    assert snapshot.episode_bit_rate_requested == 0.0
    assert snapshot.episode_modulation_histogram == ((2, 0), (4, 0))


def test_statistics_validate_invariants_rejects_inconsistent_internal_state() -> None:
    statistics = Statistics(_config())
    statistics.record_transition(
        StepTransition(
            request=_request(20),
            allocation=Allocation.reject(Status.BLOCKED_RESOURCES),
        )
    )
    statistics._services_processed = 0

    with pytest.raises(AssertionError, match="processed"):
        statistics.validate_invariants()


def test_statistics_validate_invariants_rejects_negative_served_count() -> None:
    statistics = Statistics(_config())
    statistics.record_transition(
        StepTransition.accept(
            request=_request(40),
            allocation=Allocation.accept(
                path_index=0,
                modulation_index=0,
                service_slot_start=1,
                service_num_slots=2,
                occupied_slot_start=1,
                occupied_slot_end_exclusive=4,
            ),
            modulation_spectral_efficiency=2,
        )
    )
    statistics.record_dropped_qot(2)

    with pytest.raises(AssertionError, match="served"):
        statistics.validate_invariants()
