from __future__ import annotations

import pytest

from optical_networking_gym_v2 import (
    ServiceRequest,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)


def test_service_request_exposes_release_time() -> None:
    request = ServiceRequest(
        request_index=3,
        service_id=9,
        source_id=1,
        destination_id=4,
        bit_rate=100,
        arrival_time=12.5,
        holding_time=4.25,
    )

    assert request.traffic_mode is TrafficMode.DYNAMIC
    assert request.release_time == pytest.approx(16.75)


def test_service_request_release_time_matches_legacy_float32_precision() -> None:
    request = ServiceRequest(
        request_index=0,
        service_id=0,
        source_id=4,
        destination_id=3,
        bit_rate=400,
        arrival_time=73.681922912598,
        holding_time=18342.84765625,
    )

    assert request.release_time == pytest.approx(18416.529296875)


def test_service_request_rejects_invalid_endpoints() -> None:
    with pytest.raises(ValueError, match="source_id and destination_id"):
        ServiceRequest(
            request_index=0,
            service_id=1,
            source_id=2,
            destination_id=2,
            bit_rate=40,
            arrival_time=1.0,
            holding_time=2.0,
        )


def test_traffic_table_row_converts_to_service_request() -> None:
    row = TrafficRecord(
        request_index=7,
        service_id=17,
        source_id=0,
        destination_id=5,
        bit_rate=40,
        arrival_time=3.0,
        holding_time=11.0,
        table_id="static_fixture_v1",
        row_index=7,
    )

    request = row.to_service_request()

    assert request.traffic_mode is TrafficMode.STATIC
    assert request.traffic_origin == "traffic_table"
    assert request.table_id == "static_fixture_v1"
    assert request.table_row_index == 7
    assert request.release_time == pytest.approx(14.0)


def test_traffic_table_manifest_validates_request_count() -> None:
    with pytest.raises(ValueError, match="request_count"):
        TrafficTable(
            traffic_table_version="v1",
            table_id="fixture_1",
            scenario_id="static_ring_hand_smoke_v1",
            topology_id="ring_4",
            traffic_mode_source="hand_authored",
            request_count=-1,
            time_unit="hours",
            bit_rate_unit="Gbps",
        )
