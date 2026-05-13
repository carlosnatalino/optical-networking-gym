from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import TrafficRecord, TrafficTable
from optical_networking_gym_v2.network import read_traffic_table_jsonl, write_traffic_table_jsonl


def _table() -> TrafficTable:
    return TrafficTable(
        traffic_table_version="v1",
        table_id="table_ring_4",
        scenario_id="traffic_fixture",
        topology_id="ring_4",
        traffic_mode_source="dynamic_export",
        request_count=2,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
        seed=123,
    )


def _records() -> tuple[TrafficRecord, ...]:
    return (
        TrafficRecord(
            request_index=0,
            service_id=0,
            source_id=0,
            destination_id=1,
            bit_rate=40,
            arrival_time=1.0,
            holding_time=4.0,
            table_id="table_ring_4",
            row_index=0,
            source_label="1",
            destination_label="2",
        ),
        TrafficRecord(
            request_index=1,
            service_id=1,
            source_id=1,
            destination_id=3,
            bit_rate=100,
            arrival_time=2.5,
            holding_time=5.0,
            table_id="table_ring_4",
            row_index=1,
            source_label="2",
            destination_label="4",
        ),
    )


def test_write_and_read_traffic_table_jsonl_roundtrip(tmp_path: Path) -> None:
    file_path = tmp_path / "ring_4_requests.jsonl"

    write_traffic_table_jsonl(file_path, _table(), _records())
    table, records = read_traffic_table_jsonl(file_path)

    assert table == _table()
    assert records == _records()


def test_read_traffic_table_jsonl_rejects_missing_manifest(tmp_path: Path) -> None:
    file_path = tmp_path / "broken.jsonl"
    file_path.write_text('{"record_type":"traffic_record","request_index":0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        read_traffic_table_jsonl(file_path)
