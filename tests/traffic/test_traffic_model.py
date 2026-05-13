from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import (
    ScenarioConfig,
    TopologyModel,
    TrafficModel,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)
from optical_networking_gym_v2.network import write_traffic_table_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RING_4_PATH = PROJECT_ROOT / "optical_networking_gym_v2" / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _ring_4_topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _dynamic_source() -> dict[str, object]:
    return {
        "bit_rates": (10, 40),
        "bit_rate_probabilities": (0.7, 0.3),
        "mean_holding_time": 10.0,
        "mean_inter_arrival_time": 2.0,
    }


def test_dynamic_traffic_is_seed_deterministic() -> None:
    topology = _ring_4_topology()
    config = ScenarioConfig(
        scenario_id="dynamic_seeded",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        seed=123,
        traffic_source=_dynamic_source(),
    )

    model_a = TrafficModel(config, topology)
    model_b = TrafficModel(config, topology)

    seq_a = [model_a.next_request() for _ in range(6)]
    seq_b = [model_b.next_request() for _ in range(6)]

    assert seq_a == seq_b
    assert [request.request_index for request in seq_a] == list(range(6))
    assert [request.service_id for request in seq_a] == list(range(6))


def test_dynamic_traffic_generates_valid_requests() -> None:
    topology = _ring_4_topology()
    config = ScenarioConfig(
        scenario_id="dynamic_valid",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        seed=7,
        traffic_source=_dynamic_source(),
    )

    model = TrafficModel(config, topology)
    requests = [model.next_request() for _ in range(12)]

    assert all(request.traffic_mode is TrafficMode.DYNAMIC for request in requests)
    assert all(request.source_id != request.destination_id for request in requests)
    assert all(request.bit_rate in {10, 40} for request in requests)
    assert [request.arrival_time for request in requests] == sorted(
        request.arrival_time for request in requests
    )


def test_static_traffic_replays_records_and_stops() -> None:
    topology = _ring_4_topology()
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="fixture_ring_static",
        scenario_id="static_fixture",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=2,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(
            request_index=0,
            service_id=0,
            source_id=0,
            destination_id=1,
            bit_rate=10,
            arrival_time=1.0,
            holding_time=4.0,
            table_id=table.table_id,
            row_index=0,
        ),
        TrafficRecord(
            request_index=1,
            service_id=1,
            source_id=1,
            destination_id=3,
            bit_rate=40,
            arrival_time=3.0,
            holding_time=5.0,
            table_id=table.table_id,
            row_index=1,
        ),
    )
    config = ScenarioConfig(
        scenario_id="static_fixture",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source={"table": table, "records": records},
    )

    model = TrafficModel(config, topology)
    first = model.next_request()
    second = model.next_request()

    assert first.traffic_mode is TrafficMode.STATIC
    assert first.table_id == table.table_id
    assert second.table_row_index == 1

    with pytest.raises(StopIteration):
        model.next_request()


def test_dynamic_export_supports_roundtrip_when_enabled() -> None:
    topology = _ring_4_topology()
    config = ScenarioConfig(
        scenario_id="dynamic_roundtrip",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        seed=99,
        traffic_source=_dynamic_source(),
    )

    dynamic_model = TrafficModel(config, topology, capture_table=True)
    generated = [dynamic_model.next_request() for _ in range(5)]
    table, records = dynamic_model.export_table()

    replay_config = ScenarioConfig(
        scenario_id="dynamic_roundtrip",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source={"table": table, "records": records},
    )
    replay_model = TrafficModel(replay_config, topology)
    replayed = [replay_model.next_request() for _ in range(5)]

    assert table.request_count == 5
    assert len(records) == 5
    assert [
        (
            request.request_index,
            request.service_id,
            request.source_id,
            request.destination_id,
            request.bit_rate,
            request.arrival_time,
            request.holding_time,
        )
        for request in generated
    ] == [
        (
            request.request_index,
            request.service_id,
            request.source_id,
            request.destination_id,
            request.bit_rate,
            request.arrival_time,
            request.holding_time,
        )
        for request in replayed
    ]


def test_export_requires_capture_mode() -> None:
    topology = _ring_4_topology()
    config = ScenarioConfig(
        scenario_id="dynamic_no_export",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        seed=11,
        traffic_source=_dynamic_source(),
    )

    model = TrafficModel(config, topology)
    model.next_request()

    with pytest.raises(RuntimeError, match="capture_table"):
        model.export_table()


def test_static_traffic_can_load_from_jsonl_path(tmp_path: Path) -> None:
    topology = _ring_4_topology()
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="fixture_ring_static_file",
        scenario_id="static_fixture_file",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=2,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(
            request_index=0,
            service_id=0,
            source_id=0,
            destination_id=1,
            bit_rate=10,
            arrival_time=1.0,
            holding_time=4.0,
            table_id=table.table_id,
            row_index=0,
        ),
        TrafficRecord(
            request_index=1,
            service_id=1,
            source_id=1,
            destination_id=3,
            bit_rate=40,
            arrival_time=3.0,
            holding_time=5.0,
            table_id=table.table_id,
            row_index=1,
        ),
    )
    file_path = tmp_path / "traffic_table.jsonl"
    write_traffic_table_jsonl(file_path, table, records)

    config = ScenarioConfig(
        scenario_id="static_fixture_file",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=file_path,
    )

    model = TrafficModel(config, topology)

    first = model.next_request()
    second = model.next_request()

    assert first.table_id == table.table_id
    assert second.table_row_index == 1


def test_dynamic_capture_can_persist_jsonl_table(tmp_path: Path) -> None:
    topology = _ring_4_topology()
    config = ScenarioConfig(
        scenario_id="dynamic_persisted",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        seed=77,
        traffic_source=_dynamic_source(),
    )
    model = TrafficModel(config, topology, capture_table=True)
    for _ in range(4):
        model.next_request()

    file_path = tmp_path / "captured.jsonl"
    model.save_table_jsonl(file_path)

    replay = TrafficModel(
        ScenarioConfig(
            scenario_id="dynamic_persisted",
            topology_id="ring_4",
            k_paths=2,
            num_spectrum_resources=24,
            traffic_mode=TrafficMode.STATIC,
            traffic_source=file_path,
        ),
        topology,
    )

    replayed = [replay.next_request() for _ in range(4)]
    assert [request.service_id for request in replayed] == [0, 1, 2, 3]
