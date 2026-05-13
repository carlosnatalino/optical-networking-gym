from __future__ import annotations

import importlib.util
from pathlib import Path

from optical_networking_gym_v2 import TrafficRecord, TrafficTable, write_traffic_table_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = PROJECT_ROOT / "optical_networking_gym_v2" / "examples" / "static_first_fit_trace.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("static_first_fit_trace_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_static_first_fit_trace_example_runs_with_small_static_table(tmp_path: Path) -> None:
    module = _load_module()
    traffic_path = tmp_path / "ring4_static.jsonl"
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="ring4_static_test",
        scenario_id="ring4_static_example_test",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=4,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
        seed=7,
    )
    records = (
        TrafficRecord(0, 0, 0, 1, 100, 0.5, 10.0, table.table_id, row_index=0),
        TrafficRecord(1, 1, 1, 2, 40, 1.0, 12.0, table.table_id, row_index=1),
        TrafficRecord(2, 2, 2, 3, 100, 1.5, 8.0, table.table_id, row_index=2),
        TrafficRecord(3, 3, 3, 0, 10, 2.0, 6.0, table.table_id, row_index=3),
    )
    write_traffic_table_jsonl(traffic_path, table, records)

    summary, trace = module.run_episode(
        traffic_table_path=traffic_path,
        topology_name="ring_4",
        seed=7,
        episode_length=4,
        num_spectrum_resources=24,
        k_paths=2,
    )

    assert summary["topology_name"] == "ring_4"
    assert summary["steps"] == 4
    assert trace["header"]["topology_id"] == "ring_4"
    assert trace["footer"]["steps"] == 4
