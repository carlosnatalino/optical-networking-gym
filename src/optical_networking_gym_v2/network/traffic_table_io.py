from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from optical_networking_gym_v2.contracts import TrafficRecord, TrafficTable


def write_traffic_table_jsonl(
    file_path: str | Path,
    table: TrafficTable,
    records: tuple[TrafficRecord, ...] | list[TrafficRecord],
) -> Path:
    path = Path(file_path)
    payload_lines = [
        json.dumps({"record_type": "traffic_table", **asdict(table)}, separators=(",", ":"))
    ]
    for record in records:
        payload_lines.append(
            json.dumps({"record_type": "traffic_record", **asdict(record)}, separators=(",", ":"))
        )
    path.write_text("\n".join(payload_lines) + "\n", encoding="utf-8")
    return path


def read_traffic_table_jsonl(file_path: str | Path) -> tuple[TrafficTable, tuple[TrafficRecord, ...]]:
    path = Path(file_path)
    manifest: TrafficTable | None = None
    records: list[TrafficRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record_type = payload.pop("record_type", None)
            if record_type == "traffic_table":
                if manifest is not None:
                    raise ValueError("traffic table JSONL contains more than one manifest")
                manifest = TrafficTable(**payload)
            elif record_type == "traffic_record":
                if manifest is None:
                    raise ValueError("traffic table JSONL must declare the manifest before any record")
                records.append(TrafficRecord(**payload))
            else:
                raise ValueError("traffic table JSONL contains an unknown record type")
    if manifest is None:
        raise ValueError("traffic table JSONL is missing the manifest record")
    if manifest.request_count != len(records):
        raise ValueError("traffic table JSONL request_count does not match the number of records")
    return manifest, tuple(records)


__all__ = ["read_traffic_table_jsonl", "write_traffic_table_jsonl"]
