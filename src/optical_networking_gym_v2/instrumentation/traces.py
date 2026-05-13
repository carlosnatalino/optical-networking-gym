from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def write_step_trace_jsonl(trace_payload: Mapping[str, object], file_path: str | Path) -> Path:
    resolved_path = Path(file_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(trace_payload["header"], separators=(",", ":")) + "\n")
        for step in trace_payload["steps"]:
            handle.write(json.dumps(step, separators=(",", ":")) + "\n")
        handle.write(json.dumps(trace_payload["footer"], separators=(",", ":")) + "\n")
    return resolved_path


__all__ = ["write_step_trace_jsonl"]
