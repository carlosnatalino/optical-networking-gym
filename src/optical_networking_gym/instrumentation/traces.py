from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping, cast


def write_step_trace_jsonl(trace_payload: Mapping[str, object], file_path: str | Path) -> Path:
    resolved_path = Path(file_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    steps = cast(Iterable[object], trace_payload["steps"])
    with resolved_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(trace_payload["header"], separators=(",", ":")) + "\n")
        for step in steps:
            handle.write(json.dumps(step, separators=(",", ":")) + "\n")
        handle.write(json.dumps(trace_payload["footer"], separators=(",", ":")) + "\n")
    return resolved_path


__all__ = ["write_step_trace_jsonl"]
