from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Link": (".topology", "Link"),
    "PathRecord": (".topology", "PathRecord"),
    "Span": (".topology", "Span"),
    "TopologyModel": (".topology", "TopologyModel"),
    "available_slots_for_path": (".allocation", "available_slots_for_path"),
    "build_first_fit_allocation": (".allocation", "build_first_fit_allocation"),
    "candidate_starts": (".allocation", "candidate_starts"),
    "compute_required_slots": (".allocation", "compute_required_slots"),
    "occupied_slot_range": (".allocation", "occupied_slot_range"),
    "path_is_free": (".allocation", "path_is_free"),
    "read_traffic_table_jsonl": (".traffic_table_io", "read_traffic_table_jsonl"),
    "write_traffic_table_jsonl": (".traffic_table_io", "write_traffic_table_jsonl"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
