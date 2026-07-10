from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-analysis view of the lazy export table below. Kept lazy at
    # runtime because eager subpackage imports create import cycles
    # (e.g. contracts -> network -> runtime -> contracts and
    # optical -> envs -> runtime -> features -> optical).
    from .allocation import (
        available_slots_for_path,
        build_first_fit_allocation,
        candidate_starts,
        compute_required_slots,
        occupied_slot_range,
        path_is_free,
    )
    from .topology import Link, PathRecord, Span, TopologyModel
    from .traffic_table_io import read_traffic_table_jsonl, write_traffic_table_jsonl

_EXPORTS: dict[str, tuple[str, str]] = {
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

__all__ = [
    "Link",
    "PathRecord",
    "Span",
    "TopologyModel",
    "available_slots_for_path",
    "build_first_fit_allocation",
    "candidate_starts",
    "compute_required_slots",
    "occupied_slot_range",
    "path_is_free",
    "read_traffic_table_jsonl",
    "write_traffic_table_jsonl",
]


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
