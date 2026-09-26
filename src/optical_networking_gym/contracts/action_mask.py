from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActionSelection:
    path_index: int
    modulation_index: int
    initial_slot: int


__all__ = ["ActionSelection"]
