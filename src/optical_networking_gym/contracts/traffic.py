from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

from .enums import TrafficMode


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_number(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_number(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class ServiceRequest:
    request_index: int
    service_id: int
    source_id: int
    destination_id: int
    bit_rate: int
    arrival_time: float
    holding_time: float
    traffic_mode: TrafficMode = TrafficMode.DYNAMIC
    traffic_origin: str | None = None
    table_row_index: int | None = None
    table_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("request_index", self.request_index)
        _require_non_negative_int("service_id", self.service_id)
        _require_positive_number("bit_rate", self.bit_rate)
        _require_non_negative_number("arrival_time", self.arrival_time)
        _require_positive_number("holding_time", self.holding_time)
        if self.source_id == self.destination_id:
            raise ValueError("source_id and destination_id must differ")
        if self.table_row_index is not None:
            _require_non_negative_int("table_row_index", self.table_row_index)
        if self.traffic_mode is TrafficMode.STATIC:
            if self.traffic_origin is None:
                object.__setattr__(self, "traffic_origin", "traffic_table")
            if self.table_row_index is None:
                object.__setattr__(self, "table_row_index", self.request_index)

    @property
    def release_time(self) -> float:
        # Legacy V1 schedules releases using float32-valued service times.
        # Match that precision so static replay and trace parity stay exact.
        return float(np.float32(self.arrival_time + self.holding_time))


@dataclass(frozen=True, slots=True)
class TrafficRecord:
    request_index: int
    service_id: int
    source_id: int
    destination_id: int
    bit_rate: int
    arrival_time: float
    holding_time: float
    table_id: str
    row_index: int | None = None
    source_label: str | None = None
    destination_label: str | None = None
    bit_rate_class: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int("request_index", self.request_index)
        _require_non_negative_int("service_id", self.service_id)
        _require_positive_number("bit_rate", self.bit_rate)
        _require_non_negative_number("arrival_time", self.arrival_time)
        _require_positive_number("holding_time", self.holding_time)
        if self.source_id == self.destination_id:
            raise ValueError("source_id and destination_id must differ")
        if not self.table_id:
            raise ValueError("table_id must be a non-empty string")
        if self.row_index is None:
            object.__setattr__(self, "row_index", self.request_index)
        else:
            _require_non_negative_int("row_index", self.row_index)

    def to_service_request(self) -> ServiceRequest:
        return ServiceRequest(
            request_index=self.request_index,
            service_id=self.service_id,
            source_id=self.source_id,
            destination_id=self.destination_id,
            bit_rate=self.bit_rate,
            arrival_time=self.arrival_time,
            holding_time=self.holding_time,
            traffic_mode=TrafficMode.STATIC,
            traffic_origin="traffic_table",
            table_row_index=self.row_index,
            table_id=self.table_id,
        )


@dataclass(frozen=True, slots=True)
class TrafficTable:
    traffic_table_version: str
    table_id: str
    scenario_id: str
    topology_id: str
    traffic_mode_source: str
    request_count: int
    time_unit: str
    bit_rate_unit: str
    seed: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "traffic_table_version",
            "table_id",
            "scenario_id",
            "topology_id",
            "traffic_mode_source",
            "time_unit",
            "bit_rate_unit",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be a non-empty string")
        _require_non_negative_int("request_count", self.request_count)
        if self.seed is not None:
            _require_non_negative_int("seed", self.seed)
