from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import random

from optical_networking_gym_v2.contracts.enums import TrafficMode
from optical_networking_gym_v2.contracts.traffic import ServiceRequest, TrafficRecord, TrafficTable
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.network.traffic_table_io import read_traffic_table_jsonl, write_traffic_table_jsonl


class TrafficModel:
    def __init__(
        self,
        config: ScenarioConfig,
        topology: TopologyModel,
        *,
        capture_table: bool = False,
    ) -> None:
        self.config = config
        self.topology = topology
        self.capture_table = capture_table
        self._request_index = 0
        self._current_time = 0.0
        self._rng = random.Random(config.seed)
        self._table_id = self._build_table_id()
        self._captured_records: list[TrafficRecord] = []

        if config.traffic_mode is TrafficMode.DYNAMIC:
            self._dynamic_source = self._parse_dynamic_source(config.traffic_source)
            self._static_table = None
            self._static_records = ()
            self._static_cursor = 0
            return

        self._dynamic_source = None
        self._static_table, self._static_records = self._parse_static_source(config.traffic_source)
        self._static_cursor = 0

    def next_request(self) -> ServiceRequest:
        if self.config.traffic_mode is TrafficMode.DYNAMIC:
            return self._next_dynamic_request()
        return self._next_static_request()

    def export_table(self) -> tuple[TrafficTable, tuple[TrafficRecord, ...]]:
        if not self.capture_table:
            raise RuntimeError("capture_table must be enabled to export a traffic table")
        table = TrafficTable(
            traffic_table_version="v1",
            table_id=self._table_id,
            scenario_id=self.config.scenario_id,
            topology_id=self.config.topology_id,
            traffic_mode_source="dynamic_export",
            request_count=len(self._captured_records),
            time_unit="simulation_time",
            bit_rate_unit="Gbps",
            seed=self.config.seed,
        )
        return table, tuple(self._captured_records)

    def save_table_jsonl(self, file_path: str | Path) -> Path:
        table, records = self.export_table()
        return write_traffic_table_jsonl(file_path, table, records)

    def _next_dynamic_request(self) -> ServiceRequest:
        assert self._dynamic_source is not None
        self._current_time += self._rng.expovariate(
            1.0 / self._dynamic_source["mean_inter_arrival_time"]
        )
        holding_time = self._rng.expovariate(1.0 / self._dynamic_source["mean_holding_time"])
        source_id, destination_id = self._sample_node_pair()
        bit_rate = self._rng.choices(
            self._dynamic_source["bit_rates"],
            weights=self._dynamic_source["bit_rate_probabilities"],
            k=1,
        )[0]
        request = ServiceRequest(
            request_index=self._request_index,
            service_id=self._request_index,
            source_id=source_id,
            destination_id=destination_id,
            bit_rate=bit_rate,
            arrival_time=self._current_time,
            holding_time=holding_time,
            traffic_mode=TrafficMode.DYNAMIC,
            traffic_origin="generator",
        )
        if self.capture_table:
            self._captured_records.append(
                TrafficRecord(
                    request_index=request.request_index,
                    service_id=request.service_id,
                    source_id=request.source_id,
                    destination_id=request.destination_id,
                    bit_rate=request.bit_rate,
                    arrival_time=request.arrival_time,
                    holding_time=request.holding_time,
                    table_id=self._table_id,
                    row_index=request.request_index,
                    source_label=self.topology.node_names[request.source_id],
                    destination_label=self.topology.node_names[request.destination_id],
                )
            )
        self._request_index += 1
        return request

    def _next_static_request(self) -> ServiceRequest:
        if self._static_cursor >= len(self._static_records):
            raise StopIteration
        record = self._static_records[self._static_cursor]
        self._static_cursor += 1
        self._request_index += 1
        return record.to_service_request()

    def _sample_node_pair(self) -> tuple[int, int]:
        source_id = self._rng.randrange(self.topology.node_count)
        destination_id = self._rng.randrange(self.topology.node_count - 1)
        if destination_id >= source_id:
            destination_id += 1
        return source_id, destination_id

    def _build_table_id(self) -> str:
        seed_fragment = "none" if self.config.seed is None else str(self.config.seed)
        return f"{self.config.scenario_id}__seed_{seed_fragment}"

    def _parse_dynamic_source(
        self, traffic_source: object | None
    ) -> dict[str, tuple[int, ...] | tuple[float, ...] | float]:
        source = {} if traffic_source is None else traffic_source
        if not isinstance(source, Mapping):
            raise ValueError("dynamic traffic_source must be a mapping when provided")
        raw_bit_rates = source.get("bit_rates", (10, 40, 100))
        if not isinstance(raw_bit_rates, Sequence) or len(raw_bit_rates) == 0:
            raise ValueError("bit_rates must be a non-empty sequence")
        bit_rates = tuple(int(bit_rate) for bit_rate in raw_bit_rates)
        raw_probabilities = source.get("bit_rate_probabilities")
        if raw_probabilities is None:
            bit_rate_probabilities = tuple(1.0 / len(bit_rates) for _ in bit_rates)
        else:
            if not isinstance(raw_probabilities, Sequence) or len(raw_probabilities) != len(bit_rates):
                raise ValueError("bit_rate_probabilities must match bit_rates length")
            bit_rate_probabilities = tuple(float(probability) for probability in raw_probabilities)
        mean_holding_time = float(source.get("mean_holding_time", 10800.0))
        if mean_holding_time <= 0:
            raise ValueError("mean_holding_time must be positive")
        if "mean_inter_arrival_time" in source:
            mean_inter_arrival_time = float(source["mean_inter_arrival_time"])
        elif "load" in source:
            load = float(source["load"])
            if load <= 0:
                raise ValueError("load must be positive")
            mean_inter_arrival_time = mean_holding_time / load
        else:
            mean_inter_arrival_time = 1.0
        if mean_inter_arrival_time <= 0:
            raise ValueError("mean_inter_arrival_time must be positive")
        return {
            "bit_rates": bit_rates,
            "bit_rate_probabilities": bit_rate_probabilities,
            "mean_holding_time": mean_holding_time,
            "mean_inter_arrival_time": mean_inter_arrival_time,
        }

    def _parse_static_source(
        self, traffic_source: object | None
    ) -> tuple[TrafficTable, tuple[TrafficRecord, ...]]:
        if isinstance(traffic_source, (str, Path)):
            return read_traffic_table_jsonl(traffic_source)
        if not isinstance(traffic_source, Mapping):
            raise ValueError("static traffic_source must be a mapping with table and records")
        if "path" in traffic_source:
            return read_traffic_table_jsonl(traffic_source["path"])
        table = traffic_source.get("table")
        records = traffic_source.get("records")
        if not isinstance(table, TrafficTable):
            raise ValueError("static traffic_source.table must be a TrafficTable")
        if not isinstance(records, Sequence):
            raise ValueError("static traffic_source.records must be a sequence")
        typed_records = tuple(records)
        if len(typed_records) != table.request_count:
            raise ValueError("TrafficTable request_count must match the number of records")
        for index, record in enumerate(typed_records):
            if not isinstance(record, TrafficRecord):
                raise ValueError("static traffic_source.records must contain TrafficRecord values")
            if record.table_id != table.table_id:
                raise ValueError("TrafficRecord table_id must match TrafficTable table_id")
            if record.row_index != index:
                raise ValueError("TrafficRecord row_index must match canonical row order")
            if record.source_id >= self.topology.node_count or record.destination_id >= self.topology.node_count:
                raise ValueError("TrafficRecord endpoints must exist in the topology")
        return table, typed_records
