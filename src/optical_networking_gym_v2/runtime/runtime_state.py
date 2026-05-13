from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Mapping

import numpy as np

from optical_networking_gym_v2.contracts.modulation import Modulation
from optical_networking_gym_v2.contracts.traffic import ServiceRequest
from optical_networking_gym_v2.network.topology import PathRecord, TopologyModel
from optical_networking_gym_v2.optical.kernels.allocation_kernel import block_is_free, fill_range
from optical_networking_gym_v2.config.scenario import ScenarioConfig


@dataclass(slots=True)
class ActiveService:
    request: ServiceRequest
    path: PathRecord
    service_slot_start: int
    service_slot_end_exclusive: int
    service_num_slots: int
    occupied_slot_start: int
    occupied_slot_end_exclusive: int
    modulation: Modulation | None = None
    center_frequency: float = 0.0
    bandwidth: float = 0.0
    launch_power: float = 0.0
    osnr: float = 0.0
    ase: float = 0.0
    nli: float = 0.0
    established_osnr: float = 0.0
    shared_link_service_count_at_establishment: int = 0
    is_disrupted: bool = False
    disruption_count: int = 0

    @property
    def service_id(self) -> int:
        return self.request.service_id

    @property
    def link_ids(self) -> tuple[int, ...]:
        return self.path.link_ids

    @property
    def release_time(self) -> float:
        return self.request.release_time


class RuntimeState:
    _next_state_id = 0

    def __init__(self, config: ScenarioConfig, topology: TopologyModel) -> None:
        self.state_id = RuntimeState._next_state_id
        RuntimeState._next_state_id += 1
        self.config = config
        self.topology = topology
        self.current_time = 0.0
        self.current_request: ServiceRequest | None = None
        self.slot_allocation = np.full(
            (topology.link_count, config.num_spectrum_resources),
            fill_value=-1,
            dtype=np.int32,
        )
        self.active_services_by_id: dict[int, ActiveService] = {}
        self.disrupted_services_by_id: dict[int, ActiveService] = {}
        self.release_queue: list[tuple[float, int]] = []
        self.release_times_by_service_id: dict[int, float] = {}
        self.link_active_service_ids: tuple[set[int], ...] = tuple(
            set() for _ in range(topology.link_count)
        )
        self.service_link_indices: dict[int, tuple[int, ...]] = {}
        self.service_slot_ranges: dict[int, tuple[int, int]] = {}
        self.service_occupied_ranges: dict[int, tuple[int, int]] = {}
        self.service_qot_by_id: dict[int, tuple[float, float, float]] = {}
        self.global_state_version = 0
        self.allocation_state_version = 0
        self.link_versions = np.zeros(topology.link_count, dtype=np.int64)
        self._path_link_indices_cache: dict[int, np.ndarray] = {}

    def _get_path_link_indices(self, path: PathRecord) -> np.ndarray:
        cached = self._path_link_indices_cache.get(path.id)
        if cached is None:
            cached = np.asarray(path.link_ids, dtype=np.intp)
            self._path_link_indices_cache[path.id] = cached
        return cached

    def set_current_request(self, request: ServiceRequest) -> None:
        self.current_request = request

    def apply_provision(
        self,
        *,
        request: ServiceRequest,
        path: PathRecord,
        service_slot_start: int,
        service_num_slots: int,
        occupied_slot_start: int | None = None,
        occupied_slot_end_exclusive: int | None = None,
        modulation: Modulation | None = None,
        center_frequency: float = 0.0,
        bandwidth: float = 0.0,
        launch_power: float = 0.0,
    ) -> ActiveService:
        service_id = request.service_id
        link_ids = path.link_ids
        link_indices = self._get_path_link_indices(path)

        if service_id in self.active_services_by_id:
            raise ValueError(f"service_id {service_id} is already active")
        if {path.node_indices[0], path.node_indices[-1]} != {request.source_id, request.destination_id}:
            raise ValueError("path endpoints must match the request endpoints as an undirected pair")
        if service_num_slots <= 0:
            raise ValueError("service_num_slots must be positive")
        logical_end = service_slot_start + service_num_slots
        occupied_start = service_slot_start if occupied_slot_start is None else occupied_slot_start
        occupied_end = logical_end if occupied_slot_end_exclusive is None else occupied_slot_end_exclusive
        if service_slot_start < 0 or occupied_start < 0:
            raise ValueError("slot indices must be non-negative")
        if logical_end > self.config.num_spectrum_resources:
            raise ValueError("service slot range exceeds spectrum capacity")
        if occupied_end > self.config.num_spectrum_resources:
            raise ValueError("occupied slot range exceeds spectrum capacity")
        if occupied_end <= occupied_start:
            raise ValueError("occupied slot range must be non-empty")

        if link_ids and not block_is_free(self.slot_allocation, link_indices, occupied_start, occupied_end):
            raise ValueError("occupied slot range overlaps an occupied allocation")

        active_service = ActiveService(
            request=request,
            path=path,
            service_slot_start=service_slot_start,
            service_slot_end_exclusive=logical_end,
            service_num_slots=service_num_slots,
            occupied_slot_start=occupied_start,
            occupied_slot_end_exclusive=occupied_end,
            modulation=modulation,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
            launch_power=launch_power,
        )

        if link_ids:
            fill_range(self.slot_allocation, link_indices, occupied_start, occupied_end, service_id)
            self.link_versions[link_indices] += 1
            for link_id in link_ids:
                self.link_active_service_ids[link_id].add(service_id)

        self.active_services_by_id[service_id] = active_service
        self.service_link_indices[service_id] = link_ids
        self.service_slot_ranges[service_id] = (service_slot_start, logical_end)
        self.service_occupied_ranges[service_id] = (occupied_start, occupied_end)
        self.service_qot_by_id[service_id] = (0.0, 0.0, 0.0)
        self.release_times_by_service_id[service_id] = request.release_time
        heapq.heappush(self.release_queue, (request.release_time, service_id))
        self.allocation_state_version += 1
        self.global_state_version += 1
        return active_service

    def apply_release(self, service_id: int) -> ActiveService:
        service = self.active_services_by_id.pop(service_id)
        occupied_start, occupied_end = self.service_occupied_ranges.pop(service_id)
        link_ids = self.service_link_indices.pop(service_id)
        link_indices = self._get_path_link_indices(service.path)
        self.service_slot_ranges.pop(service_id)
        self.service_qot_by_id.pop(service_id, None)
        self.release_times_by_service_id.pop(service_id, None)

        if link_ids:
            fill_range(self.slot_allocation, link_indices, occupied_start, occupied_end, -1)
            self.link_versions[link_indices] += 1
            for link_id in link_ids:
                self.link_active_service_ids[link_id].discard(service_id)

        self.allocation_state_version += 1
        self.global_state_version += 1
        return service

    def apply_qot_updates(self, updates: Mapping[int, Mapping[str, float]]) -> None:
        for service_id, values in updates.items():
            if service_id not in self.active_services_by_id:
                raise KeyError(f"service_id {service_id} is not active")
            service = self.active_services_by_id[service_id]
            service.osnr = float(values.get("osnr", service.osnr))
            service.ase = float(values.get("ase", service.ase))
            service.nli = float(values.get("nli", service.nli))
            self.service_qot_by_id[service_id] = (service.osnr, service.ase, service.nli)
        if updates:
            self.global_state_version += 1

    def apply_disruption(self, service_id: int, *, terminal: bool = False) -> ActiveService:
        if terminal:
            service = self.apply_release(service_id)
        else:
            service = self.active_services_by_id[service_id]
        service.is_disrupted = True
        service.disruption_count += 1
        if terminal:
            self.disrupted_services_by_id[service_id] = service
        self.global_state_version += 1
        return service

    def advance_time_and_release_due_services(self, until_time: float) -> tuple[ActiveService, ...]:
        if until_time < self.current_time:
            raise ValueError("until_time cannot move backwards")
        self.current_time = until_time
        released: list[ActiveService] = []
        while self.release_queue and self.release_queue[0][0] <= until_time:
            release_time, service_id = heapq.heappop(self.release_queue)
            scheduled_release_time = self.release_times_by_service_id.get(service_id)
            if scheduled_release_time is None:
                continue
            if scheduled_release_time != release_time:
                continue
            released.append(self.apply_release(service_id))
        return tuple(released)

    def release_queue_snapshot(self) -> tuple[tuple[float, int], ...]:
        return tuple(
            sorted(
                (release_time, service_id)
                for service_id, release_time in self.release_times_by_service_id.items()
            )
        )

    def validate_invariants(self) -> None:
        for service_id, service in self.active_services_by_id.items():
            if service_id not in self.service_link_indices:
                raise AssertionError(f"service {service_id} is missing link indices")
            if service_id not in self.service_slot_ranges:
                raise AssertionError(f"service {service_id} is missing logical slot range")
            if service_id not in self.service_occupied_ranges:
                raise AssertionError(f"service {service_id} is missing occupied slot range")
            if service_id not in self.service_qot_by_id:
                raise AssertionError(f"service {service_id} is missing qot snapshot")

            occupied_start, occupied_end = self.service_occupied_ranges[service_id]
            for link_id in service.link_ids:
                if service_id not in self.link_active_service_ids[link_id]:
                    raise AssertionError(f"service {service_id} missing from link_active_service_ids")
                if not np.all(self.slot_allocation[link_id, occupied_start:occupied_end] == service_id):
                    raise AssertionError(f"service {service_id} slot allocation is inconsistent")

        active_service_ids = set(self.active_services_by_id)
        disrupted_service_ids = set(self.disrupted_services_by_id)
        if active_service_ids & disrupted_service_ids:
            raise AssertionError("active and disrupted service registries must be disjoint")
        for link_id, service_ids in enumerate(self.link_active_service_ids):
            for service_id in service_ids:
                if service_id not in active_service_ids:
                    raise AssertionError(
                        f"link_active_service_ids contains inactive service {service_id} on link {link_id}"
                    )

        if set(self.release_times_by_service_id) != active_service_ids:
            raise AssertionError("release schedule does not match active services")

        for service_id, release_time in self.release_times_by_service_id.items():
            if release_time != self.active_services_by_id[service_id].release_time:
                raise AssertionError(
                    f"release schedule for service {service_id} is inconsistent with the request"
                )
        for service_id in disrupted_service_ids:
            if service_id in self.release_times_by_service_id:
                raise AssertionError("disrupted services cannot remain in the release schedule")
            if service_id in self.service_qot_by_id:
                raise AssertionError("disrupted services cannot remain in active qos snapshots")
