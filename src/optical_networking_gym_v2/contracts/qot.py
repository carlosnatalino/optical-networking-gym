from __future__ import annotations

from dataclasses import dataclass

from .modulation import Modulation
from .traffic import ServiceRequest
from optical_networking_gym_v2.network.topology import PathRecord


@dataclass(frozen=True, slots=True)
class QoTRequest:
    request: ServiceRequest
    path: PathRecord
    modulation: Modulation
    service_slot_start: int
    service_num_slots: int
    center_frequency: float
    bandwidth: float
    launch_power: float

    @property
    def service_id(self) -> int:
        return self.request.service_id


@dataclass(frozen=True, slots=True)
class QoTResult:
    osnr: float
    ase: float
    nli: float
    meets_threshold: bool


@dataclass(frozen=True, slots=True)
class ServiceQoTUpdate:
    service_id: int
    osnr: float
    ase: float
    nli: float

    def to_mapping(self) -> dict[str, float]:
        return {"osnr": self.osnr, "ase": self.ase, "nli": self.nli}
