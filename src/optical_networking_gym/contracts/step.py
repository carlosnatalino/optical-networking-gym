from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .allocation import Allocation
from .traffic import ServiceRequest


@dataclass(slots=True)
class StepTransition:
    request: ServiceRequest
    allocation: Allocation
    action: int | None = None
    modulation_spectral_efficiency: int | None = None
    osnr: float = 0.0
    osnr_requirement: float = 0.0
    disrupted_services: int = 0
    dropped_qot: int = 0
    fragmentation_shannon_entropy: float = 0.0
    fragmentation_route_cuts: float = 0.0
    fragmentation_route_rss: float = 0.0
    mask: np.ndarray | None = None

    @classmethod
    def accept(
        cls,
        *,
        request: ServiceRequest,
        allocation: Allocation,
        modulation_spectral_efficiency: int | None = None,
        action: int | None = None,
        osnr: float = 0.0,
        osnr_requirement: float = 0.0,
        disrupted_services: int = 0,
        dropped_qot: int = 0,
        fragmentation_shannon_entropy: float = 0.0,
        fragmentation_route_cuts: float = 0.0,
        fragmentation_route_rss: float = 0.0,
        mask: np.ndarray | None = None,
    ) -> "StepTransition":
        if not allocation.accepted:
            raise ValueError("accepted transitions require an accepted allocation")
        return cls(
            request=request,
            allocation=allocation,
            action=action,
            modulation_spectral_efficiency=modulation_spectral_efficiency,
            osnr=osnr,
            osnr_requirement=osnr_requirement,
            disrupted_services=disrupted_services,
            dropped_qot=dropped_qot,
            fragmentation_shannon_entropy=fragmentation_shannon_entropy,
            fragmentation_route_cuts=fragmentation_route_cuts,
            fragmentation_route_rss=fragmentation_route_rss,
            mask=mask,
        )

    @property
    def accepted(self) -> bool:
        return self.allocation.accepted

    @property
    def chosen_path_index(self) -> int | None:
        return self.allocation.path_index

    @property
    def chosen_slot(self) -> int | None:
        return self.allocation.service_slot_start

    @property
    def chosen_modulation_index(self) -> int | None:
        return self.allocation.modulation_index


@dataclass(frozen=True, slots=True)
class StatisticsSnapshot:
    services_processed: int
    services_accepted: int
    services_blocked_resources: int
    services_blocked_qot: int
    services_rejected_by_agent: int
    bit_rate_requested: float
    bit_rate_provisioned: float
    disrupted_services: int
    services_dropped_qot: int
    episode_services_processed: int
    episode_services_accepted: int
    episode_services_blocked_resources: int
    episode_services_blocked_qot: int
    episode_services_rejected_by_agent: int
    episode_bit_rate_requested: float
    episode_bit_rate_provisioned: float
    episode_disrupted_services: int
    episode_services_dropped_qot: int
    episode_modulation_histogram: tuple[tuple[int, int], ...]

    @property
    def services_blocked(self) -> int:
        return self.services_blocked_resources + self.services_blocked_qot + self.services_rejected_by_agent

    @property
    def episode_services_blocked(self) -> int:
        return (
            self.episode_services_blocked_resources
            + self.episode_services_blocked_qot
            + self.episode_services_rejected_by_agent
        )

    @property
    def services_served(self) -> int:
        return self.services_accepted - self.services_dropped_qot

    @property
    def episode_services_served(self) -> int:
        return self.episode_services_accepted - self.episode_services_dropped_qot

    @property
    def service_blocking_rate(self) -> float:
        if self.services_processed == 0:
            return 0.0
        return float(self.services_blocked) / float(self.services_processed)

    @property
    def episode_service_blocking_rate(self) -> float:
        if self.episode_services_processed == 0:
            return 0.0
        return float(self.episode_services_blocked) / float(self.episode_services_processed)

    @property
    def service_served_rate(self) -> float:
        if self.services_processed == 0:
            return 0.0
        return float(self.services_served) / float(self.services_processed)

    @property
    def episode_service_served_rate(self) -> float:
        if self.episode_services_processed == 0:
            return 0.0
        return float(self.episode_services_served) / float(self.episode_services_processed)

    @property
    def bit_rate_blocking_rate(self) -> float:
        if self.bit_rate_requested <= 0:
            return 0.0
        return float(self.bit_rate_requested - self.bit_rate_provisioned) / float(self.bit_rate_requested)

    @property
    def episode_bit_rate_blocking_rate(self) -> float:
        if self.episode_bit_rate_requested <= 0:
            return 0.0
        return float(self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / float(
            self.episode_bit_rate_requested
        )

    @property
    def disrupted_services_rate(self) -> float:
        if self.services_accepted <= 0:
            return 0.0
        return float(self.disrupted_services) / float(self.services_accepted)

    @property
    def episode_disrupted_services_rate(self) -> float:
        if self.episode_services_accepted <= 0:
            return 0.0
        return float(self.episode_disrupted_services) / float(self.episode_services_accepted)


__all__ = ["StatisticsSnapshot", "StepTransition"]
