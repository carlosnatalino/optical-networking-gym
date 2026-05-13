from __future__ import annotations

from optical_networking_gym_v2.contracts import StatisticsSnapshot, Status, StepTransition
from optical_networking_gym_v2.config.scenario import ScenarioConfig


class Statistics:
    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self._configured_spectral_efficiencies = tuple(
            sorted({int(modulation.spectral_efficiency) for modulation in config.modulations})
        )
        self._services_processed = 0
        self._services_accepted = 0
        self._services_blocked_resources = 0
        self._services_blocked_qot = 0
        self._services_rejected_by_agent = 0
        self._bit_rate_requested = 0.0
        self._bit_rate_provisioned = 0.0
        self._disrupted_services = 0
        self._services_dropped_qot = 0
        self._episode_modulation_histogram: dict[int, int] = {}
        self.reset_episode()

    def reset_episode(self) -> None:
        self._episode_services_processed = 0
        self._episode_services_accepted = 0
        self._episode_services_blocked_resources = 0
        self._episode_services_blocked_qot = 0
        self._episode_services_rejected_by_agent = 0
        self._episode_bit_rate_requested = 0.0
        self._episode_bit_rate_provisioned = 0.0
        self._episode_disrupted_services = 0
        self._episode_services_dropped_qot = 0
        self._episode_modulation_histogram = {
            spectral_efficiency: 0 for spectral_efficiency in self._configured_spectral_efficiencies
        }
        self._episode_modulation_histogram_cache = tuple(
            (spectral_efficiency, 0) for spectral_efficiency in self._configured_spectral_efficiencies
        )

    def record_transition(self, transition: StepTransition) -> None:
        self._services_processed += 1
        self._episode_services_processed += 1
        self._bit_rate_requested += float(transition.request.bit_rate)
        self._episode_bit_rate_requested += float(transition.request.bit_rate)

        status = transition.allocation.status
        if status is Status.ACCEPTED:
            self._services_accepted += 1
            self._episode_services_accepted += 1
            self._bit_rate_provisioned += float(transition.request.bit_rate)
            self._episode_bit_rate_provisioned += float(transition.request.bit_rate)
            if transition.modulation_spectral_efficiency is not None:
                self._episode_modulation_histogram.setdefault(transition.modulation_spectral_efficiency, 0)
                self._episode_modulation_histogram[transition.modulation_spectral_efficiency] += 1
                self._episode_modulation_histogram_cache = None
        elif status is Status.BLOCKED_RESOURCES:
            self._services_blocked_resources += 1
            self._episode_services_blocked_resources += 1
        elif status is Status.BLOCKED_QOT:
            self._services_blocked_qot += 1
            self._episode_services_blocked_qot += 1
        elif status is Status.REJECTED_BY_AGENT:
            self._services_rejected_by_agent += 1
            self._episode_services_rejected_by_agent += 1
        else:
            raise ValueError(f"unsupported status {status!r}")

        self._disrupted_services += transition.disrupted_services
        self._episode_disrupted_services += transition.disrupted_services

    def record_dropped_qot(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._services_dropped_qot += count
        self._episode_services_dropped_qot += count

    def record_post_admission_effects(self, *, disrupted_services: int = 0, dropped_qot: int = 0) -> None:
        if disrupted_services < 0:
            raise ValueError("disrupted_services must be non-negative")
        if dropped_qot < 0:
            raise ValueError("dropped_qot must be non-negative")
        self._disrupted_services += disrupted_services
        self._episode_disrupted_services += disrupted_services
        self.record_dropped_qot(dropped_qot)

    @property
    def services_processed(self) -> int:
        return self._services_processed

    @property
    def services_accepted(self) -> int:
        return self._services_accepted

    @property
    def services_blocked_resources(self) -> int:
        return self._services_blocked_resources

    @property
    def services_blocked_qot(self) -> int:
        return self._services_blocked_qot

    @property
    def services_rejected_by_agent(self) -> int:
        return self._services_rejected_by_agent

    @property
    def bit_rate_requested(self) -> float:
        return self._bit_rate_requested

    @property
    def bit_rate_provisioned(self) -> float:
        return self._bit_rate_provisioned

    @property
    def disrupted_services(self) -> int:
        return self._disrupted_services

    @property
    def services_dropped_qot(self) -> int:
        return self._services_dropped_qot

    @property
    def episode_services_processed(self) -> int:
        return self._episode_services_processed

    @property
    def episode_services_accepted(self) -> int:
        return self._episode_services_accepted

    @property
    def episode_services_blocked_resources(self) -> int:
        return self._episode_services_blocked_resources

    @property
    def episode_services_blocked_qot(self) -> int:
        return self._episode_services_blocked_qot

    @property
    def episode_services_rejected_by_agent(self) -> int:
        return self._episode_services_rejected_by_agent

    @property
    def episode_bit_rate_requested(self) -> float:
        return self._episode_bit_rate_requested

    @property
    def episode_bit_rate_provisioned(self) -> float:
        return self._episode_bit_rate_provisioned

    @property
    def episode_disrupted_services(self) -> int:
        return self._episode_disrupted_services

    @property
    def episode_services_dropped_qot(self) -> int:
        return self._episode_services_dropped_qot

    @property
    def episode_modulation_histogram(self) -> tuple[tuple[int, int], ...]:
        cached = self._episode_modulation_histogram_cache
        if cached is not None:
            return cached

        extra_keys = tuple(
            sorted(
                spectral_efficiency
                for spectral_efficiency in self._episode_modulation_histogram
                if spectral_efficiency not in self._configured_spectral_efficiencies
            )
        )
        histogram_keys = self._configured_spectral_efficiencies + extra_keys
        histogram = tuple(
            (spectral_efficiency, self._episode_modulation_histogram.get(spectral_efficiency, 0))
            for spectral_efficiency in histogram_keys
        )
        self._episode_modulation_histogram_cache = histogram
        return histogram

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

    def snapshot(self) -> StatisticsSnapshot:
        return StatisticsSnapshot(
            services_processed=self._services_processed,
            services_accepted=self._services_accepted,
            services_blocked_resources=self._services_blocked_resources,
            services_blocked_qot=self._services_blocked_qot,
            services_rejected_by_agent=self._services_rejected_by_agent,
            bit_rate_requested=self._bit_rate_requested,
            bit_rate_provisioned=self._bit_rate_provisioned,
            disrupted_services=self._disrupted_services,
            services_dropped_qot=self._services_dropped_qot,
            episode_services_processed=self._episode_services_processed,
            episode_services_accepted=self._episode_services_accepted,
            episode_services_blocked_resources=self._episode_services_blocked_resources,
            episode_services_blocked_qot=self._episode_services_blocked_qot,
            episode_services_rejected_by_agent=self._episode_services_rejected_by_agent,
            episode_bit_rate_requested=self._episode_bit_rate_requested,
            episode_bit_rate_provisioned=self._episode_bit_rate_provisioned,
            episode_disrupted_services=self._episode_disrupted_services,
            episode_services_dropped_qot=self._episode_services_dropped_qot,
            episode_modulation_histogram=self.episode_modulation_histogram,
        )

    def validate_invariants(self) -> None:
        if self._services_processed != (
            self._services_accepted
            + self._services_blocked_resources
            + self._services_blocked_qot
            + self._services_rejected_by_agent
        ):
            raise AssertionError("processed counters are inconsistent")
        if self._episode_services_processed != (
            self._episode_services_accepted
            + self._episode_services_blocked_resources
            + self._episode_services_blocked_qot
            + self._episode_services_rejected_by_agent
        ):
            raise AssertionError("episode processed counters are inconsistent")
        if self._bit_rate_provisioned > self._bit_rate_requested:
            raise AssertionError("bit_rate_provisioned cannot exceed bit_rate_requested")
        if self._episode_bit_rate_provisioned > self._episode_bit_rate_requested:
            raise AssertionError("episode_bit_rate_provisioned cannot exceed episode_bit_rate_requested")
        if self._disrupted_services < 0 or self._episode_disrupted_services < 0:
            raise AssertionError("disrupted services cannot be negative")
        if self._services_dropped_qot < 0 or self._episode_services_dropped_qot < 0:
            raise AssertionError("dropped qot services cannot be negative")
        if self.services_served < 0 or self.episode_services_served < 0:
            raise AssertionError("served services cannot be negative")
        for spectral_efficiency, count in self._episode_modulation_histogram.items():
            if spectral_efficiency <= 0:
                raise AssertionError("modulation histogram keys must be positive")
            if count < 0:
                raise AssertionError("modulation histogram counts must be non-negative")


__all__ = ["Statistics"]
