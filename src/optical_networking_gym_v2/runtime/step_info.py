from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from optical_networking_gym_v2.contracts import RewardBreakdown, StatisticsSnapshot, StepTransition
from optical_networking_gym_v2.config.scenario import ScenarioConfig

if TYPE_CHECKING:
    from optical_networking_gym_v2.stats.statistics import Statistics


class StepInfo:
    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config

    def build(
        self,
        statistics: StatisticsSnapshot | "Statistics",
        transition: StepTransition,
        *,
        terminated: bool = False,
        truncated: bool = False,
        reward: float | None = None,
        reward_breakdown: RewardBreakdown | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        info: dict[str, object] = {
            "accepted": transition.accepted,
            "status": transition.allocation.status.value,
            "services_processed": statistics.services_processed,
            "services_accepted": statistics.services_accepted,
            "episode_services_processed": statistics.episode_services_processed,
            "episode_services_accepted": statistics.episode_services_accepted,
            "services_served": statistics.services_served,
            "episode_services_served": statistics.episode_services_served,
            "service_blocking_rate": statistics.service_blocking_rate,
            "episode_service_blocking_rate": statistics.episode_service_blocking_rate,
            "service_served_rate": statistics.service_served_rate,
            "episode_service_served_rate": statistics.episode_service_served_rate,
            "bit_rate_blocking_rate": statistics.bit_rate_blocking_rate,
            "episode_bit_rate_blocking_rate": statistics.episode_bit_rate_blocking_rate,
            "disrupted_services": statistics.disrupted_services_rate,
            "episode_disrupted_services": statistics.episode_disrupted_services_rate,
            "osnr": float(transition.osnr),
            "osnr_req": float(transition.osnr_requirement),
            "chosen_path_index": transition.chosen_path_index,
            "chosen_slot": transition.chosen_slot,
            "chosen_modulation_index": transition.chosen_modulation_index,
            "fragmentation_shannon_entropy": float(transition.fragmentation_shannon_entropy),
            "fragmentation_route_cuts": float(transition.fragmentation_route_cuts),
            "fragmentation_route_rss": float(transition.fragmentation_route_rss),
            "terminated": terminated,
            "truncated": truncated,
        }
        if transition.action is not None:
            info["action"] = transition.action
        if transition.mask is not None:
            info["mask"] = transition.mask
        if reward is not None:
            info["reward"] = reward
        if reward_breakdown is not None:
            info.update(reward_breakdown.to_mapping())

        for spectral_efficiency, count in statistics.episode_modulation_histogram:
            info[f"modulation_{spectral_efficiency}"] = count

        if terminated:
            info["blocked_due_to_resources"] = statistics.episode_services_blocked_resources
            info["blocked_due_to_osnr"] = statistics.episode_services_blocked_qot
            info["disrupted_or_dropped_services"] = statistics.episode_services_dropped_qot
            info["rejected"] = statistics.episode_services_rejected_by_agent

        if extra:
            info.update(extra)
        return info


__all__ = ["StepInfo"]
