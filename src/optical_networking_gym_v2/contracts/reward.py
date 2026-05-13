from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysis
    from .step import StatisticsSnapshot, StepTransition


@dataclass(frozen=True, slots=True)
class CandidateRewardMetrics:
    osnr_margin: float = 0.0
    nli_share: float = 0.0
    worst_link_nli_share: float = 0.0
    fragmentation_damage_num_blocks: float = 0.0
    fragmentation_damage_largest_block: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardInput:
    transition: "StepTransition"
    statistics: "StatisticsSnapshot"
    request_analysis: "RequestAnalysis | None" = None
    selected_candidate_metrics: CandidateRewardMetrics | None = None
    has_valid_non_reject_action: bool | None = None


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    profile: str
    raw_reward: float
    clipped_reward: float
    accept_component: float
    spectral_efficiency_bonus: float
    fragmentation_penalty: float
    physical_penalty: float
    reject_penalty: float

    def to_mapping(self) -> dict[str, float | str]:
        return {
            "reward_profile": self.profile,
            "reward_raw": self.raw_reward,
            "reward_clipped": self.clipped_reward,
            "reward_accept_component": self.accept_component,
            "reward_spectral_efficiency_bonus": self.spectral_efficiency_bonus,
            "reward_fragmentation_penalty": self.fragmentation_penalty,
            "reward_physical_penalty": self.physical_penalty,
            "reward_reject_penalty": self.reject_penalty,
        }


__all__ = ["CandidateRewardMetrics", "RewardBreakdown", "RewardInput"]
