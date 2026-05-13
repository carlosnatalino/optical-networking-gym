from __future__ import annotations

from optical_networking_gym_v2.contracts import (
    CandidateRewardMetrics,
    RewardBreakdown,
    RewardInput,
    RewardProfile,
    Status,
)
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.config.scenario import ScenarioConfig


class RewardFunction:
    def __init__(
        self,
        config: ScenarioConfig,
        topology: TopologyModel,
        *,
        profile: RewardProfile | str | None = None,
    ) -> None:
        self.config = config
        self.topology = topology
        self.profile = RewardProfile(profile or config.reward_profile)
        self._max_spectral_efficiency = max(
            (modulation.spectral_efficiency for modulation in config.modulations),
            default=1,
        )

    def evaluate(self, reward_input: RewardInput) -> tuple[float, RewardBreakdown]:
        return self.evaluate_transition(
            reward_input.transition,
            request_analysis=reward_input.request_analysis,
            selected_candidate_metrics=reward_input.selected_candidate_metrics,
            has_valid_non_reject_action=reward_input.has_valid_non_reject_action,
        )

    def evaluate_transition(
        self,
        transition,
        *,
        request_analysis=None,
        selected_candidate_metrics: CandidateRewardMetrics | None = None,
        has_valid_non_reject_action: bool | None = None,
    ) -> tuple[float, RewardBreakdown]:
        if self.profile is RewardProfile.BALANCED:
            breakdown = self._evaluate_balanced(
                transition,
                request_analysis=request_analysis,
                selected_candidate_metrics=selected_candidate_metrics,
                has_valid_non_reject_action=has_valid_non_reject_action,
            )
        elif self.profile is RewardProfile.LEGACY:
            breakdown = self._evaluate_legacy(
                transition,
                request_analysis=request_analysis,
                selected_candidate_metrics=selected_candidate_metrics,
                has_valid_non_reject_action=has_valid_non_reject_action,
            )
        else:
            raise ValueError(f"unsupported reward profile {self.profile!r}")
        return breakdown.clipped_reward, breakdown

    def _evaluate_balanced(
        self,
        transition,
        *,
        request_analysis=None,
        selected_candidate_metrics: CandidateRewardMetrics | None = None,
        has_valid_non_reject_action: bool | None = None,
    ) -> RewardBreakdown:
        status = transition.allocation.status

        if status is not Status.ACCEPTED:
            reject_penalty = self._balanced_reject_penalty(
                status=status,
                has_valid_non_reject_action=self._resolve_has_valid_non_reject_action(
                    request_analysis=request_analysis,
                    has_valid_non_reject_action=has_valid_non_reject_action,
                ),
            )
            raw_reward = -reject_penalty
            clipped_reward = _clip_reward(raw_reward)
            return RewardBreakdown(
                profile=RewardProfile.BALANCED.value,
                raw_reward=raw_reward,
                clipped_reward=clipped_reward,
                accept_component=0.0,
                spectral_efficiency_bonus=0.0,
                fragmentation_penalty=0.0,
                physical_penalty=0.0,
                reject_penalty=reject_penalty,
            )

        metrics = self._resolve_selected_candidate_metrics(
            transition,
            request_analysis=request_analysis,
            selected_candidate_metrics=selected_candidate_metrics,
        )
        accept_component = 1.0
        spectral_efficiency_bonus = 0.20 * _spectral_efficiency_norm(
            transition.modulation_spectral_efficiency,
            self._max_spectral_efficiency,
        )
        fragmentation_damage = (
            (0.6 * metrics.fragmentation_damage_num_blocks)
            + (0.4 * metrics.fragmentation_damage_largest_block)
        )
        fragmentation_penalty = 0.35 * fragmentation_damage
        margin_risk = 1.0 - min(max(metrics.osnr_margin / 3.0, 0.0), 1.0)
        physical_risk = (0.6 * margin_risk) + (0.4 * metrics.worst_link_nli_share)
        physical_penalty = 0.25 * physical_risk

        raw_reward = (
            accept_component
            + spectral_efficiency_bonus
            - fragmentation_penalty
            - physical_penalty
        )
        clipped_reward = _clip_reward(raw_reward)
        return RewardBreakdown(
            profile=RewardProfile.BALANCED.value,
            raw_reward=raw_reward,
            clipped_reward=clipped_reward,
            accept_component=accept_component,
            spectral_efficiency_bonus=spectral_efficiency_bonus,
            fragmentation_penalty=fragmentation_penalty,
            physical_penalty=physical_penalty,
            reject_penalty=0.0,
        )

    def _evaluate_legacy(
        self,
        transition,
        *,
        request_analysis=None,
        selected_candidate_metrics: CandidateRewardMetrics | None = None,
        has_valid_non_reject_action: bool | None = None,
    ) -> RewardBreakdown:
        status = transition.allocation.status

        if status is not Status.ACCEPTED:
            if status in (Status.BLOCKED_RESOURCES, Status.BLOCKED_QOT):
                raw_reward = -1.8
                reject_penalty = 1.8
            elif status is Status.REJECTED_BY_AGENT:
                raw_reward = -2.0
                reject_penalty = 2.0
            else:
                raise ValueError(f"unsupported status {status!r}")
            clipped_reward = _clip_reward(raw_reward)
            return RewardBreakdown(
                profile=RewardProfile.LEGACY.value,
                raw_reward=raw_reward,
                clipped_reward=clipped_reward,
                accept_component=0.0,
                spectral_efficiency_bonus=0.0,
                fragmentation_penalty=0.0,
                physical_penalty=0.0,
                reject_penalty=reject_penalty,
            )

        accept_component = 1.0
        spectral_efficiency_bonus = 0.5 * _spectral_efficiency_norm(
            transition.modulation_spectral_efficiency,
            self._max_spectral_efficiency,
        )
        route_cuts_norm = min(
            float(transition.fragmentation_route_cuts) / max(1.0, float(self.topology.link_count) * 2.0),
            1.0,
        )
        frag_score = (
            0.4 * float(transition.fragmentation_shannon_entropy)
            + 0.3 * route_cuts_norm
            + 0.3 * float(transition.fragmentation_route_rss)
        )
        fragmentation_penalty = 0.3 * frag_score

        physical_penalty = 0.0
        if (
            transition.modulation_spectral_efficiency is not None
            and transition.modulation_spectral_efficiency < self._max_spectral_efficiency
        ):
            osnr_margin = float(transition.osnr) - float(transition.osnr_requirement)
            osnr_waste_normalized = min(max(osnr_margin / 3.0, 0.0), 3.0)
            physical_penalty = 0.20 * osnr_waste_normalized

        raw_reward = accept_component + spectral_efficiency_bonus - fragmentation_penalty - physical_penalty
        clipped_reward = _clip_reward(raw_reward)
        return RewardBreakdown(
            profile=RewardProfile.LEGACY.value,
            raw_reward=raw_reward,
            clipped_reward=clipped_reward,
            accept_component=accept_component,
            spectral_efficiency_bonus=spectral_efficiency_bonus,
            fragmentation_penalty=fragmentation_penalty,
            physical_penalty=physical_penalty,
            reject_penalty=0.0,
        )

    def _resolve_selected_candidate_metrics(
        self,
        transition,
        *,
        request_analysis=None,
        selected_candidate_metrics: CandidateRewardMetrics | None = None,
    ) -> CandidateRewardMetrics:
        if selected_candidate_metrics is not None:
            return selected_candidate_metrics
        if request_analysis is None:
            return CandidateRewardMetrics()
        if not transition.accepted:
            return CandidateRewardMetrics()
        resolved = request_analysis.selected_candidate_metrics(
            path_index=transition.chosen_path_index if transition.chosen_path_index is not None else -1,
            modulation_index=transition.chosen_modulation_index if transition.chosen_modulation_index is not None else -1,
            initial_slot=transition.chosen_slot if transition.chosen_slot is not None else -1,
        )
        if resolved is None:
            return CandidateRewardMetrics()
        return resolved

    def _resolve_has_valid_non_reject_action(
        self,
        *,
        request_analysis=None,
        has_valid_non_reject_action: bool | None = None,
    ) -> bool:
        if has_valid_non_reject_action is not None:
            return has_valid_non_reject_action
        if request_analysis is None:
            return False
        return request_analysis.has_valid_non_reject_action

    def _balanced_reject_penalty(self, *, status: Status, has_valid_non_reject_action: bool) -> float:
        if status is Status.REJECTED_BY_AGENT:
            return 1.0 if has_valid_non_reject_action else 0.15
        if status is Status.BLOCKED_RESOURCES:
            return 1.10
        if status is Status.BLOCKED_QOT:
            return 1.20
        raise ValueError(f"unsupported status {status!r}")


def _spectral_efficiency_norm(spectral_efficiency: int | None, max_spectral_efficiency: int) -> float:
    if spectral_efficiency is None or max_spectral_efficiency <= 0:
        return 0.0
    return float(spectral_efficiency) / float(max_spectral_efficiency)


def _clip_reward(raw_reward: float) -> float:
    return max(-2.0, min(2.0, raw_reward))


__all__ = ["RewardFunction"]
