from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym_v2 import (
    ActionMask,
    Allocation,
    CandidateRewardMetrics,
    Modulation,
    QoTEngine,
    RequestAnalysisEngine,
    RewardFunction,
    RewardInput,
    RewardProfile,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    Statistics,
    Status,
    StepTransition,
    TopologyModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="reward",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        reward_profile=RewardProfile.BALANCED,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )


def _request(service_id: int, *, bit_rate: int = 40) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=2,
        bit_rate=bit_rate,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def _snapshot() -> object:
    return Statistics(_config()).snapshot()


def test_balanced_reward_accepted_uses_candidate_metrics() -> None:
    config = _config()
    reward = RewardFunction(config, _topology(), profile=RewardProfile.BALANCED)
    transition = StepTransition.accept(
        request=_request(1),
        allocation=Allocation.accept(
            path_index=0,
            modulation_index=1,
            service_slot_start=4,
            service_num_slots=2,
            occupied_slot_start=4,
            occupied_slot_end_exclusive=7,
        ),
        modulation_spectral_efficiency=4,
    )
    metrics = CandidateRewardMetrics(
        osnr_margin=1.5,
        nli_share=0.25,
        worst_link_nli_share=0.5,
        fragmentation_damage_num_blocks=0.4,
        fragmentation_damage_largest_block=0.3,
    )

    value, breakdown = reward.evaluate(
        RewardInput(
            transition=transition,
            statistics=_snapshot(),
            selected_candidate_metrics=metrics,
            has_valid_non_reject_action=True,
        )
    )

    expected_fragmentation_damage = (0.6 * 0.4) + (0.4 * 0.3)
    expected_fragmentation_penalty = 0.35 * expected_fragmentation_damage
    expected_margin_risk = 1.0 - 0.5
    expected_physical_risk = (0.6 * expected_margin_risk) + (0.4 * 0.5)
    expected_physical_penalty = 0.25 * expected_physical_risk
    expected_raw = 1.0 + 0.2 - expected_fragmentation_penalty - expected_physical_penalty

    assert value == pytest.approx(expected_raw)
    assert breakdown.raw_reward == pytest.approx(expected_raw)
    assert breakdown.clipped_reward == pytest.approx(expected_raw)
    assert breakdown.spectral_efficiency_bonus == pytest.approx(0.2)
    assert breakdown.fragmentation_penalty == pytest.approx(expected_fragmentation_penalty)
    assert breakdown.physical_penalty == pytest.approx(expected_physical_penalty)
    assert breakdown.fragmentation_penalty > breakdown.physical_penalty


def test_balanced_reward_reject_without_valid_action_is_lightly_negative() -> None:
    reward = RewardFunction(_config(), _topology(), profile=RewardProfile.BALANCED)
    transition = StepTransition(
        request=_request(2),
        allocation=Allocation.reject(Status.REJECTED_BY_AGENT),
    )

    value, breakdown = reward.evaluate(
        RewardInput(
            transition=transition,
            statistics=_snapshot(),
            has_valid_non_reject_action=False,
        )
    )

    assert value == pytest.approx(-0.15)
    assert breakdown.raw_reward == pytest.approx(-0.15)
    assert breakdown.reject_penalty == pytest.approx(0.15)


def test_balanced_reward_reject_with_valid_action_is_penalized_more() -> None:
    reward = RewardFunction(_config(), _topology(), profile=RewardProfile.BALANCED)
    transition = StepTransition(
        request=_request(3),
        allocation=Allocation.reject(Status.REJECTED_BY_AGENT),
    )

    value, breakdown = reward.evaluate(
        RewardInput(
            transition=transition,
            statistics=_snapshot(),
            has_valid_non_reject_action=True,
        )
    )

    assert value == pytest.approx(-1.0)
    assert breakdown.reject_penalty == pytest.approx(1.0)


def test_legacy_reward_matches_documented_formula() -> None:
    config = _config()
    reward = RewardFunction(config, _topology(), profile=RewardProfile.LEGACY)
    transition = StepTransition.accept(
        request=_request(4),
        allocation=Allocation.accept(
            path_index=0,
            modulation_index=0,
            service_slot_start=3,
            service_num_slots=2,
            occupied_slot_start=3,
            occupied_slot_end_exclusive=6,
        ),
        modulation_spectral_efficiency=2,
        osnr=10.72,
        osnr_requirement=6.72,
        fragmentation_shannon_entropy=0.5,
        fragmentation_route_cuts=4.0,
        fragmentation_route_rss=0.25,
    )

    value, breakdown = reward.evaluate(
        RewardInput(
            transition=transition,
            statistics=_snapshot(),
            has_valid_non_reject_action=True,
        )
    )

    route_cuts_norm = min(4.0 / (_topology().link_count * 2.0), 1.0)
    frag_score = (0.4 * 0.5) + (0.3 * route_cuts_norm) + (0.3 * 0.25)
    osnr_waste = min(max((10.72 - 6.72) / 3.0, 0.0), 3.0)
    expected_raw = 1.0 + (0.5 * 0.5) - (0.3 * frag_score) - (0.2 * osnr_waste)

    assert value == pytest.approx(expected_raw)
    assert breakdown.raw_reward == pytest.approx(expected_raw)
    assert breakdown.profile == RewardProfile.LEGACY.value


def test_reward_function_can_resolve_metrics_from_request_analysis() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    qot_engine = QoTEngine(config, topology)
    analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
    action_mask = ActionMask(config, topology, qot_engine, analysis_engine=analysis_engine)
    request = _request(5)
    analysis = analysis_engine.build(state, request)
    valid_action = int(np.flatnonzero(analysis.action_mask)[0])
    selection = action_mask.decode_action(valid_action, state, request)

    transition = StepTransition.accept(
        request=request,
        allocation=Allocation.accept(
            path_index=selection.path_index,
            modulation_index=selection.modulation_index,
            service_slot_start=selection.initial_slot,
            service_num_slots=int(
                analysis.required_slots_by_path_mod[
                    selection.path_index,
                    int(np.flatnonzero(np.asarray(analysis.modulation_indices) == selection.modulation_index)[0]),
                ]
            ),
            occupied_slot_start=selection.initial_slot,
            occupied_slot_end_exclusive=selection.initial_slot + 2,
        ),
        modulation_spectral_efficiency=config.modulations[selection.modulation_index].spectral_efficiency,
    )

    value, breakdown = RewardFunction(config, topology).evaluate(
        RewardInput(
            transition=transition,
            statistics=_snapshot(),
            request_analysis=analysis,
        )
    )

    resolved = analysis.selected_candidate_metrics(
        path_index=selection.path_index,
        modulation_index=selection.modulation_index,
        initial_slot=selection.initial_slot,
    )
    assert resolved is not None
    assert value == pytest.approx(breakdown.clipped_reward)
    assert breakdown.fragmentation_penalty == pytest.approx(
        0.35
        * ((0.6 * resolved.fragmentation_damage_num_blocks) + (0.4 * resolved.fragmentation_damage_largest_block))
    )
    assert breakdown.raw_reward <= 2.0


def test_direct_reward_evaluation_matches_reward_input_api() -> None:
    config = _config()
    topology = _topology()
    reward = RewardFunction(config, topology, profile=RewardProfile.BALANCED)
    transition = StepTransition.accept(
        request=_request(6),
        allocation=Allocation.accept(
            path_index=0,
            modulation_index=1,
            service_slot_start=4,
            service_num_slots=2,
            occupied_slot_start=4,
            occupied_slot_end_exclusive=7,
        ),
        modulation_spectral_efficiency=4,
    )
    metrics = CandidateRewardMetrics(
        osnr_margin=1.2,
        nli_share=0.15,
        worst_link_nli_share=0.3,
        fragmentation_damage_num_blocks=0.2,
        fragmentation_damage_largest_block=0.1,
    )

    value_from_input, breakdown_from_input = reward.evaluate(
        RewardInput(
            transition=transition,
            statistics=_snapshot(),
            selected_candidate_metrics=metrics,
            has_valid_non_reject_action=True,
        )
    )
    value_direct, breakdown_direct = reward.evaluate_transition(
        transition,
        selected_candidate_metrics=metrics,
        has_valid_non_reject_action=True,
    )

    assert value_direct == pytest.approx(value_from_input)
    assert breakdown_direct.raw_reward == pytest.approx(breakdown_from_input.raw_reward)
    assert breakdown_direct.clipped_reward == pytest.approx(breakdown_from_input.clipped_reward)
    assert breakdown_direct.fragmentation_penalty == pytest.approx(
        breakdown_from_input.fragmentation_penalty
    )
    assert breakdown_direct.physical_penalty == pytest.approx(breakdown_from_input.physical_penalty)
