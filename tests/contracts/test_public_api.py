from __future__ import annotations

from optical_networking_gym_v2 import (
    ActionMask,
    ActionSelection,
    Allocation,
    CandidateRewardMetrics,
    MaskMode,
    Modulation,
    Observation,
    ObservationSchema,
    ObservationSnapshot,
    OpticalEnv,
    QoTEngine,
    QoTResult,
    RewardBreakdown,
    RewardFunction,
    RewardInput,
    RewardProfile,
    RuntimeHeuristicContext,
    RequestAnalysisEngine,
    ServiceRequest,
    Simulator,
    Statistics,
    StatisticsSnapshot,
    Status,
    StepInfo,
    StepTransition,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
    build_scenario,
    build_scenario_grid,
    build_runtime_heuristic_context,
    iter_scenarios,
    list_scenarios,
    select_highest_snr_first_fit_runtime_action,
    select_ksp_best_mod_last_fit_runtime_action,
    select_load_balancing_runtime_action,
    select_lowest_fragmentation_runtime_action,
    select_random_action,
    select_random_runtime_action,
)


def test_public_api_exports_contracts() -> None:
    assert ActionMask is not None
    assert ActionSelection is not None
    assert ServiceRequest is not None
    assert TrafficRecord is not None
    assert TrafficTable is not None
    assert Allocation is not None
    assert CandidateRewardMetrics is not None
    assert Modulation is not None
    assert Observation is not None
    assert ObservationSchema is not None
    assert ObservationSnapshot is not None
    assert OpticalEnv is not None
    assert QoTEngine is not None
    assert QoTResult is not None
    assert RewardBreakdown is not None
    assert RewardFunction is not None
    assert RewardInput is not None
    assert RewardProfile.BALANCED.value == "balanced"
    assert RequestAnalysisEngine is not None
    assert RuntimeHeuristicContext is not None
    assert Simulator is not None
    assert Statistics is not None
    assert StatisticsSnapshot is not None
    assert StepInfo is not None
    assert StepTransition is not None
    assert build_runtime_heuristic_context is not None
    assert build_scenario is not None
    assert build_scenario_grid is not None
    assert iter_scenarios is not None
    assert list_scenarios is not None
    assert select_highest_snr_first_fit_runtime_action is not None
    assert select_ksp_best_mod_last_fit_runtime_action is not None
    assert select_load_balancing_runtime_action is not None
    assert select_lowest_fragmentation_runtime_action is not None
    assert select_random_action is not None
    assert select_random_runtime_action is not None
    assert TrafficMode.DYNAMIC.value == "dynamic"
    assert MaskMode.RESOURCE_AND_QOT.value == "resource_and_qot"
    assert Status.BLOCKED_QOT.value == "blocked_qot"
