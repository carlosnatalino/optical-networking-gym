from __future__ import annotations

from .api.factory import make_env
from .bench.benchmarking import (
    benchmark_action_mask,
    benchmark_allocation,
    benchmark_observation,
    benchmark_qot_engine,
    benchmark_request_analysis,
    benchmark_reward_function,
    benchmark_runtime_state,
    benchmark_statistics_step_info,
)
from .bench.integrated_benchmarking import benchmark_simulator_episode, profile_simulator_episode
from .config.scenario import ScenarioConfig
from .contracts.action_mask import ActionSelection
from .contracts.allocation import Allocation
from .contracts.enums import MaskMode, RewardProfile, Status, TrafficMode
from .contracts.modulation import Modulation
from .contracts.observation import ObservationSchema, ObservationSnapshot
from .contracts.qot import QoTRequest, QoTResult, ServiceQoTUpdate
from .contracts.reward import CandidateRewardMetrics, RewardBreakdown, RewardInput
from .contracts.step import StatisticsSnapshot, StepTransition
from .contracts.traffic import ServiceRequest, TrafficRecord, TrafficTable
from .defaults import (
    BUILTIN_TOPOLOGY_DIR,
    MODULATION_CATALOG,
    get_modulations,
    resolve_topology,
    set_topology_dir,
)
from .envs.optical_env import OpticalEnv
from .features.action_mask import ActionMask
from .features.observation import Observation
from .features.reward_function import RewardFunction
from .heuristics.masked_heuristics import select_first_fit_action, select_random_action
from .heuristics.runtime_heuristics import (
    RuntimeHeuristicContext,
    build_runtime_heuristic_context,
    select_disruption_aware_first_fit_action,
    select_first_fit_action as select_first_fit_runtime_action,
    select_highest_snr_first_fit_action as select_highest_snr_first_fit_runtime_action,
    select_jocn_bm_ksp_lb_action,
    select_jocn_ksp_lb_bm_action,
    select_jocn_ls_bm_ksp_action,
    select_ksp_best_mod_last_fit_action as select_ksp_best_mod_last_fit_runtime_action,
    select_load_balancing_action as select_load_balancing_runtime_action,
    select_lowest_fragmentation_action as select_lowest_fragmentation_runtime_action,
    select_random_action as select_random_runtime_action,
)
from .network.allocation import (
    available_slots_for_path,
    build_first_fit_allocation,
    candidate_starts,
    compute_required_slots,
    occupied_slot_range,
    path_is_free,
)
from .network.topology import Link, PathRecord, Span, TopologyModel
from .network.traffic_table_io import read_traffic_table_jsonl, write_traffic_table_jsonl
from .optical.first_fit import (
    select_first_fit_action_from_env,
    shortest_available_path_first_fit_best_modulation,
)
from .optical.first_fit_example import run_episode as run_first_fit_episode
from .optical.qot_engine import QoTEngine
from .runtime.request_analysis import RequestAnalysis, RequestAnalysisEngine
from .runtime.runtime_state import ActiveService, RuntimeState
from .runtime.simulator import Simulator
from .runtime.step_info import StepInfo
from .runtime.traffic_model import TrafficModel
from .scenarios import build_scenario, build_scenario_grid, iter_scenarios, list_scenarios
from .stats.statistics import Statistics

__all__ = [
    "ActionMask",
    "ActionSelection",
    "ActiveService",
    "Allocation",
    "BUILTIN_TOPOLOGY_DIR",
    "CandidateRewardMetrics",
    "Link",
    "MODULATION_CATALOG",
    "MaskMode",
    "Modulation",
    "Observation",
    "ObservationSchema",
    "ObservationSnapshot",
    "OpticalEnv",
    "PathRecord",
    "QoTEngine",
    "QoTRequest",
    "QoTResult",
    "RequestAnalysis",
    "RequestAnalysisEngine",
    "RewardBreakdown",
    "RewardFunction",
    "RewardInput",
    "RewardProfile",
    "RuntimeHeuristicContext",
    "RuntimeState",
    "ScenarioConfig",
    "ServiceQoTUpdate",
    "ServiceRequest",
    "Simulator",
    "Span",
    "Statistics",
    "StatisticsSnapshot",
    "Status",
    "StepInfo",
    "StepTransition",
    "TopologyModel",
    "TrafficMode",
    "TrafficModel",
    "TrafficRecord",
    "TrafficTable",
    "available_slots_for_path",
    "benchmark_action_mask",
    "benchmark_allocation",
    "benchmark_observation",
    "benchmark_qot_engine",
    "benchmark_request_analysis",
    "benchmark_reward_function",
    "benchmark_runtime_state",
    "benchmark_simulator_episode",
    "benchmark_statistics_step_info",
    "build_first_fit_allocation",
    "build_runtime_heuristic_context",
    "build_scenario",
    "build_scenario_grid",
    "candidate_starts",
    "compute_required_slots",
    "get_modulations",
    "iter_scenarios",
    "list_scenarios",
    "make_env",
    "occupied_slot_range",
    "path_is_free",
    "profile_simulator_episode",
    "read_traffic_table_jsonl",
    "resolve_topology",
    "run_first_fit_episode",
    "select_disruption_aware_first_fit_action",
    "select_first_fit_action",
    "select_first_fit_action_from_env",
    "select_first_fit_runtime_action",
    "select_highest_snr_first_fit_runtime_action",
    "select_jocn_bm_ksp_lb_action",
    "select_jocn_ksp_lb_bm_action",
    "select_jocn_ls_bm_ksp_action",
    "select_ksp_best_mod_last_fit_runtime_action",
    "select_load_balancing_runtime_action",
    "select_lowest_fragmentation_runtime_action",
    "select_random_action",
    "select_random_runtime_action",
    "set_topology_dir",
    "shortest_available_path_first_fit_best_modulation",
    "write_traffic_table_jsonl",
]
