from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from optical_networking_gym_v2.contracts import ActionSelection, CandidateRewardMetrics
from optical_networking_gym_v2.contracts.enums import MaskMode
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.runtime.action_codec import decode_action, encode_action, reject_action
from optical_networking_gym_v2.runtime.request_analysis import PATH_FEATURE_NAMES, RequestAnalysis
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState

if TYPE_CHECKING:
    from optical_networking_gym_v2.config.scenario import ScenarioConfig
    from optical_networking_gym_v2.contracts import ServiceRequest
    from optical_networking_gym_v2.envs.optical_env import OpticalEnv
    from optical_networking_gym_v2.runtime.simulator import Simulator


@dataclass(frozen=True, slots=True)
class RuntimeHeuristicContext:
    simulator: Simulator
    config: ScenarioConfig
    topology: TopologyModel
    state: RuntimeState
    request: ServiceRequest
    analysis: RequestAnalysis
    action_mask: np.ndarray | None

    @property
    def reject_action(self) -> int:
        return reject_action(self.config)

    def decode_action(self, action: int) -> ActionSelection | None:
        decoded = decode_action(self.config, action)
        if decoded is None:
            return None
        if decoded.path_index >= len(self.analysis.paths):
            return None
        if decoded.modulation_offset >= len(self.analysis.modulation_indices):
            return None
        return ActionSelection(
            path_index=int(decoded.path_index),
            modulation_index=int(self.analysis.modulation_indices[decoded.modulation_offset]),
            initial_slot=int(decoded.initial_slot),
        )

    def selected_candidate_metrics(self, action: int) -> CandidateRewardMetrics | None:
        selection = self.decode_action(action)
        if selection is None:
            return None
        return self.analysis.selected_candidate_metrics(
            path_index=selection.path_index,
            modulation_index=selection.modulation_index,
            initial_slot=selection.initial_slot,
        )


def _resolve_simulator(source: OpticalEnv | Simulator | RuntimeHeuristicContext):
    if isinstance(source, RuntimeHeuristicContext):
        return source.simulator

    simulator = getattr(source, "simulator", None)
    if simulator is not None and hasattr(simulator, "current_analysis") and hasattr(simulator, "topology"):
        return simulator
    if hasattr(source, "current_analysis") and hasattr(source, "topology") and hasattr(source, "config"):
        return source
    raise TypeError("runtime heuristic requires an OpticalEnv, Simulator, or RuntimeHeuristicContext")


def build_runtime_heuristic_context(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> RuntimeHeuristicContext:
    if isinstance(source, RuntimeHeuristicContext):
        return source

    simulator = _resolve_simulator(source)
    if simulator.state is None or simulator.current_request is None:
        raise RuntimeError("reset() must be called before evaluating runtime heuristics")

    analysis = simulator.current_analysis
    if analysis is None:
        analysis = simulator.analysis_engine.build(simulator.state, simulator.current_request)

    return RuntimeHeuristicContext(
        simulator=simulator,
        config=simulator.config,
        topology=simulator.topology,
        state=simulator.state,
        request=simulator.current_request,
        analysis=analysis,
        action_mask=simulator.current_mask,
    )


def _valid_start_flags(context: RuntimeHeuristicContext) -> np.ndarray:
    if context.config.mask_mode is MaskMode.RESOURCE_ONLY:
        return context.analysis.resource_valid_starts
    return context.analysis.qot_valid_starts


_PATH_LINK_UTIL_MEAN_INDEX = int(PATH_FEATURE_NAMES.index("path_link_util_mean"))
_PATH_LINK_UTIL_MAX_INDEX = int(PATH_FEATURE_NAMES.index("path_link_util_max"))
_PATH_COMMON_FREE_RATIO_INDEX = int(PATH_FEATURE_NAMES.index("path_common_free_ratio"))
_PATH_COMMON_NUM_BLOCKS_NORM_INDEX = int(PATH_FEATURE_NAMES.index("path_common_num_blocks_norm"))
_PATH_ROUTE_CUTS_NORM_INDEX = int(PATH_FEATURE_NAMES.index("path_route_cuts_norm"))
_PATH_ROUTE_RSS_INDEX = int(PATH_FEATURE_NAMES.index("path_route_rss"))
_LOCAL_FRAGMENTATION_INDEX = 4


def _modulation_offsets_by_descending_efficiency(context: RuntimeHeuristicContext) -> list[int]:
    return sorted(
        range(len(context.analysis.modulation_indices)),
        key=lambda modulation_offset: (
            -int(context.config.modulations[context.analysis.modulation_indices[modulation_offset]].spectral_efficiency),
            int(modulation_offset),
        ),
    )


def _legacy_path_load(context: RuntimeHeuristicContext, path_index: int) -> float:
    path = context.analysis.paths[path_index]
    if path.link_ids:
        link_indices = np.asarray(path.link_ids, dtype=np.intp)
        free_mask = np.all(context.state.slot_allocation[link_indices, :] == -1, axis=0)
    else:
        free_mask = np.ones(context.config.num_spectrum_resources, dtype=np.bool_)
    free_slots = float(free_mask.sum())
    hops = max(1, int(path.hops))
    return float(free_slots / np.sqrt(hops))


def select_jocn_ls_bm_ksp_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    modulation_offsets = _modulation_offsets_by_descending_efficiency(context)

    for initial_slot in range(context.config.num_spectrum_resources):
        for modulation_offset in modulation_offsets:
            for path_index, _path in enumerate(context.analysis.paths):
                if not bool(valid_flags[path_index, modulation_offset, initial_slot]):
                    continue
                return encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )

    return context.reject_action


def select_jocn_bm_ksp_lb_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    modulation_offsets = _modulation_offsets_by_descending_efficiency(context)

    for modulation_offset in modulation_offsets:
        best_key: tuple[float, int, int] | None = None
        best_action = context.reject_action
        for path_index, _path in enumerate(context.analysis.paths):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            if candidate_indices.size == 0:
                continue
            initial_slot = int(candidate_indices[0])
            action = encode_action(
                context.config,
                path_index=path_index,
                modulation_offset=modulation_offset,
                initial_slot=initial_slot,
            )
            candidate_key = (
                _legacy_path_load(context, path_index),
                int(path_index),
                int(initial_slot),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_action = int(action)
        if best_key is not None:
            return best_action

    return context.reject_action


def select_jocn_ksp_lb_bm_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    modulation_offsets = _modulation_offsets_by_descending_efficiency(context)
    best_key: tuple[float, int, int, int] | None = None
    best_action = context.reject_action

    for path_index, _path in enumerate(context.analysis.paths):
        current_load = _legacy_path_load(context, path_index)
        if best_key is not None and current_load >= best_key[0]:
            continue
        for modulation_offset in modulation_offsets:
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            if candidate_indices.size == 0:
                continue
            initial_slot = int(candidate_indices[0])
            action = encode_action(
                context.config,
                path_index=path_index,
                modulation_offset=modulation_offset,
                initial_slot=initial_slot,
            )
            candidate_key = (
                current_load,
                int(path_index),
                int(modulation_offset),
                int(initial_slot),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_action = int(action)
            break

    return best_action


def select_first_fit_decision(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> tuple[int, bool, bool]:
    context = build_runtime_heuristic_context(source)
    blocked_due_to_resources = False
    blocked_due_to_qot = False
    use_resource_only = context.config.mask_mode is MaskMode.RESOURCE_ONLY

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            resource_candidates = np.flatnonzero(
                context.analysis.resource_valid_starts[path_index, modulation_offset, :]
            )
            if resource_candidates.size == 0:
                blocked_due_to_resources = True
                continue

            if use_resource_only:
                return (
                    encode_action(
                        context.config,
                        path_index=path_index,
                        modulation_offset=modulation_offset,
                        initial_slot=int(resource_candidates[0]),
                    ),
                    False,
                    False,
                )

            qot_candidates = np.flatnonzero(context.analysis.qot_valid_starts[path_index, modulation_offset, :])
            if qot_candidates.size == 0:
                blocked_due_to_qot = True
                continue

            return (
                encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(qot_candidates[0]),
                ),
                False,
                False,
            )

    if blocked_due_to_qot:
        blocked_due_to_resources = False
    return context.reject_action, blocked_due_to_resources, blocked_due_to_qot


def select_first_fit_action(source: OpticalEnv | Simulator | RuntimeHeuristicContext) -> int:
    action, _, _ = select_first_fit_decision(source)
    return action


def _candidate_causes_disruption(
    context: RuntimeHeuristicContext,
    *,
    path_index: int,
    modulation_offset: int,
    initial_slot: int,
) -> bool:
    simulator = context.simulator
    state = context.state
    if not context.config.measure_disruptions:
        return False

    path = context.analysis.paths[path_index]
    modulation_index = int(context.analysis.modulation_indices[modulation_offset])
    modulation = context.config.modulations[modulation_index]
    service_num_slots = int(context.analysis.required_slots_by_path_mod[path_index, modulation_offset])
    if service_num_slots <= 0:
        return True

    occupied_slot_start = int(initial_slot)
    occupied_slot_end_exclusive = int(initial_slot + service_num_slots)
    if occupied_slot_end_exclusive < context.config.num_spectrum_resources:
        occupied_slot_end_exclusive += 1

    candidate = simulator.qot_engine.build_candidate(
        request=context.request,
        path=path,
        modulation=modulation,
        service_slot_start=int(initial_slot),
        service_num_slots=service_num_slots,
    )
    qot_result = simulator.qot_engine.evaluate_candidate(state, candidate)
    if not qot_result.meets_threshold:
        return True

    state.apply_provision(
        request=context.request,
        path=path,
        service_slot_start=int(initial_slot),
        service_num_slots=service_num_slots,
        occupied_slot_start=occupied_slot_start,
        occupied_slot_end_exclusive=occupied_slot_end_exclusive,
        modulation=modulation,
        center_frequency=candidate.center_frequency,
        bandwidth=candidate.bandwidth,
        launch_power=candidate.launch_power,
    )
    try:
        state.apply_qot_updates(
            {
                context.request.service_id: {
                    "osnr": qot_result.osnr,
                    "ase": qot_result.ase,
                    "nli": qot_result.nli,
                }
            }
        )
        impacted_ids = simulator.qot_engine.impacted_service_ids(
            state,
            path,
            exclude_service_id=context.request.service_id,
        )
        if not impacted_ids:
            return False
        for update in simulator.qot_engine.refresh_services(state, impacted_ids):
            if update.service_id not in state.active_services_by_id:
                continue
            service = state.active_services_by_id[update.service_id]
            if service.modulation is None:
                continue
            threshold = float(service.modulation.minimum_osnr + context.config.margin)
            if float(update.osnr) < threshold:
                return True
        return False
    finally:
        state.apply_release(context.request.service_id)


def select_disruption_aware_first_fit_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    use_resource_only = context.config.mask_mode is MaskMode.RESOURCE_ONLY

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            if use_resource_only:
                candidate_indices = np.flatnonzero(
                    context.analysis.resource_valid_starts[path_index, modulation_offset, :]
                )
            else:
                candidate_indices = np.flatnonzero(
                    context.analysis.qot_valid_starts[path_index, modulation_offset, :]
                )
            for initial_slot in candidate_indices.tolist():
                if _candidate_causes_disruption(
                    context,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                ):
                    continue
                return encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )

    return context.reject_action


def select_random_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    context = build_runtime_heuristic_context(source)
    generator = rng if rng is not None else np.random.default_rng()
    valid_flags = _valid_start_flags(context)
    selected_action = context.reject_action
    valid_count = 0

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices:
                valid_count += 1
                if int(generator.integers(valid_count)) != 0:
                    continue
                selected_action = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )

    return selected_action


def select_load_balancing_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    best_key: tuple[float, float, float, float, float, int] | None = None
    best_action = context.reject_action

    for path_index, _path in enumerate(context.analysis.paths):
        path_features = context.analysis.path_features[path_index]
        path_link_util_mean = float(path_features[_PATH_LINK_UTIL_MEAN_INDEX])
        path_link_util_max = float(path_features[_PATH_LINK_UTIL_MAX_INDEX])
        path_common_free_ratio = float(path_features[_PATH_COMMON_FREE_RATIO_INDEX])

        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices.tolist():
                action = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )
                metrics = context.selected_candidate_metrics(action)
                if metrics is None:
                    continue
                candidate_key = (
                    path_link_util_max,
                    path_link_util_mean,
                    -path_common_free_ratio,
                    float(metrics.fragmentation_damage_num_blocks),
                    float(metrics.fragmentation_damage_largest_block),
                    int(action),
                )
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_action = int(action)

    return best_action


def select_highest_snr_first_fit_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    best_key: tuple[float, int, int, int, int] | None = None
    best_action = context.reject_action

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices.tolist():
                action = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )
                metrics = context.selected_candidate_metrics(action)
                if metrics is None:
                    continue
                candidate_key = (
                    float(metrics.osnr_margin),
                    -int(path_index),
                    -int(modulation_offset),
                    -int(initial_slot),
                    -int(action),
                )
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_action = int(action)

    return best_action


def select_ksp_best_mod_last_fit_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    modulation_offsets = _modulation_offsets_by_descending_efficiency(context)

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset in modulation_offsets:
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            if candidate_indices.size == 0:
                continue
            return encode_action(
                context.config,
                path_index=path_index,
                modulation_offset=modulation_offset,
                initial_slot=int(candidate_indices[-1]),
            )

    return context.reject_action


def select_lowest_fragmentation_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    best_key: tuple[float, ...] | None = None
    best_action = context.reject_action

    for path_index, _path in enumerate(context.analysis.paths):
        path_features = context.analysis.path_features[path_index]
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices.tolist():
                action = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )
                metrics = context.selected_candidate_metrics(action)
                if metrics is None:
                    continue
                slot_features = context.analysis.path_slot_features[path_index, int(initial_slot)]
                candidate_key = (
                    float(metrics.fragmentation_damage_num_blocks),
                    float(metrics.fragmentation_damage_largest_block),
                    float(slot_features[_LOCAL_FRAGMENTATION_INDEX]),
                    float(path_features[_PATH_COMMON_NUM_BLOCKS_NORM_INDEX]),
                    float(path_features[_PATH_ROUTE_CUTS_NORM_INDEX]),
                    -float(path_features[_PATH_ROUTE_RSS_INDEX]),
                    float(context.analysis.required_slots_by_path_mod[path_index, modulation_offset]),
                    float(path_features[_PATH_LINK_UTIL_MAX_INDEX]),
                    -float(np.clip(metrics.osnr_margin, 0.0, 3.0)),
                    float(action),
                )
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_action = int(action)

    return best_action


__all__ = [
    "RuntimeHeuristicContext",
    "build_runtime_heuristic_context",
    "select_disruption_aware_first_fit_action",
    "select_first_fit_action",
    "select_first_fit_decision",
    "select_highest_snr_first_fit_action",
    "select_jocn_ls_bm_ksp_action",
    "select_jocn_bm_ksp_lb_action",
    "select_jocn_ksp_lb_bm_action",
    "select_ksp_best_mod_last_fit_action",
    "select_load_balancing_action",
    "select_lowest_fragmentation_action",
    "select_random_action",
]
