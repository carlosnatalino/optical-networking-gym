from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from optical_networking_gym_v2.contracts import CandidateRewardMetrics, MaskMode, ServiceRequest
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState
from optical_networking_gym_v2.network.allocation import compute_required_slots
from optical_networking_gym_v2.network.topology import PathRecord, TopologyModel
from optical_networking_gym_v2.optical.kernels.allocation_kernel import candidate_starts_array
from optical_networking_gym_v2.optical.qot_engine import QoTEngine

try:
    from optical_networking_gym_v2.simulation import _request_analysis_kernels as _request_analysis_kernels_module
except ImportError:
    _request_analysis_kernels_module = None

_analyze_free_mask_kernel = getattr(_request_analysis_kernels_module, "analyze_free_mask_kernel", None)
_build_link_metrics_kernel = getattr(_request_analysis_kernels_module, "build_link_metrics_kernel", None)
_build_global_features_kernel = getattr(_request_analysis_kernels_module, "build_global_features_kernel", None)
_build_path_features_kernel = getattr(_request_analysis_kernels_module, "build_path_features_kernel", None)
_build_path_mod_features_kernel = getattr(
    _request_analysis_kernels_module,
    "build_path_mod_features_kernel",
    None,
)
_build_path_slot_features_kernel = getattr(
    _request_analysis_kernels_module,
    "build_path_slot_features_kernel",
    None,
)
_fragmentation_damage_by_candidates_kernel = getattr(
    _request_analysis_kernels_module,
    "fragmentation_damage_by_candidates_kernel",
    None,
)
_summary_after_allocation_kernel = getattr(
    _request_analysis_kernels_module,
    "summary_after_allocation_kernel",
    None,
)


REQUEST_FEATURE_NAMES = ("bit_rate_norm",)
GLOBAL_FEATURE_NAMES = (
    "network_util_mean",
    "network_util_std",
    "network_util_max",
    "free_slots_ratio",
    "mean_link_entropy",
    "mean_link_external_fragmentation",
    "mean_link_compactness",
    "active_services_norm",
)
PATH_FEATURE_NAMES = (
    "path_length_norm",
    "path_hops_norm",
    "path_link_util_mean",
    "path_link_util_max",
    "path_common_free_ratio",
    "path_common_largest_block_ratio",
    "path_common_num_blocks_norm",
    "path_common_entropy",
    "path_route_cuts_norm",
    "path_route_rss",
    "path_link_entropy_mean",
    "path_external_fragmentation_mean",
    "path_compactness_mean",
)
PATH_MOD_FEATURE_NAMES = (
    "required_slots_norm",
    "resource_candidate_ratio",
    "qot_candidate_ratio",
    "first_qot_start_norm",
    "last_qot_start_norm",
    "best_osnr_margin_norm",
    "best_nli_share",
    "best_worst_link_nli_share",
    "best_fragmentation_damage_num_blocks",
    "best_fragmentation_damage_largest_block",
)
PATH_SLOT_FEATURE_NAMES = (
    "is_common_free",
    "common_block_length_norm",
    "left_free_span_norm",
    "right_free_span_norm",
    "local_fragmentation",
    "is_candidate_start_resource_any",
    "is_candidate_start_qot_any",
    "best_slot_osnr_margin_norm",
    "best_slot_nli_share",
)


@dataclass(slots=True)
class _BlockSummary:
    count: int
    largest: int
    total_free: int
    entropy: float
    rss: float


@dataclass(slots=True)
class _FreeRunAnalysis:
    summary: _BlockSummary
    run_starts: np.ndarray
    run_ends: np.ndarray
    run_lengths: np.ndarray
    slot_to_run_index: np.ndarray
    largest_other_by_run: np.ndarray
    sum_squares: float
    sum_length_log_length: float


@dataclass(slots=True)
class RequestAnalysisInspection:
    common_free_masks: np.ndarray
    link_metrics: np.ndarray


@dataclass(slots=True)
class RequestAnalysis:
    config: ScenarioConfig
    topology: TopologyModel
    state_id: int
    allocation_state_version: int
    request: ServiceRequest
    paths: tuple[PathRecord, ...]
    modulation_indices: tuple[int, ...]
    resource_valid_starts: np.ndarray
    qot_valid_starts: np.ndarray
    osnr_margin_by_start: np.ndarray
    nli_share_by_start: np.ndarray
    worst_link_nli_share_by_start: np.ndarray
    fragmentation_damage_num_blocks_by_start: np.ndarray
    fragmentation_damage_largest_block_by_start: np.ndarray
    required_slots_by_path_mod: np.ndarray
    action_mask: np.ndarray | None
    mean_link_entropy: float
    path_route_cuts_norm_by_path: np.ndarray
    path_route_rss_by_path: np.ndarray
    free_slots_ratio: float
    active_services_norm: float
    inspection: RequestAnalysisInspection | None = None
    _request_features: np.ndarray | None = None
    _global_features: np.ndarray | None = None
    _path_features: np.ndarray | None = None
    _path_mod_features: np.ndarray | None = None
    _path_slot_features: np.ndarray | None = None

    @property
    def request_features(self) -> np.ndarray:
        if self._request_features is None:
            max_spectral_efficiency = max(
                modulation.spectral_efficiency for modulation in self.config.modulations
            )
            max_bit_rate = (
                self.config.num_spectrum_resources * self.config.channel_width * max_spectral_efficiency
            )
            self._request_features = np.array(
                [float(np.clip(self.request.bit_rate / max_bit_rate, 0.0, 1.0))],
                dtype=np.float32,
            )
        return self._request_features

    @property
    def global_features(self) -> np.ndarray:
        if self._global_features is None:
            self._global_features = _build_global_features(self)
        return self._global_features

    @property
    def path_features(self) -> np.ndarray:
        if self._path_features is None:
            self._path_features = _build_path_features(self)
        return self._path_features

    @property
    def path_mod_features(self) -> np.ndarray:
        if self._path_mod_features is None:
            self._path_mod_features = _build_path_mod_features(self)
        return self._path_mod_features

    @property
    def path_slot_features(self) -> np.ndarray:
        if self._path_slot_features is None:
            self._path_slot_features = _build_path_slot_features(self)
        return self._path_slot_features

    @property
    def common_free_masks(self) -> np.ndarray:
        if self.inspection is None:
            raise RuntimeError("common_free_masks are unavailable when observation inspection is disabled")
        return self.inspection.common_free_masks

    @property
    def link_metrics(self) -> np.ndarray:
        if self.inspection is None:
            raise RuntimeError("link_metrics are unavailable when observation inspection is disabled")
        return self.inspection.link_metrics

    @property
    def has_valid_non_reject_action(self) -> bool:
        if self.action_mask is not None:
            return bool(self.action_mask.any())
        if self.config.mask_mode is MaskMode.RESOURCE_ONLY:
            return bool(self.resource_valid_starts.any())
        return bool(self.qot_valid_starts.any())

    def modulation_offset_for_index(self, modulation_index: int) -> int | None:
        try:
            return self.modulation_indices.index(modulation_index)
        except ValueError:
            return None

    def selected_candidate_metrics(
        self,
        *,
        path_index: int,
        modulation_index: int,
        initial_slot: int,
    ) -> CandidateRewardMetrics | None:
        modulation_offset = self.modulation_offset_for_index(modulation_index)
        if modulation_offset is None:
            return None
        if path_index < 0 or path_index >= self.config.k_paths:
            return None
        if initial_slot < 0 or initial_slot >= self.config.num_spectrum_resources:
            return None
        if not self.resource_valid_starts[path_index, modulation_offset, initial_slot]:
            return None
        osnr_margin = float(self.osnr_margin_by_start[path_index, modulation_offset, initial_slot])
        nli_share = float(self.nli_share_by_start[path_index, modulation_offset, initial_slot])
        worst_link_nli_share = float(
            self.worst_link_nli_share_by_start[path_index, modulation_offset, initial_slot]
        )
        if math.isnan(osnr_margin):
            osnr_margin = 0.0
        if math.isnan(nli_share):
            nli_share = 0.0
        if math.isnan(worst_link_nli_share):
            worst_link_nli_share = 0.0
        return CandidateRewardMetrics(
            osnr_margin=osnr_margin,
            nli_share=nli_share,
            worst_link_nli_share=worst_link_nli_share,
            fragmentation_damage_num_blocks=float(
                self.fragmentation_damage_num_blocks_by_start[path_index, modulation_offset, initial_slot]
            ),
            fragmentation_damage_largest_block=float(
                self.fragmentation_damage_largest_block_by_start[path_index, modulation_offset, initial_slot]
            ),
        )


class RequestAnalysisEngine:
    def __init__(self, config: ScenarioConfig, topology: TopologyModel, qot_engine: QoTEngine) -> None:
        if not config.modulations:
            raise ValueError("RequestAnalysisEngine requires ScenarioConfig.modulations")
        if config.modulations_to_consider <= 0:
            raise ValueError("RequestAnalysisEngine requires modulations_to_consider > 0")
        self.config = config
        self.topology = topology
        self.qot_engine = qot_engine
        self.cache_hits = 0
        self.cache_misses = 0
        self._analysis_cache: dict[tuple[int, int, int, int, float], RequestAnalysis] = {}
        self._path_link_indices: dict[int, np.ndarray] = {
            path.id: np.asarray(path.link_ids, dtype=np.intp) for path in topology.paths
        }

    @property
    def request_feature_names(self) -> tuple[str, ...]:
        return REQUEST_FEATURE_NAMES

    @property
    def global_feature_names(self) -> tuple[str, ...]:
        return GLOBAL_FEATURE_NAMES

    @property
    def path_feature_names(self) -> tuple[str, ...]:
        return PATH_FEATURE_NAMES

    @property
    def path_mod_feature_names(self) -> tuple[str, ...]:
        return PATH_MOD_FEATURE_NAMES

    @property
    def path_slot_feature_names(self) -> tuple[str, ...]:
        return PATH_SLOT_FEATURE_NAMES

    def build(
        self,
        state: RuntimeState,
        request: ServiceRequest,
        *,
        include_inspection: bool = False,
    ) -> RequestAnalysis:
        needs_inspection = include_inspection or self.config.enable_observation
        cache_key = (
            state.state_id,
            state.allocation_state_version,
            request.source_id,
            request.destination_id,
            float(request.bit_rate),
            needs_inspection,
        )
        cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        analysis = self._build_analysis(state, request, include_inspection=needs_inspection)
        self._analysis_cache[cache_key] = analysis
        self.cache_misses += 1
        return analysis

    def clear_cache(self) -> None:
        self._analysis_cache.clear()

    def _build_analysis(
        self,
        state: RuntimeState,
        request: ServiceRequest,
        *,
        include_inspection: bool,
    ) -> RequestAnalysis:
        paths = self.topology.get_paths_by_ids(request.source_id, request.destination_id)[: self.config.k_paths]
        total_slots = self.config.num_spectrum_resources
        max_paths = self.config.k_paths
        full_modulation_count = len(self.config.modulations)
        selected_count = self.config.modulations_to_consider
        block_count_scale = max(1, math.ceil(total_slots / 2))
        free_slots_ratio = float(np.count_nonzero(state.slot_allocation == -1)) / state.slot_allocation.size
        active_services_norm = min(
            len(state.active_services_by_id) / max(1, self.topology.link_count * self.config.num_spectrum_resources),
            1.0,
        )
        required_slots_by_modulation = np.asarray(
            [
                compute_required_slots(
                    bit_rate=request.bit_rate,
                    spectral_efficiency=modulation.spectral_efficiency,
                    channel_width=self.config.channel_width,
                )
                for modulation in self.config.modulations
            ],
            dtype=np.int16,
        )

        common_free_masks = np.zeros((max_paths, total_slots), dtype=np.bool_)
        link_metrics = _build_link_metrics(state, self.topology, total_slots)
        mean_link_entropy = float(link_metrics[:, 1].mean()) if link_metrics.size else 0.0
        path_free_runs: list[_FreeRunAnalysis] = []

        for path_index, path in enumerate(paths):
            if not path.link_ids:
                common_free_masks[path_index, :] = True
            else:
                common_free_masks[path_index, :] = np.all(
                    state.slot_allocation[self._path_link_indices[path.id], :] == -1,
                    axis=0,
                )
            path_free_runs.append(_analyze_free_mask(common_free_masks[path_index, :]))

        resource_valid_full = np.zeros((len(paths), full_modulation_count, total_slots), dtype=np.bool_)
        qot_valid_full = np.zeros((len(paths), full_modulation_count, total_slots), dtype=np.bool_)
        osnr_margin_full = np.full((len(paths), full_modulation_count, total_slots), np.nan, dtype=np.float32)
        nli_share_full = np.full((len(paths), full_modulation_count, total_slots), np.nan, dtype=np.float32)
        worst_link_nli_share_full = np.full(
            (len(paths), full_modulation_count, total_slots),
            np.nan,
            dtype=np.float32,
        )
        required_slots_full = np.zeros((len(paths), full_modulation_count), dtype=np.int16)
        fragmentation_damage_num_blocks_full = np.zeros(
            (len(paths), full_modulation_count, total_slots),
            dtype=np.float32,
        )
        fragmentation_damage_largest_block_full = np.zeros(
            (len(paths), full_modulation_count, total_slots),
            dtype=np.float32,
        )

        prepared_qot_inputs_by_path: list[object | None] = []
        for path in paths:
            if self.config.mask_mode is MaskMode.RESOURCE_ONLY or self.config.qot_constraint == "DIST":
                prepared_qot_inputs_by_path.append(None)
                continue
            prepared_qot_inputs_by_path.append(self.qot_engine._prepare_candidate_summary_inputs(state, path))

        max_feasible_modulation_index: int | None = None
        lowest_required_modulation_index = 0
        for modulation_index in range(full_modulation_count - 1, -1, -1):
            modulation = self.config.modulations[modulation_index]
            required_slots = int(required_slots_by_modulation[modulation_index])
            modulation_has_feasible_path = False

            for path_index, path in enumerate(paths):
                free_mask = common_free_masks[path_index, :]
                common_analysis = path_free_runs[path_index]
                required_slots_full[path_index, modulation_index] = required_slots
                if required_slots > total_slots:
                    continue

                candidate_indices = candidate_starts_array(free_mask, required_slots)
                if candidate_indices.size == 0:
                    continue
                resource_valid_full[path_index, modulation_index, candidate_indices] = True

                (
                    fragmentation_damage_num_blocks_full[path_index, modulation_index, candidate_indices],
                    fragmentation_damage_largest_block_full[path_index, modulation_index, candidate_indices],
                ) = _fragmentation_damage_by_candidates(
                    free_runs=common_analysis,
                    candidate_indices=candidate_indices,
                    service_num_slots=required_slots,
                    total_slots=total_slots,
                    block_count_scale=block_count_scale,
                )

                if self.config.mask_mode is MaskMode.RESOURCE_ONLY:
                    modulation_has_feasible_path = True
                    continue

                if self.config.qot_constraint == "DIST":
                    if path.length_km <= modulation.maximum_length:
                        qot_valid_full[path_index, modulation_index, candidate_indices] = True
                        modulation_has_feasible_path = True
                    continue

                batch = self.qot_engine._summarize_candidate_starts_prepared(
                    prepared_inputs=prepared_qot_inputs_by_path[path_index],
                    service_id=request.service_id,
                    service_num_slots=required_slots,
                    candidate_starts=candidate_indices,
                    threshold=modulation.minimum_osnr + self.config.margin,
                )
                osnr_margin_full[path_index, modulation_index, candidate_indices] = batch.osnr_margin
                nli_share_full[path_index, modulation_index, candidate_indices] = batch.nli_share
                worst_link_nli_share_full[path_index, modulation_index, candidate_indices] = (
                    batch.worst_link_nli_share
                )
                qot_valid_full[path_index, modulation_index, candidate_indices] = batch.meets_threshold
                if batch.meets_threshold.any():
                    modulation_has_feasible_path = True

            if modulation_has_feasible_path and max_feasible_modulation_index is None:
                max_feasible_modulation_index = modulation_index
                lowest_required_modulation_index = max(
                    0,
                    modulation_index - (selected_count - 1),
                )

            if (
                max_feasible_modulation_index is not None
                and modulation_index <= lowest_required_modulation_index
            ):
                break

        modulation_indices = _modulation_window_from_max_feasible(
            config=self.config,
            max_feasible_modulation_index=max_feasible_modulation_index,
        )
        selected_positions = np.asarray(modulation_indices, dtype=np.intp)
        resource_valid = _pad_array(
            resource_valid_full[:, selected_positions, :],
            (max_paths, selected_count, total_slots),
            False,
        )
        qot_valid = _pad_array(
            qot_valid_full[:, selected_positions, :],
            (max_paths, selected_count, total_slots),
            False,
        )
        osnr_margin = _pad_array(
            osnr_margin_full[:, selected_positions, :],
            (max_paths, selected_count, total_slots),
            np.nan,
        )
        nli_share = _pad_array(
            nli_share_full[:, selected_positions, :],
            (max_paths, selected_count, total_slots),
            np.nan,
        )
        worst_link_nli_share = _pad_array(
            worst_link_nli_share_full[:, selected_positions, :],
            (max_paths, selected_count, total_slots),
            np.nan,
        )
        required_slots = _pad_array(
            required_slots_full[:, selected_positions],
            (max_paths, selected_count),
            0,
        )
        fragmentation_damage_num_blocks = _pad_array(
            fragmentation_damage_num_blocks_full[:, selected_positions, :],
            (max_paths, selected_count, total_slots),
            0.0,
        )
        fragmentation_damage_largest_block = _pad_array(
            fragmentation_damage_largest_block_full[:, selected_positions, :],
            (max_paths, selected_count, total_slots),
            0.0,
        )

        path_route_cuts_norm_by_path = np.zeros(max_paths, dtype=np.float32)
        path_route_rss_by_path = np.zeros(max_paths, dtype=np.float32)
        for path_index, path in enumerate(paths):
            if not path.link_ids:
                continue
            route_cuts_sum = 0.0
            route_rss_sum = 0.0
            link_count = len(path.link_ids)
            for link_id in path.link_ids:
                metrics = link_metrics[int(link_id)]
                route_cuts_sum += float(metrics[4])
                route_rss_sum += float(metrics[5])
            path_route_cuts_norm_by_path[path_index] = route_cuts_sum / link_count
            path_route_rss_by_path[path_index] = route_rss_sum / link_count

        action_mask: np.ndarray | None = None
        if self.config.enable_action_mask:
            action_mask = np.zeros(max_paths * selected_count * total_slots, dtype=np.uint8)
            path_stride = selected_count * total_slots
            for path_index in range(max_paths):
                for modulation_offset in range(selected_count):
                    base_index = (path_index * path_stride) + (modulation_offset * total_slots)
                    flags = (
                        resource_valid[path_index, modulation_offset, :]
                        if self.config.mask_mode is MaskMode.RESOURCE_ONLY
                        else qot_valid[path_index, modulation_offset, :]
                    )
                    action_mask[base_index : base_index + total_slots] = flags

        inspection: RequestAnalysisInspection | None = None
        if include_inspection:
            inspection = RequestAnalysisInspection(
                common_free_masks=common_free_masks,
                link_metrics=link_metrics,
            )

        return RequestAnalysis(
            config=self.config,
            topology=self.topology,
            state_id=state.state_id,
            allocation_state_version=state.allocation_state_version,
            request=request,
            paths=tuple(paths),
            modulation_indices=modulation_indices,
            resource_valid_starts=resource_valid,
            qot_valid_starts=qot_valid,
            osnr_margin_by_start=osnr_margin,
            nli_share_by_start=nli_share,
            worst_link_nli_share_by_start=worst_link_nli_share,
            fragmentation_damage_num_blocks_by_start=fragmentation_damage_num_blocks,
            fragmentation_damage_largest_block_by_start=fragmentation_damage_largest_block,
            required_slots_by_path_mod=required_slots,
            action_mask=action_mask,
            mean_link_entropy=mean_link_entropy,
            path_route_cuts_norm_by_path=path_route_cuts_norm_by_path,
            path_route_rss_by_path=path_route_rss_by_path,
            free_slots_ratio=free_slots_ratio,
            active_services_norm=active_services_norm,
            inspection=inspection,
        )


def _pad_array(array: np.ndarray, target_shape: tuple[int, ...], fill_value: float | bool | int) -> np.ndarray:
    result = np.full(target_shape, fill_value=fill_value, dtype=array.dtype)
    slices = tuple(slice(0, size) for size in array.shape)
    result[slices] = array
    return result


def _fragmentation_damage_by_candidates(
    *,
    free_runs: _FreeRunAnalysis,
    candidate_indices: np.ndarray,
    service_num_slots: int,
    total_slots: int,
    block_count_scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    if _fragmentation_damage_by_candidates_kernel is not None:
        return _fragmentation_damage_by_candidates_kernel(
            candidate_indices,
            free_runs.slot_to_run_index,
            free_runs.run_starts,
            free_runs.run_ends,
            free_runs.largest_other_by_run,
            free_runs.summary.count,
            free_runs.summary.largest,
            service_num_slots,
            total_slots,
            float(block_count_scale),
            float(total_slots),
        )

    num_blocks_damage = np.zeros(candidate_indices.shape[0], dtype=np.float32)
    largest_block_damage = np.zeros(candidate_indices.shape[0], dtype=np.float32)
    summary_count = free_runs.summary.count
    summary_largest = free_runs.summary.largest
    slot_to_run_index = free_runs.slot_to_run_index
    run_starts = free_runs.run_starts
    run_ends = free_runs.run_ends
    largest_other_by_run = free_runs.largest_other_by_run

    for offset, initial_slot in enumerate(candidate_indices):
        run_index = int(slot_to_run_index[int(initial_slot)])
        if run_index < 0:
            continue
        run_start = int(run_starts[run_index])
        run_end = int(run_ends[run_index])
        removed_end = int(initial_slot) + service_num_slots
        if removed_end < total_slots:
            removed_end += 1
        left_length = max(int(initial_slot) - run_start, 0)
        right_length = max(run_end - removed_end, 0)
        post_count = summary_count - 1 + int(left_length > 0) + int(right_length > 0)
        post_largest = max(int(largest_other_by_run[run_index]), left_length, right_length)
        if post_count > summary_count:
            num_blocks_damage[offset] = (post_count - summary_count) / block_count_scale
        if summary_largest > post_largest:
            largest_block_damage[offset] = (summary_largest - post_largest) / total_slots
    return num_blocks_damage, largest_block_damage


def _modulation_window_from_max_feasible(
    *,
    config: ScenarioConfig,
    max_feasible_modulation_index: int | None,
) -> tuple[int, ...]:
    if max_feasible_modulation_index is None:
        max_feasible_modulation_index = config.modulations_to_consider - 1
    else:
        max_feasible_modulation_index = max(
            max_feasible_modulation_index,
            config.modulations_to_consider - 1,
        )

    start_index = max(0, max_feasible_modulation_index - (config.modulations_to_consider - 1))
    modulation_indices = tuple(
        reversed(
            tuple(range(start_index, max_feasible_modulation_index + 1))[: config.modulations_to_consider]
        )
    )
    if len(modulation_indices) != config.modulations_to_consider:
        return tuple(range(config.modulations_to_consider - 1, -1, -1))
    return modulation_indices


def _build_link_metrics(state: RuntimeState, topology: TopologyModel, total_slots: int) -> np.ndarray:
    if _build_link_metrics_kernel is not None:
        return _build_link_metrics_kernel(state.slot_allocation, total_slots)

    metrics = np.zeros((topology.link_count, 6), dtype=np.float32)
    max_block_count = max(1, math.ceil(total_slots / 2))

    for link_id in range(topology.link_count):
        slot_row = state.slot_allocation[link_id, :]
        free_mask = slot_row == -1
        free_summary = _free_block_summary(free_mask)
        occupied_slots = 0
        first_used = -1
        last_used = -1
        for slot_index in range(total_slots):
            if int(slot_row[slot_index]) == -1:
                continue
            occupied_slots += 1
            if first_used < 0:
                first_used = slot_index
            last_used = slot_index

        if occupied_slots == 0 or occupied_slots == total_slots:
            compactness = 1.0
        else:
            span_width = (last_used - first_used) + 1
            compactness = occupied_slots / span_width if span_width > 0 else 1.0

        metrics[link_id, 0] = occupied_slots / total_slots
        metrics[link_id, 1] = free_summary.entropy
        metrics[link_id, 2] = (
            1.0 - (free_summary.largest / free_summary.total_free)
            if free_summary.total_free > 0
            else 0.0
        )
        metrics[link_id, 3] = compactness
        metrics[link_id, 4] = free_summary.count / max_block_count
        metrics[link_id, 5] = free_summary.rss
    return metrics


def _build_global_features(analysis: RequestAnalysis) -> np.ndarray:
    if _build_global_features_kernel is not None:
        return _build_global_features_kernel(
            analysis.link_metrics,
            float(analysis.free_slots_ratio),
            float(analysis.active_services_norm),
        )

    link_metrics = analysis.link_metrics
    return np.array(
        [
            float(link_metrics[:, 0].mean()) if link_metrics.size else 0.0,
            float(link_metrics[:, 0].std()) if link_metrics.size else 0.0,
            float(link_metrics[:, 0].max()) if link_metrics.size else 0.0,
            analysis.free_slots_ratio,
            float(link_metrics[:, 1].mean()) if link_metrics.size else 0.0,
            float(link_metrics[:, 2].mean()) if link_metrics.size else 0.0,
            float(link_metrics[:, 3].mean()) if link_metrics.size else 0.0,
            analysis.active_services_norm,
        ],
        dtype=np.float32,
    )


def _build_path_features(analysis: RequestAnalysis) -> np.ndarray:
    total_slots = analysis.config.num_spectrum_resources
    if _build_path_features_kernel is not None:
        max_length = max((path.length_km for path in analysis.paths), default=1.0)
        max_hops = max((path.hops for path in analysis.paths), default=1)
        max_block_count = max(1, math.ceil(total_slots / 2))
        max_path_links = max((len(path.link_ids) for path in analysis.paths), default=0)
        path_link_ids = np.full((analysis.config.k_paths, max_path_links), -1, dtype=np.int32)
        path_link_counts = np.zeros(analysis.config.k_paths, dtype=np.int32)
        path_length_norms = np.zeros(analysis.config.k_paths, dtype=np.float32)
        path_hops_norms = np.zeros(analysis.config.k_paths, dtype=np.float32)

        for path_index, path in enumerate(analysis.paths):
            link_count = len(path.link_ids)
            path_link_counts[path_index] = link_count
            if link_count > 0:
                path_link_ids[path_index, :link_count] = path.link_ids
            path_length_norms[path_index] = float(path.length_km / max_length)
            path_hops_norms[path_index] = float(path.hops / max_hops)

        return _build_path_features_kernel(
            analysis.common_free_masks.view(np.uint8),
            analysis.link_metrics,
            path_link_ids,
            path_link_counts,
            path_length_norms,
            path_hops_norms,
            total_slots,
            float(max_block_count),
        )

    result = np.zeros((analysis.config.k_paths, len(PATH_FEATURE_NAMES)), dtype=np.float32)
    max_length = max((path.length_km for path in analysis.paths), default=1.0)
    max_hops = max((path.hops for path in analysis.paths), default=1)
    max_block_count = max(1, math.ceil(total_slots / 2))

    for path_index, path in enumerate(analysis.paths):
        common_summary = _free_block_summary(analysis.common_free_masks[path_index, :])
        if path.link_ids:
            util_sum = 0.0
            route_cuts_sum = 0.0
            route_rss_sum = 0.0
            entropy_sum = 0.0
            external_frag_sum = 0.0
            compactness_sum = 0.0
            util_max = 0.0
            link_count = len(path.link_ids)
            for link_id in path.link_ids:
                link_metrics = analysis.link_metrics[int(link_id)]
                util = float(link_metrics[0])
                util_sum += util
                route_cuts_sum += float(link_metrics[4])
                route_rss_sum += float(link_metrics[5])
                entropy_sum += float(link_metrics[1])
                external_frag_sum += float(link_metrics[2])
                compactness_sum += float(link_metrics[3])
                if util > util_max:
                    util_max = util
            util_mean = util_sum / link_count
            route_cuts_norm = route_cuts_sum / link_count
            route_rss = route_rss_sum / link_count
            entropy_mean = entropy_sum / link_count
            external_frag_mean = external_frag_sum / link_count
            compactness_mean = compactness_sum / link_count
        else:
            util_mean = 0.0
            util_max = 0.0
            route_cuts_norm = 0.0
            route_rss = 0.0
            entropy_mean = 0.0
            external_frag_mean = 0.0
            compactness_mean = 1.0

        result[path_index, :] = (
            float(path.length_km / max_length),
            float(path.hops / max_hops),
            util_mean,
            util_max,
            float(common_summary.total_free / total_slots),
            float(common_summary.largest / total_slots),
            float(common_summary.count / max_block_count),
            common_summary.entropy,
            route_cuts_norm,
            route_rss,
            entropy_mean,
            external_frag_mean,
            compactness_mean,
        )
    return result


def _build_path_mod_features(analysis: RequestAnalysis) -> np.ndarray:
    if _build_path_mod_features_kernel is not None:
        return _build_path_mod_features_kernel(
            analysis.resource_valid_starts.view(np.uint8),
            analysis.qot_valid_starts.view(np.uint8),
            analysis.osnr_margin_by_start,
            analysis.nli_share_by_start,
            analysis.worst_link_nli_share_by_start,
            analysis.fragmentation_damage_num_blocks_by_start,
            analysis.fragmentation_damage_largest_block_by_start,
            analysis.required_slots_by_path_mod,
        )

    path_count, modulation_count, total_slots = analysis.resource_valid_starts.shape
    result = np.zeros((path_count, modulation_count, len(PATH_MOD_FEATURE_NAMES)), dtype=np.float32)

    for path_index in range(path_count):
        for modulation_offset in range(modulation_count):
            required_slots = int(analysis.required_slots_by_path_mod[path_index, modulation_offset])
            if required_slots <= 0:
                continue

            denominator = max(1, total_slots - required_slots + 1)
            resource_count = 0
            qot_count = 0
            first_resource_slot = -1
            first_qot_slot = -1
            last_qot_slot = -1
            best_margin = 0.0
            best_nli_share = 0.0
            best_worst_link_share = 0.0
            damage_num_blocks = 0.0
            damage_largest_block = 0.0
            best_slot = -1

            for slot_index in range(total_slots):
                if analysis.resource_valid_starts[path_index, modulation_offset, slot_index]:
                    resource_count += 1
                    if first_resource_slot < 0:
                        first_resource_slot = slot_index
                if not analysis.qot_valid_starts[path_index, modulation_offset, slot_index]:
                    continue
                qot_count += 1
                if first_qot_slot < 0:
                    first_qot_slot = slot_index
                last_qot_slot = slot_index
                margin = float(analysis.osnr_margin_by_start[path_index, modulation_offset, slot_index])
                if math.isnan(margin):
                    continue
                if best_slot < 0 or margin > best_margin:
                    best_slot = slot_index
                    best_margin = margin

            if best_slot >= 0:
                best_nli_share = float(analysis.nli_share_by_start[path_index, modulation_offset, best_slot])
                best_worst_link_share = float(
                    analysis.worst_link_nli_share_by_start[path_index, modulation_offset, best_slot]
                )
                damage_num_blocks = float(
                    analysis.fragmentation_damage_num_blocks_by_start[path_index, modulation_offset, best_slot]
                )
                damage_largest_block = float(
                    analysis.fragmentation_damage_largest_block_by_start[path_index, modulation_offset, best_slot]
                )
            elif first_resource_slot >= 0:
                damage_num_blocks = float(
                    analysis.fragmentation_damage_num_blocks_by_start[
                        path_index,
                        modulation_offset,
                        first_resource_slot,
                    ]
                )
                damage_largest_block = float(
                    analysis.fragmentation_damage_largest_block_by_start[
                        path_index,
                        modulation_offset,
                        first_resource_slot,
                    ]
                )

            result[path_index, modulation_offset, :] = (
                float(required_slots / total_slots),
                float(resource_count / denominator),
                float(qot_count / denominator),
                float(first_qot_slot / (total_slots - 1)) if first_qot_slot >= 0 and total_slots > 1 else 0.0,
                float(last_qot_slot / (total_slots - 1)) if last_qot_slot >= 0 and total_slots > 1 else 0.0,
                _normalize_margin(best_margin),
                best_nli_share,
                best_worst_link_share,
                _clamp_unit(damage_num_blocks),
                _clamp_unit(damage_largest_block),
            )
    return result


def _build_path_slot_features(analysis: RequestAnalysis) -> np.ndarray:
    if _build_path_slot_features_kernel is not None:
        return _build_path_slot_features_kernel(
            analysis.common_free_masks.view(np.uint8),
            analysis.resource_valid_starts.view(np.uint8),
            analysis.qot_valid_starts.view(np.uint8),
            analysis.osnr_margin_by_start,
            analysis.nli_share_by_start,
            5,
        )

    path_count, total_slots = analysis.common_free_masks.shape
    result = np.zeros((path_count, total_slots, len(PATH_SLOT_FEATURE_NAMES)), dtype=np.float32)
    modulation_count = analysis.resource_valid_starts.shape[1]

    for path_index in range(path_count):
        common_free_mask = analysis.common_free_masks[path_index, :]
        block_length_norm, left_span_norm, right_span_norm = _slot_block_vectors(common_free_mask)
        local_fragmentation = _local_fragmentation(common_free_mask)
        for slot_index in range(total_slots):
            result[path_index, slot_index, 0] = 1.0 if common_free_mask[slot_index] else 0.0
            result[path_index, slot_index, 1] = block_length_norm[slot_index]
            result[path_index, slot_index, 2] = left_span_norm[slot_index]
            result[path_index, slot_index, 3] = right_span_norm[slot_index]
            result[path_index, slot_index, 4] = local_fragmentation[slot_index]

            has_resource_candidate = False
            has_qot_candidate = False
            best_margin = 0.0
            best_slot_nli = 0.0
            best_margin_found = False

            for modulation_offset in range(modulation_count):
                if analysis.resource_valid_starts[path_index, modulation_offset, slot_index]:
                    has_resource_candidate = True
                if not analysis.qot_valid_starts[path_index, modulation_offset, slot_index]:
                    continue
                has_qot_candidate = True
                margin = float(analysis.osnr_margin_by_start[path_index, modulation_offset, slot_index])
                if math.isnan(margin):
                    continue
                if not best_margin_found or margin > best_margin:
                    best_margin_found = True
                    best_margin = margin
                    best_slot_nli = float(analysis.nli_share_by_start[path_index, modulation_offset, slot_index])

            result[path_index, slot_index, 5] = 1.0 if has_resource_candidate else 0.0
            result[path_index, slot_index, 6] = 1.0 if has_qot_candidate else 0.0
            if best_margin_found:
                result[path_index, slot_index, 7] = _normalize_margin(best_margin)
                result[path_index, slot_index, 8] = best_slot_nli
    return result


def _free_block_summary(free_mask: np.ndarray) -> _BlockSummary:
    return _analyze_free_mask(free_mask).summary


def _block_lengths(free_mask: np.ndarray) -> list[int]:
    analysis = _analyze_free_mask(free_mask)
    return [int(length) for length in analysis.run_lengths]


def _slot_block_vectors(free_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_slots = free_mask.shape[0]
    block_length = np.zeros(total_slots, dtype=np.float32)
    left_span = np.zeros(total_slots, dtype=np.float32)
    right_span = np.zeros(total_slots, dtype=np.float32)
    analysis = _analyze_free_mask(free_mask)

    for run_start, run_end in zip(analysis.run_starts, analysis.run_ends, strict=False):
        _fill_slot_run_vectors(block_length, left_span, right_span, int(run_start), int(run_end), total_slots)
    return block_length, left_span, right_span


def _fill_slot_run_vectors(
    block_length: np.ndarray,
    left_span: np.ndarray,
    right_span: np.ndarray,
    run_start: int,
    run_end: int,
    total_slots: int,
) -> None:
    run_length = run_end - run_start
    block_value = run_length / total_slots
    for slot_index in range(run_start, run_end):
        block_length[slot_index] = block_value
        left_span[slot_index] = (slot_index - run_start) / total_slots
        right_span[slot_index] = (run_end - slot_index - 1) / total_slots


def _local_fragmentation(free_mask: np.ndarray, window: int = 5) -> np.ndarray:
    total_slots = free_mask.shape[0]
    window = max(1, min(window, total_slots))
    left_pad = window // 2
    right_pad = window - left_pad - 1
    result = np.zeros(total_slots, dtype=np.float32)
    for slot_index in range(total_slots):
        local_free = 0
        start = slot_index - left_pad
        end = slot_index + right_pad
        for neighbor_index in range(start, end + 1):
            if 0 <= neighbor_index < total_slots and free_mask[neighbor_index]:
                local_free += 1
        result[slot_index] = 1.0 - (local_free / window)
    return result


def _normalize_margin(osnr_margin_db: float) -> float:
    normalized = osnr_margin_db / 10.0
    if normalized < -1.0:
        return -1.0
    if normalized > 1.0:
        return 1.0
    return float(normalized)


def _analyze_free_mask(free_mask: np.ndarray) -> _FreeRunAnalysis:
    if _analyze_free_mask_kernel is not None:
        (
            count,
            largest,
            total_free,
            entropy,
            rss,
            run_starts,
            run_ends,
            run_lengths,
            slot_to_run_index,
            largest_other_by_run,
            sum_squares,
            sum_length_log_length,
        ) = _analyze_free_mask_kernel(free_mask)
        return _FreeRunAnalysis(
            summary=_BlockSummary(
                count=int(count),
                largest=int(largest),
                total_free=int(total_free),
                entropy=float(entropy),
                rss=float(rss),
            ),
            run_starts=run_starts,
            run_ends=run_ends,
            run_lengths=run_lengths,
            slot_to_run_index=slot_to_run_index,
            largest_other_by_run=largest_other_by_run,
            sum_squares=float(sum_squares),
            sum_length_log_length=float(sum_length_log_length),
        )

    free_values = np.asarray(free_mask, dtype=np.bool_)
    total_slots = free_values.shape[0]
    slot_to_run_index = np.full(total_slots, -1, dtype=np.int32)

    if total_slots == 0:
        empty = np.empty(0, dtype=np.int32)
        return _FreeRunAnalysis(
            summary=_BlockSummary(count=0, largest=0, total_free=0, entropy=0.0, rss=0.0),
            run_starts=empty,
            run_ends=empty,
            run_lengths=empty,
            slot_to_run_index=slot_to_run_index,
            largest_other_by_run=empty,
            sum_squares=0.0,
            sum_length_log_length=0.0,
        )

    run_starts_list: list[int] = []
    run_ends_list: list[int] = []
    run_lengths_list: list[int] = []
    sum_squares = 0.0
    sum_length_log_length = 0.0
    total_free = 0
    largest = 0
    current_run_start = -1

    for slot_index in range(total_slots):
        if free_values[slot_index]:
            if current_run_start < 0:
                current_run_start = slot_index
            continue
        if current_run_start < 0:
            continue
        run_index = len(run_starts_list)
        run_length = slot_index - current_run_start
        run_starts_list.append(current_run_start)
        run_ends_list.append(slot_index)
        run_lengths_list.append(run_length)
        slot_to_run_index[current_run_start:slot_index] = run_index
        total_free += run_length
        if run_length > largest:
            largest = run_length
        sum_squares += float(run_length * run_length)
        sum_length_log_length += _length_log_length(float(run_length))
        current_run_start = -1

    if current_run_start >= 0:
        run_index = len(run_starts_list)
        run_length = total_slots - current_run_start
        run_starts_list.append(current_run_start)
        run_ends_list.append(total_slots)
        run_lengths_list.append(run_length)
        slot_to_run_index[current_run_start:total_slots] = run_index
        total_free += run_length
        if run_length > largest:
            largest = run_length
        sum_squares += float(run_length * run_length)
        sum_length_log_length += _length_log_length(float(run_length))

    if not run_lengths_list:
        empty = np.empty(0, dtype=np.int32)
        return _FreeRunAnalysis(
            summary=_BlockSummary(count=0, largest=0, total_free=0, entropy=0.0, rss=0.0),
            run_starts=empty,
            run_ends=empty,
            run_lengths=empty,
            slot_to_run_index=slot_to_run_index,
            largest_other_by_run=empty,
            sum_squares=0.0,
            sum_length_log_length=0.0,
        )

    run_starts = np.asarray(run_starts_list, dtype=np.int32)
    run_ends = np.asarray(run_ends_list, dtype=np.int32)
    run_lengths = np.asarray(run_lengths_list, dtype=np.int32)
    if len(run_lengths_list) <= 1:
        entropy = 0.0
    else:
        entropy = (
            math.log(total_free) - (sum_length_log_length / total_free)
        ) / math.log(len(run_lengths_list))
    rss = math.sqrt(sum_squares) / total_free if total_free > 0 else 0.0

    return _FreeRunAnalysis(
        summary=_BlockSummary(
            count=len(run_lengths_list),
            largest=largest,
            total_free=total_free,
            entropy=_clamp_unit(entropy),
            rss=_clamp_unit(rss),
        ),
        run_starts=run_starts,
        run_ends=run_ends,
        run_lengths=run_lengths,
        slot_to_run_index=slot_to_run_index,
        largest_other_by_run=_largest_other_by_run(run_lengths),
        sum_squares=sum_squares,
        sum_length_log_length=sum_length_log_length,
    )


def _largest_other_by_run(run_lengths: np.ndarray) -> np.ndarray:
    if run_lengths.size == 0:
        return np.empty(0, dtype=np.int32)

    prefix = np.zeros(run_lengths.shape[0], dtype=np.int32)
    suffix = np.zeros(run_lengths.shape[0], dtype=np.int32)
    current_max = 0
    for run_index in range(run_lengths.shape[0]):
        value = int(run_lengths[run_index])
        if value > current_max:
            current_max = value
        prefix[run_index] = current_max
    current_max = 0
    for run_index in range(run_lengths.shape[0] - 1, -1, -1):
        value = int(run_lengths[run_index])
        if value > current_max:
            current_max = value
        suffix[run_index] = current_max

    result = np.zeros(run_lengths.shape[0], dtype=np.int32)
    for run_index in range(run_lengths.shape[0]):
        left_max = int(prefix[run_index - 1]) if run_index > 0 else 0
        right_max = int(suffix[run_index + 1]) if run_index + 1 < run_lengths.shape[0] else 0
        result[run_index] = max(left_max, right_max)
    return result


def _summary_after_allocation(
    free_runs: _FreeRunAnalysis,
    service_slot_start: int,
    service_num_slots: int,
    total_slots: int,
) -> _BlockSummary:
    if _summary_after_allocation_kernel is not None:
        post_count, post_largest, post_total_free, post_entropy, post_rss = _summary_after_allocation_kernel(
            free_runs.slot_to_run_index,
            free_runs.run_starts,
            free_runs.run_ends,
            free_runs.run_lengths,
            free_runs.largest_other_by_run,
            free_runs.sum_squares,
            free_runs.sum_length_log_length,
            free_runs.summary.count,
            free_runs.summary.largest,
            free_runs.summary.total_free,
            service_slot_start,
            service_num_slots,
            total_slots,
        )
        return _BlockSummary(
            count=int(post_count),
            largest=int(post_largest),
            total_free=int(post_total_free),
            entropy=float(post_entropy),
            rss=float(post_rss),
        )

    run_index = int(free_runs.slot_to_run_index[service_slot_start])
    if run_index < 0:
        return free_runs.summary

    run_start = int(free_runs.run_starts[run_index])
    run_end = int(free_runs.run_ends[run_index])
    removed_end = service_slot_start + service_num_slots
    if removed_end < total_slots:
        removed_end += 1

    left_length = max(service_slot_start - run_start, 0)
    right_length = max(run_end - removed_end, 0)
    removed_length = free_runs.run_lengths[run_index] - left_length - right_length
    post_total_free = max(free_runs.summary.total_free - int(removed_length), 0)
    post_count = free_runs.summary.count - 1 + int(left_length > 0) + int(right_length > 0)
    post_largest = max(int(free_runs.largest_other_by_run[run_index]), left_length, right_length)

    post_sum_squares = (
        free_runs.sum_squares
        - float(free_runs.run_lengths[run_index] ** 2)
        + float(left_length ** 2)
        + float(right_length ** 2)
    )
    post_rss = math.sqrt(post_sum_squares) / post_total_free if post_total_free > 0 else 0.0

    post_sum_length_log_length = (
        free_runs.sum_length_log_length
        - _length_log_length(float(free_runs.run_lengths[run_index]))
        + _length_log_length(float(left_length))
        + _length_log_length(float(right_length))
    )
    if post_total_free <= 0 or post_count <= 1:
        post_entropy = 0.0
    else:
        post_entropy = (
            math.log(post_total_free) - (post_sum_length_log_length / post_total_free)
        ) / math.log(post_count)

    return _BlockSummary(
        count=post_count,
        largest=post_largest,
        total_free=post_total_free,
        entropy=_clamp_unit(post_entropy),
        rss=_clamp_unit(post_rss),
    )


def _length_log_length(length: float) -> float:
    if length <= 0.0:
        return 0.0
    return length * math.log(length)


def _clamp_unit(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


__all__ = ["RequestAnalysis", "RequestAnalysisEngine"]
