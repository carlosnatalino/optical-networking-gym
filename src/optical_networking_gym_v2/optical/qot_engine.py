from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from optical_networking_gym_v2.contracts import Modulation, QoTRequest, QoTResult, ServiceQoTUpdate
from .kernels.qot_kernel import accumulate_link_noise, summarize_candidate_starts
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState
from optical_networking_gym_v2.network.topology import PathRecord, TopologyModel
from optical_networking_gym_v2.config.scenario import ScenarioConfig


_PHI_MODULATION_BY_SE = {
    1: 1.0,
    2: 1.0,
    3: 2.0 / 3.0,
    4: 17.0 / 25.0,
    5: 69.0 / 100.0,
    6: 13.0 / 21.0,
}


@dataclass(slots=True)
class _LinkInterferenceCache:
    version: int
    service_ids: np.ndarray
    center_frequencies: np.ndarray
    bandwidths: np.ndarray
    phi_modulation: np.ndarray


@dataclass(slots=True)
class _PathSummaryStaticInputs:
    link_ids: tuple[int, ...]
    span_offsets: np.ndarray
    span_lengths: np.ndarray
    span_attenuation: np.ndarray
    span_noise_figure: np.ndarray


@dataclass(slots=True)
class _PreparedCandidateSummaryInputs:
    span_offsets: np.ndarray
    span_lengths: np.ndarray
    span_attenuation: np.ndarray
    span_noise_figure: np.ndarray
    running_offsets: np.ndarray
    running_service_ids: np.ndarray
    running_center_frequencies: np.ndarray
    running_bandwidths: np.ndarray
    running_phi_modulation: np.ndarray


@dataclass(frozen=True, slots=True)
class QoTCandidateSummary:
    osnr: float
    ase: float
    nli: float
    meets_threshold: bool
    osnr_margin: float
    nli_share: float
    worst_link_nli_share: float


@dataclass(frozen=True, slots=True)
class _MetricsSummary:
    osnr: float
    ase: float
    nli: float
    total_nli_share: float
    worst_link_nli_share: float


@dataclass(frozen=True, slots=True)
class _CandidateBatchSummary:
    meets_threshold: np.ndarray
    osnr_margin: np.ndarray
    nli_share: np.ndarray
    worst_link_nli_share: np.ndarray


class QoTEngine:
    def __init__(self, config: ScenarioConfig, topology: TopologyModel) -> None:
        self.config = config
        self.topology = topology
        self._include_nli = config.qot_constraint == "ASE+NLI"
        self._include_running_service_interference = bool(config.measure_disruptions)
        self._launch_power = 10 ** ((config.launch_power_dbm - 30.0) / 10.0)
        self._empty_service_ids = np.empty(0, dtype=np.int32)
        self._empty_float_values = np.empty(0, dtype=np.float64)
        self._link_span_lengths_km = tuple(
            np.array([span.length_km for span in link.spans], dtype=np.float64)
            for link in topology.links
        )
        self._link_span_attenuation_normalized = tuple(
            np.array([span.attenuation_normalized for span in link.spans], dtype=np.float64)
            for link in topology.links
        )
        self._link_span_noise_figure_normalized = tuple(
            np.array([span.noise_figure_normalized for span in link.spans], dtype=np.float64)
            for link in topology.links
        )
        self._link_interference_cache: dict[int, _LinkInterferenceCache] = {}
        self._path_summary_static_cache: dict[int, _PathSummaryStaticInputs] = {}

    def build_candidate(
        self,
        request: object,
        path: PathRecord,
        modulation: Modulation,
        service_slot_start: int,
        service_num_slots: int,
    ) -> QoTRequest:
        bandwidth = self.config.frequency_slot_bandwidth * service_num_slots
        center_frequency = (
            self.config.frequency_start
            + self.config.frequency_slot_bandwidth * service_slot_start
            + self.config.frequency_slot_bandwidth * (service_num_slots / 2.0)
        )
        return QoTRequest(
            request=request,
            path=path,
            modulation=modulation,
            service_slot_start=service_slot_start,
            service_num_slots=service_num_slots,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
            launch_power=self._launch_power,
        )

    def evaluate_candidate(self, state: RuntimeState, candidate: QoTRequest) -> QoTResult:
        metrics = self._calculate_metrics(
            path=candidate.path,
            service_id=candidate.service_id,
            center_frequency=candidate.center_frequency,
            bandwidth=candidate.bandwidth,
            launch_power=candidate.launch_power,
            state=state,
        )
        return QoTResult(
            osnr=metrics.osnr,
            ase=metrics.ase,
            nli=metrics.nli,
            meets_threshold=self._meets_threshold(candidate.path, candidate.modulation, metrics.osnr),
        )

    def summarize_candidate(self, state: RuntimeState, candidate: QoTRequest) -> QoTCandidateSummary:
        metrics = self._calculate_metrics(
            path=candidate.path,
            service_id=candidate.service_id,
            center_frequency=candidate.center_frequency,
            bandwidth=candidate.bandwidth,
            launch_power=candidate.launch_power,
            state=state,
        )
        threshold = candidate.modulation.minimum_osnr + self.config.margin
        return QoTCandidateSummary(
            osnr=metrics.osnr,
            ase=metrics.ase,
            nli=metrics.nli,
            meets_threshold=metrics.osnr >= threshold,
            osnr_margin=metrics.osnr - threshold,
            nli_share=metrics.total_nli_share,
            worst_link_nli_share=metrics.worst_link_nli_share,
        )

    def summarize_candidate_at(
        self,
        *,
        state: RuntimeState,
        service_id: int,
        path: PathRecord,
        modulation: Modulation,
        service_slot_start: int,
        service_num_slots: int,
    ) -> QoTCandidateSummary:
        bandwidth = self.config.frequency_slot_bandwidth * service_num_slots
        center_frequency = (
            self.config.frequency_start
            + self.config.frequency_slot_bandwidth * service_slot_start
            + self.config.frequency_slot_bandwidth * (service_num_slots / 2.0)
        )
        metrics = self._calculate_metrics(
            path=path,
            service_id=service_id,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
            launch_power=self._launch_power,
            state=state,
        )
        threshold = modulation.minimum_osnr + self.config.margin
        return QoTCandidateSummary(
            osnr=metrics.osnr,
            ase=metrics.ase,
            nli=metrics.nli,
            meets_threshold=metrics.osnr >= threshold,
            osnr_margin=metrics.osnr - threshold,
            nli_share=metrics.total_nli_share,
            worst_link_nli_share=metrics.worst_link_nli_share,
        )

    def summarize_candidate_starts(
        self,
        *,
        state: RuntimeState,
        service_id: int,
        path: PathRecord,
        modulation: Modulation,
        service_num_slots: int,
        candidate_starts: np.ndarray | list[int] | tuple[int, ...],
    ) -> _CandidateBatchSummary:
        prepared_inputs = self._prepare_candidate_summary_inputs(state, path)
        return self._summarize_candidate_starts_prepared(
            prepared_inputs=prepared_inputs,
            service_id=service_id,
            service_num_slots=service_num_slots,
            candidate_starts=candidate_starts,
            threshold=modulation.minimum_osnr + self.config.margin,
        )

    def _prepare_candidate_summary_inputs(
        self,
        state: RuntimeState,
        path: PathRecord,
    ) -> _PreparedCandidateSummaryInputs:
        static_inputs = self._path_summary_static_inputs(path)
        if not self._include_running_service_interference:
            return _PreparedCandidateSummaryInputs(
                span_offsets=static_inputs.span_offsets,
                span_lengths=static_inputs.span_lengths,
                span_attenuation=static_inputs.span_attenuation,
                span_noise_figure=static_inputs.span_noise_figure,
                running_offsets=np.zeros(len(static_inputs.link_ids) + 1, dtype=np.int32),
                running_service_ids=self._empty_service_ids,
                running_center_frequencies=self._empty_float_values,
                running_bandwidths=self._empty_float_values,
                running_phi_modulation=self._empty_float_values,
            )
        running_descriptors = tuple(
            self._link_running_service_arrays(state, link_id) for link_id in static_inputs.link_ids
        )
        running_offsets = np.zeros(len(running_descriptors) + 1, dtype=np.int32)
        total_running = 0
        for link_index, descriptor in enumerate(running_descriptors, start=1):
            total_running += int(descriptor.service_ids.shape[0])
            running_offsets[link_index] = total_running

        running_service_ids = np.empty(total_running, dtype=np.int32)
        running_center_frequencies = np.empty(total_running, dtype=np.float64)
        running_bandwidths = np.empty(total_running, dtype=np.float64)
        running_phi_modulation = np.empty(total_running, dtype=np.float64)

        cursor = 0
        for descriptor in running_descriptors:
            count = int(descriptor.service_ids.shape[0])
            if count == 0:
                continue
            next_cursor = cursor + count
            running_service_ids[cursor:next_cursor] = descriptor.service_ids
            running_center_frequencies[cursor:next_cursor] = descriptor.center_frequencies
            running_bandwidths[cursor:next_cursor] = descriptor.bandwidths
            running_phi_modulation[cursor:next_cursor] = descriptor.phi_modulation
            cursor = next_cursor

        return _PreparedCandidateSummaryInputs(
            span_offsets=static_inputs.span_offsets,
            span_lengths=static_inputs.span_lengths,
            span_attenuation=static_inputs.span_attenuation,
            span_noise_figure=static_inputs.span_noise_figure,
            running_offsets=running_offsets,
            running_service_ids=running_service_ids,
            running_center_frequencies=running_center_frequencies,
            running_bandwidths=running_bandwidths,
            running_phi_modulation=running_phi_modulation,
        )

    def _summarize_candidate_starts_prepared(
        self,
        *,
        prepared_inputs: _PreparedCandidateSummaryInputs,
        service_id: int,
        service_num_slots: int,
        candidate_starts: np.ndarray | list[int] | tuple[int, ...],
        threshold: float,
    ) -> _CandidateBatchSummary:
        starts = np.asarray(candidate_starts, dtype=np.int32)
        if starts.ndim != 1:
            raise ValueError("candidate_starts must be 1D")
        if starts.size == 0:
            empty_bool = np.zeros(0, dtype=np.bool_)
            empty_float = np.zeros(0, dtype=np.float32)
            return _CandidateBatchSummary(
                meets_threshold=empty_bool,
                osnr_margin=empty_float,
                nli_share=empty_float.copy(),
                worst_link_nli_share=empty_float.copy(),
            )
        meets_threshold, osnr_margin, nli_share, worst_link_nli_share = summarize_candidate_starts(
            prepared_inputs.span_offsets,
            prepared_inputs.span_lengths,
            prepared_inputs.span_attenuation,
            prepared_inputs.span_noise_figure,
            prepared_inputs.running_offsets,
            prepared_inputs.running_service_ids,
            prepared_inputs.running_center_frequencies,
            prepared_inputs.running_bandwidths,
            prepared_inputs.running_phi_modulation,
            starts,
            current_service_id=service_id,
            frequency_start=self.config.frequency_start,
            frequency_slot_bandwidth=self.config.frequency_slot_bandwidth,
            service_num_slots=service_num_slots,
            launch_power=self._launch_power,
            threshold=threshold,
            include_nli=self._include_nli,
        )
        return _CandidateBatchSummary(
            meets_threshold=meets_threshold,
            osnr_margin=osnr_margin,
            nli_share=nli_share,
            worst_link_nli_share=worst_link_nli_share,
        )

    def _path_summary_static_inputs(self, path: PathRecord) -> _PathSummaryStaticInputs:
        cached = self._path_summary_static_cache.get(path.id)
        if cached is not None:
            return cached

        span_offsets = np.zeros(len(path.link_ids) + 1, dtype=np.int32)
        total_spans = 0
        for link_index, link_id in enumerate(path.link_ids, start=1):
            total_spans += int(self._link_span_lengths_km[link_id].shape[0])
            span_offsets[link_index] = total_spans

        span_lengths = np.empty(total_spans, dtype=np.float64)
        span_attenuation = np.empty(total_spans, dtype=np.float64)
        span_noise_figure = np.empty(total_spans, dtype=np.float64)

        cursor = 0
        for link_id in path.link_ids:
            link_span_lengths = self._link_span_lengths_km[link_id]
            count = int(link_span_lengths.shape[0])
            next_cursor = cursor + count
            span_lengths[cursor:next_cursor] = link_span_lengths
            span_attenuation[cursor:next_cursor] = self._link_span_attenuation_normalized[link_id]
            span_noise_figure[cursor:next_cursor] = self._link_span_noise_figure_normalized[link_id]
            cursor = next_cursor

        static_inputs = _PathSummaryStaticInputs(
            link_ids=path.link_ids,
            span_offsets=span_offsets,
            span_lengths=span_lengths,
            span_attenuation=span_attenuation,
            span_noise_figure=span_noise_figure,
        )
        self._path_summary_static_cache[path.id] = static_inputs
        return static_inputs

    def recompute_service(self, state: RuntimeState, service_id: int) -> ServiceQoTUpdate:
        service = state.active_services_by_id[service_id]
        if service.modulation is None:
            raise ValueError(f"service_id {service_id} does not have modulation data for QoT")
        metrics = self._calculate_metrics(
            path=service.path,
            service_id=service.service_id,
            center_frequency=service.center_frequency,
            bandwidth=service.bandwidth,
            launch_power=service.launch_power,
            state=state,
        )
        return ServiceQoTUpdate(
            service_id=service_id,
            osnr=metrics.osnr,
            ase=metrics.ase,
            nli=metrics.nli,
        )

    def refresh_services(
        self,
        state: RuntimeState,
        service_ids: tuple[int, ...] | list[int],
    ) -> tuple[ServiceQoTUpdate, ...]:
        return tuple(self.recompute_service(state, service_id) for service_id in service_ids)

    def impacted_service_ids(
        self,
        state: RuntimeState,
        path: PathRecord,
        *,
        exclude_service_id: int | None = None,
    ) -> tuple[int, ...]:
        impacted: set[int] = set()
        for link_id in path.link_ids:
            impacted.update(state.link_active_service_ids[link_id])
        if exclude_service_id is not None:
            impacted.discard(exclude_service_id)
        return tuple(sorted(impacted))

    def _meets_threshold(self, path: PathRecord, modulation: Modulation, osnr: float) -> bool:
        if self.config.qot_constraint == "DIST":
            return path.length_km <= modulation.maximum_length
        return osnr >= modulation.minimum_osnr + self.config.margin

    def _calculate_metrics(
        self,
        *,
        path: PathRecord,
        service_id: int,
        center_frequency: float,
        bandwidth: float,
        launch_power: float,
        state: RuntimeState,
    ) -> _MetricsSummary:
        acc_gsnr = 0.0
        acc_ase = 0.0
        acc_nli = 0.0
        worst_link_nli_share = 0.0

        for link_id in path.link_ids:
            if self._include_running_service_interference:
                running = self._link_running_service_arrays(state, link_id)
                running_service_ids = running.service_ids
                running_center_frequencies = running.center_frequencies
                running_bandwidths = running.bandwidths
                running_phi_modulation = running.phi_modulation
            else:
                running_service_ids = self._empty_service_ids
                running_center_frequencies = self._empty_float_values
                running_bandwidths = self._empty_float_values
                running_phi_modulation = self._empty_float_values
            link_acc_gsnr, link_acc_ase, link_acc_nli = accumulate_link_noise(
                self._link_span_lengths_km[link_id],
                self._link_span_attenuation_normalized[link_id],
                self._link_span_noise_figure_normalized[link_id],
                running_service_ids,
                running_center_frequencies,
                running_bandwidths,
                running_phi_modulation,
                current_service_id=service_id,
                center_frequency=center_frequency,
                bandwidth=bandwidth,
                launch_power=launch_power,
                include_nli=self._include_nli,
            )
            acc_gsnr += link_acc_gsnr
            acc_ase += link_acc_ase
            acc_nli += link_acc_nli
            if link_acc_nli > 0.0 or link_acc_ase > 0.0:
                link_nli_share = link_acc_nli / (link_acc_ase + link_acc_nli)
                if link_nli_share > worst_link_nli_share:
                    worst_link_nli_share = link_nli_share

        osnr = 10.0 * math.log10(1.0 / acc_gsnr)
        ase = 10.0 * math.log10(1.0 / acc_ase)
        nli = 10.0 * math.log10(1.0 / acc_nli) if acc_nli > 0.0 else 0.0
        total_nli_share = acc_nli / (acc_ase + acc_nli) if (acc_ase > 0.0 or acc_nli > 0.0) else 0.0
        return _MetricsSummary(
            osnr=osnr,
            ase=ase,
            nli=nli,
            total_nli_share=total_nli_share,
            worst_link_nli_share=worst_link_nli_share,
        )

    def _link_running_service_arrays(
        self,
        state: RuntimeState,
        link_id: int,
    ) -> _LinkInterferenceCache:
        version = int(state.link_versions[link_id])
        cached = self._link_interference_cache.get(link_id)
        if cached is not None and cached.version == version:
            return cached

        service_ids = tuple(sorted(state.link_active_service_ids[link_id]))
        if not service_ids:
            cache = _LinkInterferenceCache(
                version=version,
                service_ids=np.empty(0, dtype=np.int32),
                center_frequencies=np.empty(0, dtype=np.float64),
                bandwidths=np.empty(0, dtype=np.float64),
                phi_modulation=np.empty(0, dtype=np.float64),
            )
            self._link_interference_cache[link_id] = cache
            return cache

        center_frequencies = np.empty(len(service_ids), dtype=np.float64)
        bandwidths = np.empty(len(service_ids), dtype=np.float64)
        phi_modulation = np.empty(len(service_ids), dtype=np.float64)
        numeric_service_ids = np.empty(len(service_ids), dtype=np.int32)

        for index, running_service_id in enumerate(service_ids):
            running_service = state.active_services_by_id[running_service_id]
            if running_service.modulation is None:
                raise ValueError(
                    f"active service {running_service_id} is missing modulation for QoT evaluation"
                )
            numeric_service_ids[index] = running_service_id
            center_frequencies[index] = running_service.center_frequency
            bandwidths[index] = running_service.bandwidth
            phi_modulation[index] = _PHI_MODULATION_BY_SE[running_service.modulation.spectral_efficiency]

        cache = _LinkInterferenceCache(
            version=version,
            service_ids=numeric_service_ids,
            center_frequencies=center_frequencies,
            bandwidths=bandwidths,
            phi_modulation=phi_modulation,
        )
        self._link_interference_cache[link_id] = cache
        return cache
