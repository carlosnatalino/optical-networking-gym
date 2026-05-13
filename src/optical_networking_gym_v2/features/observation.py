from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optical_networking_gym_v2.contracts import ObservationSchema, ObservationSnapshot, ServiceRequest
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.optical.qot_engine import QoTEngine
from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysisEngine
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState

if TYPE_CHECKING:
    from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysis


class Observation:
    def __init__(
        self,
        config: ScenarioConfig,
        topology: TopologyModel,
        analysis_engine: RequestAnalysisEngine | None = None,
        *,
        qot_engine: QoTEngine | None = None,
    ) -> None:
        if not config.modulations:
            raise ValueError("Observation requires ScenarioConfig.modulations")
        if config.modulations_to_consider <= 0:
            raise ValueError("Observation requires modulations_to_consider > 0")
        self.config = config
        self.topology = topology
        if analysis_engine is None:
            if qot_engine is None:
                raise ValueError("Observation requires either analysis_engine or qot_engine")
            analysis_engine = RequestAnalysisEngine(config, topology, qot_engine)
        self.analysis_engine = analysis_engine
        self.schema = ObservationSchema(
            request_feature_names=analysis_engine.request_feature_names,
            global_feature_names=analysis_engine.global_feature_names,
            path_feature_names=analysis_engine.path_feature_names,
            path_mod_feature_names=analysis_engine.path_mod_feature_names,
            path_slot_feature_names=analysis_engine.path_slot_feature_names,
            k_paths=config.k_paths,
            modulation_count=config.modulations_to_consider,
            num_spectrum_resources=config.num_spectrum_resources,
        )
        self.empty_observation = np.empty(0, dtype=np.float32)

    def _flatten_analysis(
        self,
        analysis: "RequestAnalysis",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        request_features = analysis.request_features
        global_features = analysis.global_features
        path_features = analysis.path_features
        path_mod_features = analysis.path_mod_features
        path_slot_features = analysis.path_slot_features

        flat = np.empty(self.schema.total_size, dtype=np.float32)
        offset = 0

        next_offset = offset + request_features.size
        flat[offset:next_offset] = request_features
        offset = next_offset

        next_offset = offset + global_features.size
        flat[offset:next_offset] = global_features
        offset = next_offset

        path_flat = path_features.reshape(-1)
        next_offset = offset + path_flat.size
        flat[offset:next_offset] = path_flat
        offset = next_offset

        path_mod_flat = path_mod_features.reshape(-1)
        next_offset = offset + path_mod_flat.size
        flat[offset:next_offset] = path_mod_flat
        offset = next_offset

        path_slot_flat = path_slot_features.reshape(-1)
        flat[offset : offset + path_slot_flat.size] = path_slot_flat

        return (
            request_features,
            global_features,
            path_features,
            path_mod_features,
            path_slot_features,
            flat,
        )

    def build_with_analysis(self, state: RuntimeState, request: ServiceRequest) -> tuple[np.ndarray, "RequestAnalysis"]:
        analysis = self.analysis_engine.build(state, request)
        if not self.config.enable_observation:
            return self.empty_observation, analysis
        flat = self.build_from_analysis(analysis)
        return flat, analysis

    def build_from_analysis(self, analysis: "RequestAnalysis") -> np.ndarray:
        if not self.config.enable_observation:
            return self.empty_observation
        _, _, _, _, _, flat = self._flatten_analysis(analysis)
        return flat

    def build_snapshot(self, state: RuntimeState, request: ServiceRequest) -> ObservationSnapshot:
        analysis = self.analysis_engine.build(state, request, include_inspection=True)
        (
            request_features,
            global_features,
            path_features,
            path_mod_features,
            path_slot_features,
            flat,
        ) = self._flatten_analysis(analysis)
        return ObservationSnapshot(
            schema=self.schema,
            analysis=analysis,
            request=request_features,
            global_features=global_features,
            path=path_features,
            path_mod=path_mod_features,
            path_slot=path_slot_features,
            flat=flat,
        )

    def build(self, state: RuntimeState, request: ServiceRequest) -> np.ndarray:
        flat, _ = self.build_with_analysis(state, request)
        return flat


__all__ = ["Observation"]
