from __future__ import annotations

import numpy as np

from optical_networking_gym_v2.contracts import ActionSelection, ServiceRequest
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.optical.qot_engine import QoTEngine
from optical_networking_gym_v2.runtime.action_codec import (
    reject_action,
    resolve_action_selection,
    total_actions,
)
from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysisEngine
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState

class ActionMask:
    def __init__(
        self,
        config: ScenarioConfig,
        topology: TopologyModel,
        qot_engine: QoTEngine,
        *,
        analysis_engine: RequestAnalysisEngine | None = None,
    ) -> None:
        if not config.modulations:
            raise ValueError("ActionMask requires ScenarioConfig.modulations")
        if config.modulations_to_consider <= 0:
            raise ValueError("ActionMask requires modulations_to_consider > 0")
        self.config = config
        self.topology = topology
        self.qot_engine = qot_engine
        self.analysis_engine = analysis_engine or RequestAnalysisEngine(config, topology, qot_engine)

    @property
    def total_actions(self) -> int:
        return total_actions(self.config)

    def build(self, state: RuntimeState, request: ServiceRequest) -> np.ndarray | None:
        analysis = self.analysis_engine.build(state, request)
        if analysis.action_mask is None:
            return None
        mask = np.zeros(self.total_actions, dtype=np.uint8)
        mask[:-1] = analysis.action_mask
        mask[-1] = 1
        mask.flags.writeable = False
        return mask

    def decode_action(self, action: int, state: RuntimeState, request: ServiceRequest) -> ActionSelection:
        if action < 0 or action >= self.total_actions:
            raise ValueError("action is outside the action space")
        if action == reject_action(self.config):
            raise ValueError("reject action does not decode to a provisioning choice")

        analysis = self.analysis_engine.build(state, request)
        selection = resolve_action_selection(
            self.config,
            modulation_indices=analysis.modulation_indices,
            action=action,
        )
        if selection is None:
            raise ValueError("action does not decode to a valid modulation subset entry")
        return selection


__all__ = ["ActionMask"]
