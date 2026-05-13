from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

# from optical_networking_gym.trace_utils import normalize_trace_payload  # legacy v1
from optical_networking_gym_v2.contracts import (
    Allocation,
    ServiceRequest,
    Status,
    StepTransition,
)
from optical_networking_gym_v2.features.action_mask import ActionMask
from optical_networking_gym_v2.features.observation import Observation
from optical_networking_gym_v2.instrumentation.traces import write_step_trace_jsonl
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.optical.qot_engine import QoTEngine
from optical_networking_gym_v2.rl.reward_function import RewardFunction
from optical_networking_gym_v2.runtime.action_codec import (
    decode_action,
    reject_action,
    total_actions,
)
from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysis, RequestAnalysisEngine
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState
from optical_networking_gym_v2.runtime.step_info import StepInfo
from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.stats.statistics import Statistics
from .traffic_model import TrafficModel


class Simulator:
    def __init__(
        self,
        config: ScenarioConfig,
        topology: TopologyModel,
        *,
        episode_length: int,
        capture_traffic_table: bool = False,
        capture_step_trace: bool = False,
    ) -> None:
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        normalized_config = replace(
            config,
            episode_length=episode_length,
            capture_traffic_table=capture_traffic_table,
            capture_step_trace=capture_step_trace,
        )
        self.base_config = normalized_config
        self.config = normalized_config
        self.topology = topology
        self.episode_length = episode_length
        self.capture_traffic_table = capture_traffic_table
        self.capture_step_trace = capture_step_trace
        self._helper_structure_key: tuple[object, ...] | None = None
        self._empty_observation = np.empty(0, dtype=np.float32)
        self._build_runtime_helpers(normalized_config)

        self.traffic_model: TrafficModel | None = None
        self.state: RuntimeState | None = None
        self.statistics: Statistics | None = None
        self.current_request: ServiceRequest | None = None
        self.current_analysis: RequestAnalysis | None = None
        self.current_observation: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.steps_completed = 0
        self._disrupted_service_ids: set[int] = set()
        self._captured_trace_steps: list[dict[str, object]] = []

    @property
    def total_actions(self) -> int:
        return total_actions(self.config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if options is not None and options.get("only_episode_counters"):
            if self.statistics is None or self.current_observation is None:
                raise RuntimeError("cannot reset only episode counters before a full reset")
            self.statistics.reset_episode()
            self.steps_completed = 0
            return self._copy_observation(self.current_observation), self._build_reset_info()

        next_config = replace(self.base_config, seed=seed) if seed is not None else self.base_config
        self._apply_runtime_config(next_config)

        self.traffic_model = TrafficModel(
            self.config,
            self.topology,
            capture_table=self.capture_traffic_table,
        )
        self.state = RuntimeState(self.config, self.topology)
        self.statistics = Statistics(self.config)
        self.current_request = None
        self.current_analysis = None
        self.current_observation = None
        self.current_mask = None
        self.steps_completed = 0
        self._disrupted_service_ids = set()
        self._captured_trace_steps = []

        self._prepare_next_request()
        if self.current_observation is None:
            raise RuntimeError("failed to prepare the first request")
        return self._copy_observation(self.current_observation), self._build_reset_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.state is None or self.statistics is None:
            raise RuntimeError("reset() must be called before step()")
        if self.current_request is None or self.current_analysis is None:
            raise RuntimeError("there is no active request to process")

        current_request = self.current_request
        current_mask = self.current_mask
        analysis = self.current_analysis
        pre_action_state = self._trace_state_snapshot() if self.capture_step_trace else None
        action_payload = None
        candidate_payload = None
        trace_osnr_requirement = 0.0
        trace_candidate_qot = None
        if self.capture_step_trace:
            action_payload, candidate_payload, trace_osnr_requirement, trace_candidate_qot = self._trace_action_details(
                action,
                analysis,
                current_request,
            )
        transition = self._apply_action(action, analysis, current_mask)
        post_action_state = self._trace_state_snapshot() if self.capture_step_trace else None
        self.statistics.record_transition(transition)
        if transition.dropped_qot:
            self.statistics.record_dropped_qot(transition.dropped_qot)

        reward_value, reward_breakdown = self.reward_function.evaluate_transition(
            transition,
            request_analysis=analysis,
            has_valid_non_reject_action=analysis.has_valid_non_reject_action,
        )

        self.steps_completed += 1
        terminated = self.steps_completed >= self.episode_length
        truncated = False
        traffic_exhausted = False

        next_mask: np.ndarray | None = None
        if not terminated:
            try:
                self._prepare_next_request()
            except StopIteration:
                terminated = True
                traffic_exhausted = True
                self.current_request = None
                self.current_analysis = None
                self.current_observation = None
                self.current_mask = None
            else:
                next_mask = self.current_mask if self.current_mask is not None else None

        if self.capture_step_trace:
            post_step_state = self._trace_state_snapshot()
            trace_mask = current_mask if current_mask is not None else self.get_trace_action_mask()
            self._captured_trace_steps.append(
                {
                    "record_type": "trace_step",
                    "step_index": len(self._captured_trace_steps),
                    "request": self._trace_request_payload(current_request),
                    "action_mask": [int(value) for value in trace_mask.tolist()],
                    "action": action_payload,
                    "candidate": candidate_payload,
                    "outcome": {
                        "status": transition.allocation.status.value,
                        "accepted": bool(transition.accepted),
                        "blocked_due_to_resources": transition.allocation.status.value == "blocked_resources",
                        "blocked_due_to_qot": transition.allocation.status.value == "blocked_qot",
                        "osnr": self._trace_float(
                            transition.osnr if transition.accepted else (trace_candidate_qot or {}).get("osnr", 0.0)
                        ),
                        "osnr_requirement": self._trace_float(
                            trace_osnr_requirement if trace_osnr_requirement else transition.osnr_requirement
                        ),
                        "ase": self._trace_float(
                            self.state.service_qot_by_id.get(current_request.service_id, (0.0, 0.0, 0.0))[1]
                            if transition.accepted and self.state is not None
                            else (trace_candidate_qot or {}).get("ase", 0.0)
                        ),
                        "nli": self._trace_float(
                            self.state.service_qot_by_id.get(current_request.service_id, (0.0, 0.0, 0.0))[2]
                            if transition.accepted and self.state is not None
                            else (trace_candidate_qot or {}).get("nli", 0.0)
                        ),
                    },
                    "pre_action_state": pre_action_state,
                    "post_action_state": post_action_state,
                    "post_step_state": post_step_state,
                    "next_request": self._trace_request_payload(self.current_request),
                }
            )

        info = self.step_info_builder.build(
            self.statistics,
            transition,
            terminated=terminated,
            truncated=truncated,
            reward=reward_value,
            reward_breakdown=reward_breakdown,
            extra={
                "mask": self._info_mask(next_mask),
                "traffic_exhausted": traffic_exhausted,
            },
        )

        if terminated:
            observation = self._terminal_observation()
        else:
            observation = self._copy_observation(self.current_observation)

        return observation, reward_value, terminated, truncated, info

    def action_masks(self) -> np.ndarray | None:
        if not self.config.enable_action_mask or self.current_mask is None:
            return None
        return self.current_mask

    def heuristic_context(self):
        if self.state is None or self.current_request is None:
            raise RuntimeError("reset() must be called before requesting heuristic context")
        from optical_networking_gym_v2.heuristics.runtime_heuristics import build_runtime_heuristic_context

        return build_runtime_heuristic_context(self)

    def get_trace_action_mask(self) -> np.ndarray:
        if self.current_mask is not None:
            return self.current_mask
        if self.current_analysis is None:
            return np.zeros(self.total_actions, dtype=np.uint8)
        trace_mask = self._mask_from_analysis(self.current_analysis, force=True)
        if trace_mask is None:
            return np.zeros(self.total_actions, dtype=np.uint8)
        return trace_mask

    def export_captured_traffic_table(self) -> tuple[object, tuple[object, ...]]:
        if self.traffic_model is None:
            raise RuntimeError("reset() must be called before exporting a traffic table")
        return self.traffic_model.export_table()

    def save_captured_traffic_table_jsonl(self, file_path: str) -> object:
        if self.traffic_model is None:
            raise RuntimeError("reset() must be called before exporting a traffic table")
        return self.traffic_model.save_table_jsonl(file_path)

    def export_step_trace(self) -> dict[str, object]:
        if not self.capture_step_trace:
            raise RuntimeError("capture_step_trace must be enabled to export a step trace")
        accepted = 0
        blocked_resources = 0
        blocked_qot = 0
        rejected_by_agent = 0
        for step in self._captured_trace_steps:
            status = str(step["outcome"]["status"])
            if status == "accepted":
                accepted += 1
            elif status == "blocked_resources":
                blocked_resources += 1
            elif status == "blocked_qot":
                blocked_qot += 1
            elif status == "rejected_by_agent":
                rejected_by_agent += 1
        trace_payload = {
            "header": {
                "record_type": "trace_header",
                "trace_version": "v1",
                "topology_id": self.config.topology_id,
                "episode_length": int(self.episode_length),
                "num_spectrum_resources": int(self.config.num_spectrum_resources),
                "k_paths": int(self.config.k_paths),
                "bit_rates": self._trace_bit_rates(),
                "modulations": [str(modulation.name) for modulation in self.config.modulations],
                "qot_constraint": str(self.config.qot_constraint),
                "seed": self.config.seed,
                "table_id": self._trace_table_id(),
                "captured_requests": self._trace_captured_request_count(),
            },
            "steps": tuple(self._captured_trace_steps),
            "footer": {
                "record_type": "trace_footer",
                "steps": len(self._captured_trace_steps),
                "accepted": accepted,
                "blocked_resources": blocked_resources,
                "blocked_qot": blocked_qot,
                "rejected_by_agent": rejected_by_agent,
            },
        }
        return trace_payload  # normalize_trace_payload removed (legacy v1)

    def save_step_trace_jsonl(self, file_path: str) -> str:
        trace_payload = self.export_step_trace()
        return str(write_step_trace_jsonl(trace_payload, file_path))

    def _trace_float(self, value: float | int | None, digits: int = 12) -> float | None:
        if value is None:
            return None
        normalized = round(float(value), digits)
        if abs(normalized) < 1e-15:
            return 0.0
        return normalized

    def _trace_table_id(self) -> str | None:
        if self.traffic_model is not None:
            static_table = getattr(self.traffic_model, "_static_table", None)
            if static_table is not None:
                return static_table.table_id
            return getattr(self.traffic_model, "_table_id", None)
        return None

    def _trace_captured_request_count(self) -> int:
        if self.traffic_model is None:
            return 0
        static_table = getattr(self.traffic_model, "_static_table", None)
        if static_table is not None:
            return int(static_table.request_count)
        return int(len(getattr(self.traffic_model, "_captured_records", ())))

    def _trace_bit_rates(self) -> list[int]:
        if self.traffic_model is not None:
            static_records = getattr(self.traffic_model, "_static_records", ())
            if static_records:
                return sorted({int(record.bit_rate) for record in static_records})
        source = self.config.traffic_source
        if isinstance(source, dict) and "bit_rates" in source:
            return [int(bit_rate) for bit_rate in source["bit_rates"]]
        return []

    def _trace_request_payload(self, request: ServiceRequest | None) -> dict[str, object] | None:
        if request is None:
            return None
        return {
            "request_index": int(request.request_index),
            "service_id": int(request.service_id),
            "source_id": int(request.source_id),
            "destination_id": int(request.destination_id),
            "bit_rate": int(request.bit_rate),
            "arrival_time": self._trace_float(request.arrival_time),
            "holding_time": self._trace_float(request.holding_time),
            "table_id": request.table_id,
            "row_index": request.table_row_index if request.table_row_index is not None else request.request_index,
        }

    def _trace_active_service_payload(self, service: object) -> dict[str, object]:
        return {
            "service_id": int(service.service_id),
            "source_id": int(service.request.source_id),
            "destination_id": int(service.request.destination_id),
            "path_k": int(service.path.k),
            "path_node_names": [str(name) for name in service.path.node_names],
            "path_link_ids": [int(link_id) for link_id in service.path.link_ids],
            "service_slot_start": int(service.service_slot_start),
            "service_slot_end_exclusive": int(service.service_slot_end_exclusive),
            "occupied_slot_start": int(service.occupied_slot_start),
            "occupied_slot_end_exclusive": int(service.occupied_slot_end_exclusive),
            "modulation_name": None if service.modulation is None else str(service.modulation.name),
            "osnr": self._trace_float(service.osnr),
            "ase": self._trace_float(service.ase),
            "nli": self._trace_float(service.nli),
            # Legacy trace serializes the active-service release timestamp as the
            # direct arrival+holding sum, not the scheduled queue value.
            "release_time": self._trace_float(service.request.arrival_time + service.request.holding_time),
        }

    def _trace_state_snapshot(self) -> dict[str, object]:
        if self.state is None:
            raise RuntimeError("simulator state is not initialized")
        links_payload: list[dict[str, object]] = []
        for link in self.topology.links:
            slot_allocation = np.asarray(self.state.slot_allocation[link.id, :], dtype=np.int32)
            links_payload.append(
                {
                    "link_id": int(link.id),
                    "source": str(link.source_name),
                    "target": str(link.target_name),
                    "running_service_ids": sorted(
                        int(service_id) for service_id in self.state.link_active_service_ids[link.id]
                    ),
                    "available_slots": [int(value) for value in (slot_allocation == -1).astype(np.int32).tolist()],
                    "slot_allocation": [int(value) for value in slot_allocation.tolist()],
                }
            )
        active_services = sorted(
            self.state.active_services_by_id.values(),
            key=lambda service: int(service.service_id),
        )
        return {
            "current_time": self._trace_float(self.state.current_time),
            "release_queue": [
                [self._trace_float(release_time), int(service_id)]
                for release_time, service_id in self.state.release_queue_snapshot()
            ],
            "active_services": [
                self._trace_active_service_payload(service) for service in active_services
            ],
            "links": links_payload,
        }

    def _trace_action_details(
        self,
        action: int,
        analysis: RequestAnalysis,
        request: ServiceRequest,
    ) -> tuple[dict[str, object], dict[str, object] | None, float, dict[str, float] | None]:
        rejected_action = reject_action(self.config)
        decoded_action = decode_action(self.config, action)
        if decoded_action is None:
            return (
                {
                    "action_index": int(action),
                    "reject_action": int(rejected_action),
                    "decoded": {
                        "path_index": None,
                        "modulation_index": None,
                        "modulation_name": None,
                        "initial_slot": None,
                    },
                },
                None,
                0.0,
                None,
            )

        decoded_payload = {
            "path_index": int(decoded_action.path_index),
            "modulation_index": None,
            "modulation_name": None,
            "initial_slot": int(decoded_action.initial_slot),
        }
        if (
            decoded_action.path_index >= len(analysis.paths)
            or decoded_action.modulation_offset >= len(analysis.modulation_indices)
        ):
            return (
                {
                    "action_index": int(action),
                    "reject_action": int(rejected_action),
                    "decoded": decoded_payload,
                },
                None,
                0.0,
                None,
            )

        path = analysis.paths[decoded_action.path_index]
        modulation_index = int(analysis.modulation_indices[decoded_action.modulation_offset])
        modulation = self.config.modulations[modulation_index]
        decoded_payload["modulation_index"] = int(modulation_index)
        decoded_payload["modulation_name"] = str(modulation.name)
        service_num_slots = int(
            analysis.required_slots_by_path_mod[decoded_action.path_index, decoded_action.modulation_offset]
        )
        candidate_payload = None
        trace_candidate_qot = None
        if service_num_slots > 0:
            candidate = self.qot_engine.build_candidate(
                request=request,
                path=path,
                modulation=modulation,
                service_slot_start=decoded_action.initial_slot,
                service_num_slots=service_num_slots,
            )
            occupied_slot_end_exclusive = decoded_action.initial_slot + service_num_slots
            if occupied_slot_end_exclusive < self.config.num_spectrum_resources:
                occupied_slot_end_exclusive += 1
            candidate_payload = {
                "path_k": int(path.k),
                "path_node_names": [str(name) for name in path.node_names],
                "path_link_ids": [int(link_id) for link_id in path.link_ids],
                "path_hops": int(path.hops),
                "path_length_km": self._trace_float(path.length_km),
                "service_num_slots": int(service_num_slots),
                "service_slot_start": int(decoded_action.initial_slot),
                "service_slot_end_exclusive": int(decoded_action.initial_slot + service_num_slots),
                "occupied_slot_start": int(decoded_action.initial_slot),
                "occupied_slot_end_exclusive": int(occupied_slot_end_exclusive),
                "center_frequency": self._trace_float(candidate.center_frequency),
                "bandwidth": self._trace_float(candidate.bandwidth),
                "launch_power": self._trace_float(candidate.launch_power),
            }
            if self.config.qot_constraint != "DIST":
                qot_result = self.qot_engine.evaluate_candidate(self.state, candidate) if self.state is not None else None
                if qot_result is not None:
                    trace_candidate_qot = {
                        "osnr": float(qot_result.osnr),
                        "ase": float(qot_result.ase),
                        "nli": float(qot_result.nli),
                    }
        return (
            {
                "action_index": int(action),
                "reject_action": int(rejected_action),
                "decoded": decoded_payload,
            },
            candidate_payload,
            float(modulation.minimum_osnr + self.config.margin),
            trace_candidate_qot,
        )

    def _prepare_next_request(self) -> None:
        if self.traffic_model is None or self.state is None:
            raise RuntimeError("reset() must be called before preparing requests")
        request = self.traffic_model.next_request()
        released = self.state.advance_time_and_release_due_services(request.arrival_time)
        if self.config.measure_disruptions and released:
            impacted_ids: set[int] = set()
            for released_service in released:
                impacted_ids.update(
                    self.qot_engine.impacted_service_ids(
                        self.state,
                        released_service.path,
                        exclude_service_id=released_service.service_id,
                    )
                )
            if impacted_ids:
                disrupted_services, dropped_qot = self._refresh_impacted_services(tuple(sorted(impacted_ids)))
                if self.statistics is not None and (disrupted_services or dropped_qot):
                    self.statistics.record_post_admission_effects(
                        disrupted_services=disrupted_services,
                        dropped_qot=dropped_qot,
                    )
        self.state.set_current_request(request)
        self.current_request = request
        self.current_analysis = self.analysis_engine.build(self.state, request)
        if self.config.enable_observation:
            self.current_observation = self.observation_builder.build_from_analysis(self.current_analysis)
        else:
            self.current_observation = self._empty_observation
        self.current_mask = self._mask_from_analysis(self.current_analysis)

    def _apply_action(
        self,
        action: int,
        analysis: RequestAnalysis,
        current_mask: np.ndarray | None,
    ) -> StepTransition:
        if self.state is None or self.current_request is None:
            raise RuntimeError("simulator state is not initialized")
        if action < 0 or action >= self.total_actions:
            raise ValueError("action is outside the action space")

        request = self.current_request
        decoded = decode_action(self.config, action)
        if decoded is None:
            return StepTransition(
                request=request,
                allocation=Allocation.reject(Status.REJECTED_BY_AGENT),
                action=action,
                mask=current_mask,
            )

        if (
            decoded.path_index >= len(analysis.paths)
            or decoded.modulation_offset >= len(analysis.modulation_indices)
        ):
            return StepTransition(
                request=request,
                allocation=Allocation.reject(Status.BLOCKED_RESOURCES),
                action=action,
                mask=current_mask,
            )

        if not analysis.resource_valid_starts[decoded.path_index, decoded.modulation_offset, decoded.initial_slot]:
            return StepTransition(
                request=request,
                allocation=Allocation.reject(Status.BLOCKED_RESOURCES),
                action=action,
                mask=current_mask,
            )

        if (
            self.config.mask_mode.value == "resource_and_qot"
            and not analysis.qot_valid_starts[decoded.path_index, decoded.modulation_offset, decoded.initial_slot]
        ):
            return StepTransition(
                request=request,
                allocation=Allocation.reject(Status.BLOCKED_QOT),
                action=action,
                mask=current_mask,
            )

        path = analysis.paths[decoded.path_index]
        modulation_index = int(analysis.modulation_indices[decoded.modulation_offset])
        modulation = self.config.modulations[modulation_index]
        service_num_slots = int(analysis.required_slots_by_path_mod[decoded.path_index, decoded.modulation_offset])
        occupied_slot_start = decoded.initial_slot
        occupied_slot_end_exclusive = decoded.initial_slot + service_num_slots
        if occupied_slot_end_exclusive < self.config.num_spectrum_resources:
            occupied_slot_end_exclusive += 1

        allocation = Allocation.accept(
            path_index=decoded.path_index,
            modulation_index=modulation_index,
            service_slot_start=decoded.initial_slot,
            service_num_slots=service_num_slots,
            occupied_slot_start=occupied_slot_start,
            occupied_slot_end_exclusive=occupied_slot_end_exclusive,
        )

        candidate = self.qot_engine.build_candidate(
            request=request,
            path=path,
            modulation=modulation,
            service_slot_start=decoded.initial_slot,
            service_num_slots=service_num_slots,
        )
        if self.config.qot_constraint == "DIST":
            osnr = 0.0
            ase = 0.0
            nli = 0.0
        else:
            qot_result = self.qot_engine.evaluate_candidate(self.state, candidate)
            osnr = qot_result.osnr
            ase = qot_result.ase
            nli = qot_result.nli

        self.state.apply_provision(
            request=request,
            path=path,
            service_slot_start=decoded.initial_slot,
            service_num_slots=service_num_slots,
            occupied_slot_start=occupied_slot_start,
            occupied_slot_end_exclusive=occupied_slot_end_exclusive,
            modulation=modulation,
            center_frequency=candidate.center_frequency,
            bandwidth=candidate.bandwidth,
            launch_power=candidate.launch_power,
        )
        self.state.apply_qot_updates(
            {request.service_id: {"osnr": osnr, "ase": ase, "nli": nli}}
        )
        provisioned_service = self.state.active_services_by_id[request.service_id]
        provisioned_service.established_osnr = float(osnr)
        provisioned_service.shared_link_service_count_at_establishment = int(
            len(self.qot_engine.impacted_service_ids(self.state, path, exclude_service_id=request.service_id))
        )

        disrupted_services = 0
        dropped_qot = 0
        if self.config.measure_disruptions:
            impacted_ids = self.qot_engine.impacted_service_ids(
                self.state,
                path,
                exclude_service_id=request.service_id,
            )
            if impacted_ids:
                disrupted_services, dropped_qot = self._refresh_impacted_services(tuple(sorted(impacted_ids)))

        transition = StepTransition.accept(
            request=request,
            allocation=allocation,
            modulation_spectral_efficiency=modulation.spectral_efficiency,
            action=action,
            osnr=osnr,
            osnr_requirement=modulation.minimum_osnr + self.config.margin,
            disrupted_services=disrupted_services,
            dropped_qot=dropped_qot,
            fragmentation_shannon_entropy=self._fragmentation_shannon_entropy(analysis),
            fragmentation_route_cuts=self._fragmentation_route_cuts(analysis, decoded.path_index),
            fragmentation_route_rss=self._fragmentation_route_rss(analysis, decoded.path_index),
            mask=current_mask,
        )
        return transition

    def _refresh_impacted_services(self, impacted_ids: tuple[int, ...]) -> tuple[int, int]:
        if self.state is None:
            return 0, 0

        pending_ids = tuple(
            service_id for service_id in impacted_ids if service_id in self.state.active_services_by_id
        )
        total_disrupted = 0
        total_dropped = 0

        while pending_ids:
            updates = self.qot_engine.refresh_services(self.state, pending_ids)
            if updates:
                self.state.apply_qot_updates(
                    {
                        update.service_id: update.to_mapping()
                        for update in updates
                        if update.service_id in self.state.active_services_by_id
                    }
                )

            candidate_failures: list[tuple[float, int]] = []
            for update in updates:
                if update.service_id not in self.state.active_services_by_id:
                    continue
                service = self.state.active_services_by_id[update.service_id]
                modulation = service.modulation
                if modulation is None:
                    continue
                threshold = modulation.minimum_osnr + self.config.margin
                if update.osnr >= threshold or update.service_id in self._disrupted_service_ids:
                    continue
                candidate_failures.append((float(update.osnr - threshold), int(update.service_id)))

            if not candidate_failures:
                break

            worst_margin, selected_service_id = min(candidate_failures, key=lambda item: (item[0], item[1]))
            self._disrupted_service_ids.add(selected_service_id)
            selected_service = self.state.apply_disruption(
                selected_service_id,
                terminal=self.config.drop_on_disruption,
            )
            total_disrupted += 1

            if not self.config.drop_on_disruption:
                break

            total_dropped += 1
            remaining_pending_ids = {
                service_id
                for service_id in pending_ids
                if service_id != selected_service_id and service_id in self.state.active_services_by_id
            }
            release_impacted_ids = set(
                self.qot_engine.impacted_service_ids(
                    self.state,
                    selected_service.path,
                    exclude_service_id=selected_service_id,
                )
            )
            pending_ids = tuple(
                sorted(
                    service_id
                    for service_id in (remaining_pending_ids | release_impacted_ids)
                    if service_id in self.state.active_services_by_id
                )
            )

        return total_disrupted, total_dropped

    def _mask_from_analysis(self, analysis: RequestAnalysis, *, force: bool = False) -> np.ndarray | None:
        if analysis.action_mask is None and not force:
            return None
        mask = np.zeros(self.total_actions, dtype=np.uint8)
        if analysis.action_mask is not None:
            mask[:-1] = analysis.action_mask
        else:
            flags = (
                analysis.resource_valid_starts
                if self.config.mask_mode.value == "resource_only"
                else analysis.qot_valid_starts
            )
            mask[:-1] = flags.reshape(-1).astype(np.uint8)
        mask[-1] = 1
        mask.flags.writeable = False
        return mask

    def _fragmentation_shannon_entropy(self, analysis: RequestAnalysis) -> float:
        return float(analysis.mean_link_entropy)

    def _fragmentation_route_cuts(self, analysis: RequestAnalysis, path_index: int) -> float:
        return float(analysis.path_route_cuts_norm_by_path[path_index] * max(1, self.topology.link_count * 2.0))

    def _fragmentation_route_rss(self, analysis: RequestAnalysis, path_index: int) -> float:
        return float(analysis.path_route_rss_by_path[path_index])

    def _build_runtime_helpers(self, config: ScenarioConfig) -> None:
        self.qot_engine = QoTEngine(config, self.topology)
        self.analysis_engine = RequestAnalysisEngine(config, self.topology, self.qot_engine)
        self.action_mask_builder = ActionMask(
            config,
            self.topology,
            self.qot_engine,
            analysis_engine=self.analysis_engine,
        )
        self.observation_builder = Observation(
            config,
            self.topology,
            self.analysis_engine,
        )
        self.reward_function = RewardFunction(config, self.topology)
        self.step_info_builder = StepInfo(config)
        self._helper_structure_key = config.runtime_structure_key()

    def _apply_runtime_config(self, config: ScenarioConfig) -> None:
        self.config = config
        self.episode_length = config.episode_length
        self.capture_traffic_table = config.capture_traffic_table
        self.capture_step_trace = config.capture_step_trace
        if self._helper_structure_key != config.runtime_structure_key():
            self._build_runtime_helpers(config)
            return
        self.qot_engine.config = config
        self.analysis_engine.config = config
        self.analysis_engine.clear_cache()
        self.action_mask_builder.config = config
        self.observation_builder.config = config
        self.reward_function.config = config
        self.step_info_builder.config = config

    def _copy_observation(self, observation: np.ndarray | None) -> np.ndarray:
        if observation is None:
            return self._terminal_observation()
        if not self.config.enable_observation:
            return self._empty_observation
        return observation.copy()

    def _terminal_observation(self) -> np.ndarray:
        if not self.config.enable_observation:
            return self._empty_observation
        return np.zeros(self.observation_builder.schema.total_size, dtype=np.float32)

    def _build_reset_info(self) -> dict[str, object]:
        return {"mask": self._info_mask(self.current_mask)}

    def _info_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        if not self.config.enable_action_mask:
            return None
        if not self.config.include_mask_in_info:
            return None
        return mask


__all__ = ["Simulator"]
