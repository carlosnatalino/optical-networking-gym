from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import optical_networking_gym.core.osnr as legacy_osnr
    from optical_networking_gym.envs.qrmsa import QRMSAEnv, Service
    from optical_networking_gym.topology import Modulation, get_topology
except ModuleNotFoundError:  # pragma: no cover - depends on optional legacy package
    pytest.skip(
        "Legacy optical_networking_gym is required for equivalence tests",
        allow_module_level=True,
    )
from optical_networking_gym_v2 import Modulation as V2Modulation, ScenarioConfig, ServiceRequest, TopologyModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def build_ring4_topology_v2(
    *,
    k_paths: int = 2,
    max_span_length_km: float = 100.0,
    default_attenuation_db_per_km: float = 0.2,
    default_noise_figure_db: float = 4.5,
) -> TopologyModel:
    return TopologyModel.from_file(
        RING_4_PATH,
        topology_id="ring_4",
        k_paths=k_paths,
        max_span_length_km=max_span_length_km,
        default_attenuation_db_per_km=default_attenuation_db_per_km,
        default_noise_figure_db=default_noise_figure_db,
    )


def build_action_mask_modulations_v2() -> tuple[V2Modulation, ...]:
    return (
        V2Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        V2Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def build_ring4_config(
    *,
    scenario_id: str,
    num_spectrum_resources: int = 24,
    k_paths: int = 2,
    modulations: tuple[V2Modulation, ...] = (),
    modulations_to_consider: int | None = None,
) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=scenario_id,
        topology_id="ring_4",
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        modulations=modulations,
        modulations_to_consider=modulations_to_consider,
    )


def build_legacy_qrmsa_env(
    *,
    num_spectrum_resources: int = 24,
    k_paths: int = 2,
    gen_observation: bool = False,
    reset: bool = True,
) -> QRMSAEnv:
    topology = get_topology(
        str(RING_4_PATH),
        topology_name="ring_4",
        modulations=(
            Modulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17),
            Modulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
        ),
        max_span_length=100.0,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=k_paths,
    )
    return QRMSAEnv(
        topology=topology,
        num_spectrum_resources=num_spectrum_resources,
        episode_length=10,
        load=10.0,
        mean_service_holding_time=100.0,
        bit_rate_selection="discrete",
        bit_rates=(40,),
        bit_rate_probabilities=(1.0,),
        bandwidth=num_spectrum_resources * 12.5e9,
        seed=7,
        reset=reset,
        gen_observation=gen_observation,
        k_paths=k_paths,
    )


def build_service_request(*, service_id: int, source_id: int = 0, destination_id: int = 2) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=source_id,
        destination_id=destination_id,
        bit_rate=40,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def build_legacy_service(request: ServiceRequest, legacy_env: QRMSAEnv) -> Service:
    source_name = legacy_env.topology.graph["node_indices"][request.source_id]
    destination_name = legacy_env.topology.graph["node_indices"][request.destination_id]
    return Service(
        service_id=request.service_id,
        source=source_name,
        source_id=request.source_id,
        destination=destination_name,
        destination_id=str(request.destination_id),
        arrival_time=request.arrival_time,
        holding_time=request.holding_time,
        bit_rate=request.bit_rate,
    )


def build_legacy_action_mask(legacy_env: QRMSAEnv) -> np.ndarray:
    current_service = legacy_env.current_service
    if current_service is None:
        raise ValueError("legacy_env.current_service is required")

    legacy_env.get_max_modulation_index()
    paths_info: list[tuple[object, np.ndarray]] = []
    for path_index, route in enumerate(
        legacy_env.k_shortest_paths[current_service.source, current_service.destination]
    ):
        if path_index >= legacy_env.k_paths:
            break
        paths_info.append((route, legacy_env.get_available_slots(route)))

    total_actions = legacy_env.k_paths * legacy_env.modulations_to_consider * legacy_env.num_spectrum_resources
    action_mask = np.zeros(total_actions + 1, dtype=np.uint8)
    path_modulations_cache: dict[int, list[object]] = {}
    path_window_masks: dict[tuple[int, int], np.ndarray] = {}

    for action_index in range(total_actions):
        path_index = action_index // (
            legacy_env.modulations_to_consider * legacy_env.num_spectrum_resources
        )
        modulation_and_slot = action_index % (
            legacy_env.modulations_to_consider * legacy_env.num_spectrum_resources
        )
        modulation_offset = modulation_and_slot // legacy_env.num_spectrum_resources
        initial_slot = modulation_and_slot % legacy_env.num_spectrum_resources

        if path_index >= len(paths_info):
            continue

        route, available_slots = paths_info[path_index]
        if available_slots[initial_slot] == 0:
            continue

        if path_index not in path_modulations_cache:
            start_index = max(
                0,
                legacy_env.max_modulation_idx - (legacy_env.modulations_to_consider - 1),
            )
            path_modulations_cache[path_index] = list(
                reversed(
                    legacy_env.modulations[
                        start_index : legacy_env.max_modulation_idx + 1
                    ][: legacy_env.modulations_to_consider]
                )
            )
        modulation_list = path_modulations_cache[path_index]
        if modulation_offset >= len(modulation_list):
            continue
        modulation = modulation_list[modulation_offset]
        number_slots = legacy_env.get_number_slots(current_service, modulation)

        if initial_slot + number_slots > legacy_env.num_spectrum_resources:
            continue

        base_key = (path_index, number_slots)
        if base_key not in path_window_masks:
            path_window_masks[base_key] = _window_mask(available_slots, number_slots)
        base_mask = path_window_masks[base_key]
        if initial_slot >= base_mask.shape[0] or base_mask[initial_slot] == 0:
            continue

        guard_needed = (initial_slot + number_slots) < legacy_env.num_spectrum_resources
        if guard_needed:
            guard_window = number_slots + 1
            guard_key = (path_index, guard_window)
            if guard_key not in path_window_masks:
                path_window_masks[guard_key] = _window_mask(available_slots, guard_window)
            guard_mask = path_window_masks[guard_key]
            if initial_slot >= guard_mask.shape[0] or guard_mask[initial_slot] == 0:
                continue

        current_service.path = route
        current_service.initial_slot = initial_slot
        current_service.number_slots = number_slots
        current_service.center_frequency = (
            legacy_env.frequency_start
            + legacy_env.frequency_slot_bandwidth * (initial_slot + number_slots / 2)
        )
        current_service.bandwidth = legacy_env.frequency_slot_bandwidth * number_slots
        current_service.launch_power = legacy_env.launch_power

        if legacy_env.qot_constraint == "DIST":
            qot_acceptable = route.length <= modulation.maximum_length
        else:
            osnr, _, _ = legacy_osnr.calculate_osnr(legacy_env, current_service, legacy_env.qot_constraint)
            qot_acceptable = osnr >= modulation.minimum_osnr + legacy_env.margin

        current_service.path = None
        current_service.initial_slot = -1
        current_service.number_slots = 0
        current_service.center_frequency = 0.0
        current_service.bandwidth = 0.0
        current_service.launch_power = 0.0

        if qot_acceptable:
            action_mask[action_index] = 1

    action_mask[-1] = 1
    return action_mask


def _window_mask(available_slots: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0 or window_size > available_slots.shape[0]:
        return np.zeros((0,), dtype=np.uint8)
    slot_view = np.asarray(available_slots, dtype=np.int8)
    kernel = np.ones(window_size, dtype=np.int8)
    window_sums = np.convolve(slot_view, kernel, mode="valid")
    return (window_sums == window_size).astype(np.uint8)
