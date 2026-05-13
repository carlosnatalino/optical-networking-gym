from __future__ import annotations

import heapq

import numpy as np
import optical_networking_gym.core.osnr as legacy_osnr

from optical_networking_gym_v2 import (
    Modulation,
    ScenarioConfig,
    Simulator,
    Status,
    TopologyModel,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)

from .helpers import (
    build_legacy_action_mask,
    build_legacy_qrmsa_env,
    build_legacy_service,
)


def _topology() -> TopologyModel:
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    ring_4_path = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"
    return TopologyModel.from_file(ring_4_path, topology_id="ring_4", k_paths=2)


def _modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def _records() -> tuple[TrafficRecord, ...]:
    table_id = "equivalence_static"
    return (
        TrafficRecord(0, 0, 0, 2, 40, 1.0, 8.0, table_id=table_id, row_index=0),
        TrafficRecord(1, 1, 0, 2, 40, 2.0, 8.0, table_id=table_id, row_index=1),
        TrafficRecord(2, 2, 0, 2, 40, 12.0, 4.0, table_id=table_id, row_index=2),
    )


def _config() -> ScenarioConfig:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="equivalence_static",
        scenario_id="equivalence_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=3,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    return ScenarioConfig(
        scenario_id="equivalence_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source={"table": table, "records": _records()},
        modulations=_modulations(),
        modulations_to_consider=2,
    )


def _first_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask[:-1])
    return int(valid[0]) if valid.size > 0 else int(mask.shape[0] - 1)


def _prepare_legacy_env():
    legacy_env = build_legacy_qrmsa_env(k_paths=2, gen_observation=False, reset=True)
    legacy_env.current_time = 0.0
    legacy_env.current_service = None
    counters = {
        "accepted": 0,
        "processed": 0,
    }
    return legacy_env, [], counters


def _legacy_prepare_request(legacy_env, release_events: list[tuple[float, int, object]], record: TrafficRecord) -> None:
    while release_events:
        release_time, _, service = heapq.heappop(release_events)
        if release_time <= record.arrival_time:
            legacy_env.current_time = release_time
            legacy_env._release_path(service)
            continue
        heapq.heappush(release_events, (release_time, service.service_id, service))
        break

    legacy_env.current_time = record.arrival_time
    legacy_env.current_service = build_legacy_service(record.to_service_request(), legacy_env)


def _legacy_step_current(
    legacy_env,
    release_events: list[tuple[float, int, object]],
    counters: dict[str, int],
    action: int,
) -> str:
    service = legacy_env.current_service
    if service is None:
        raise RuntimeError("legacy current service is required")

    counters["processed"] += 1

    status = Status.REJECTED_BY_AGENT.value
    if action == legacy_env.action_space.n - 1:
        service.accepted = False
        service.blocked_due_to_resources = False
        service.blocked_due_to_osnr = False
        legacy_env.bl_reject += 1
        return status

    route_index, modulation_index, initial_slot = legacy_env.encoded_decimal_to_array(
        action,
        [legacy_env.k_paths, legacy_env.modulations_to_consider, legacy_env.num_spectrum_resources],
    )
    modulation = legacy_env.modulations[modulation_index]
    path = legacy_env.k_shortest_paths[service.source, service.destination][route_index]
    number_slots = legacy_env.get_number_slots(service=service, modulation=modulation)

    if not legacy_env.is_path_free(path=path, initial_slot=initial_slot, number_slots=number_slots):
        service.accepted = False
        service.blocked_due_to_resources = True
        service.blocked_due_to_osnr = False
        legacy_env.bl_resource += 1
        return Status.BLOCKED_RESOURCES.value

    service.path = path
    service.initial_slot = initial_slot
    service.number_slots = number_slots
    service.center_frequency = (
        legacy_env.frequency_start
        + (legacy_env.frequency_slot_bandwidth * initial_slot)
        + (legacy_env.frequency_slot_bandwidth * (number_slots / 2.0))
    )
    service.bandwidth = legacy_env.frequency_slot_bandwidth * number_slots
    service.launch_power = legacy_env.launch_power

    if legacy_env.qot_constraint == "DIST":
        qot_acceptable = path.length <= modulation.maximum_length
        osnr = 0.0
        ase = 0.0
        nli = 0.0
    else:
        osnr, ase, nli = legacy_osnr.calculate_osnr(legacy_env, service, legacy_env.qot_constraint)
        qot_acceptable = osnr >= modulation.minimum_osnr + legacy_env.margin

    if not qot_acceptable:
        service.accepted = False
        service.blocked_due_to_resources = False
        service.blocked_due_to_osnr = True
        service.path = None
        service.initial_slot = -1
        service.number_slots = 0
        service.center_frequency = 0.0
        service.bandwidth = 0.0
        service.launch_power = 0.0
        legacy_env.bl_osnr += 1
        return Status.BLOCKED_QOT.value

    service.accepted = True
    service.blocked_due_to_resources = False
    service.blocked_due_to_osnr = False
    service.OSNR = osnr
    service.ASE = ase
    service.NLI = nli
    service.current_modulation = modulation
    legacy_env._provision_path(path, initial_slot, number_slots)
    counters["accepted"] += 1
    heapq.heappush(
        release_events,
        (service.arrival_time + service.holding_time, service.service_id, service),
    )
    return Status.ACCEPTED.value


def test_simulator_matches_legacy_for_scripted_static_episode() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=3)
    _, info = simulator.reset(seed=7)

    legacy_env, release_events, counters = _prepare_legacy_env()
    statuses = []
    legacy_statuses = []

    for step_index, record in enumerate(_records()):
        _legacy_prepare_request(legacy_env, release_events, record)
        legacy_mask = build_legacy_action_mask(legacy_env)
        v2_mask = simulator.action_masks()

        assert v2_mask is not None
        assert np.array_equal(v2_mask, legacy_mask)

        action = _first_valid_action(v2_mask)
        _, _, terminated, _, step_info = simulator.step(action)
        legacy_status = _legacy_step_current(legacy_env, release_events, counters, action)

        statuses.append(step_info["status"])
        legacy_statuses.append(legacy_status)

        assert step_info["status"] == legacy_status
        assert step_info["episode_services_accepted"] == counters["accepted"]

        if step_index < len(_records()) - 1:
            assert terminated is False

    assert statuses == legacy_statuses == [Status.ACCEPTED.value] * len(_records())
    assert np.array_equal(
        (simulator.state.slot_allocation == -1).astype(np.int32),
        legacy_env.topology.graph["available_slots"],
    )
