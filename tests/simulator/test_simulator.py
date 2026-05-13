from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym_v2 import (
    MaskMode,
    Modulation,
    QoTEngine,
    QoTResult,
    ServiceQoTUpdate,
    Simulator,
    ScenarioConfig,
    Status,
    TopologyModel,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)
from optical_networking_gym_v2.runtime.action_codec import encode_action


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RING_4_PATH = PROJECT_ROOT / "optical_networking_gym_v2" / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def _static_source() -> dict[str, object]:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="simulator_static",
        scenario_id="simulator_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=3,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 0, 2, 40, 1.0, 8.0, table_id=table.table_id, row_index=0),
        TrafficRecord(1, 1, 0, 2, 40, 2.0, 8.0, table_id=table.table_id, row_index=1),
        TrafficRecord(2, 2, 0, 2, 40, 12.0, 4.0, table_id=table.table_id, row_index=2),
    )
    return {"table": table, "records": records}


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="simulator_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=_static_source(),
        modulations=_modulations(),
        modulations_to_consider=2,
    )


def _reverse_static_source() -> dict[str, object]:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="simulator_reverse_static",
        scenario_id="simulator_reverse_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=1,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 1, 0, 40, 1.0, 8.0, table_id=table.table_id, row_index=0),
    )
    return {"table": table, "records": records}


def _reverse_config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="simulator_reverse_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=_reverse_static_source(),
        modulations=_modulations(),
        modulations_to_consider=2,
    )


def _disruption_static_source(service_order: tuple[int, ...] = (0, 1, 2, 3)) -> dict[str, object]:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="simulator_disruption_static",
        scenario_id="simulator_disruption_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=len(service_order),
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = tuple(
        TrafficRecord(
            request_index=index,
            service_id=service_id,
            source_id=0,
            destination_id=2,
            bit_rate=40,
            arrival_time=1.0 + index,
            holding_time=20.0,
            table_id=table.table_id,
            row_index=index,
        )
        for index, service_id in enumerate(service_order)
    )
    return {"table": table, "records": records}


def _disruption_config(service_order: tuple[int, ...] = (0, 1, 2, 3)) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="simulator_disruption_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=_disruption_static_source(service_order),
        modulations=(Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),),
        modulations_to_consider=1,
        measure_disruptions=True,
    )


def _drop_on_disruption_static_source() -> dict[str, object]:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="simulator_drop_on_disruption_static",
        scenario_id="simulator_drop_on_disruption_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=3,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 0, 2, 40, 1.0, 20.0, table_id=table.table_id, row_index=0),
        TrafficRecord(1, 1, 0, 2, 40, 2.0, 1.0, table_id=table.table_id, row_index=1),
        TrafficRecord(2, 2, 0, 2, 40, 4.0, 20.0, table_id=table.table_id, row_index=2),
    )
    return {"table": table, "records": records}


def _drop_on_disruption_config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="simulator_drop_on_disruption_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=_drop_on_disruption_static_source(),
        modulations=(Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),),
        modulations_to_consider=1,
        mask_mode=MaskMode.RESOURCE_ONLY,
        measure_disruptions=True,
        drop_on_disruption=True,
    )


def _cascading_drop_static_source() -> dict[str, object]:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="simulator_cascading_drop_static",
        scenario_id="simulator_cascading_drop_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=3,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 0, 2, 40, 1.0, 20.0, table_id=table.table_id, row_index=0),
        TrafficRecord(1, 1, 0, 2, 40, 2.0, 20.0, table_id=table.table_id, row_index=1),
        TrafficRecord(2, 2, 0, 2, 40, 3.0, 20.0, table_id=table.table_id, row_index=2),
    )
    return {"table": table, "records": records}


def _cascading_drop_config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="simulator_cascading_drop_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=_cascading_drop_static_source(),
        modulations=(Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),),
        modulations_to_consider=1,
        mask_mode=MaskMode.RESOURCE_ONLY,
        measure_disruptions=True,
        drop_on_disruption=True,
    )


def _first_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask[:-1])
    if valid.size == 0:
        return int(mask.shape[0] - 1)
    return int(valid[0])


def _accept_at_slot(
    simulator: Simulator,
    mask: np.ndarray,
    *,
    path_index: int,
    modulation_offset: int,
    initial_slot: int,
):
    action = encode_action(
        simulator.config,
        path_index=path_index,
        modulation_offset=modulation_offset,
        initial_slot=initial_slot,
    )
    assert mask[action] == 1
    return simulator.step(action)


def _assert_fresh_qot_matches_state(simulator: Simulator, service_ids: tuple[int, ...]) -> None:
    assert simulator.state is not None
    oracle = QoTEngine(simulator.config, simulator.topology)

    for service_id in service_ids:
        expected = oracle.recompute_service(simulator.state, service_id)
        stored = simulator.state.active_services_by_id[service_id]
        assert stored.osnr == pytest.approx(expected.osnr, abs=1e-9)
        assert stored.ase == pytest.approx(expected.ase, abs=1e-9)
        assert stored.nli == pytest.approx(expected.nli, abs=1e-9)


def _snapshot_qot_state(
    simulator: Simulator,
    service_ids: tuple[int, ...],
) -> dict[int, tuple[float, float, float]]:
    assert simulator.state is not None
    return {
        service_id: (
            simulator.state.active_services_by_id[service_id].osnr,
            simulator.state.active_services_by_id[service_id].ase,
            simulator.state.active_services_by_id[service_id].nli,
        )
        for service_id in service_ids
    }


def _run_disruption_sequence(service_order: tuple[int, ...]) -> dict[int, tuple[float, float, float]]:
    simulator = Simulator(_disruption_config(service_order), _topology(), episode_length=len(service_order))
    _, info = simulator.reset(seed=7)

    assert simulator.state is not None
    assert simulator.current_analysis is not None

    service_num_slots = int(simulator.current_analysis.required_slots_by_path_mod[0, 0])
    slot_spacing = service_num_slots + 1
    planned_slots_by_service = {
        service_id: index * slot_spacing for index, service_id in enumerate(sorted(service_order))
    }
    highest_slot = max(planned_slots_by_service.values(), default=0)
    assert highest_slot + service_num_slots <= simulator.config.num_spectrum_resources

    for provisioned_count, service_id in enumerate(service_order, start=1):
        _, _, terminated, truncated, info = _accept_at_slot(
            simulator,
            info["mask"],
            path_index=0,
            modulation_offset=0,
            initial_slot=planned_slots_by_service[service_id],
        )

        assert terminated is (provisioned_count == len(service_order))
        assert truncated is False
        assert info["status"] == Status.ACCEPTED.value

        active_service_ids = tuple(sorted(simulator.state.active_services_by_id))
        assert active_service_ids == tuple(sorted(service_order[:provisioned_count]))
        _assert_fresh_qot_matches_state(simulator, active_service_ids)

    final_service_ids = tuple(sorted(simulator.state.active_services_by_id))
    return _snapshot_qot_state(simulator, final_service_ids)


def test_simulator_reset_returns_initial_observation_and_mask() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=3)

    observation, info = simulator.reset(seed=7)

    assert observation.dtype == np.float32
    assert observation.shape == (simulator.observation_builder.schema.total_size,)
    assert "mask" in info
    assert info["mask"].shape == (simulator.total_actions,)
    assert info["mask"][-1] == 1
    assert simulator.current_request is not None


def test_simulator_processes_static_episode_and_releases_due_services() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=3)
    observation, info = simulator.reset(seed=7)
    assert observation.shape[0] > 0

    first_reward = 0.0
    final_info = {}
    for step_index in range(3):
        action = _first_valid_action(info["mask"])
        observation, reward, terminated, truncated, info = simulator.step(action)
        final_info = info
        if step_index == 0:
            first_reward = reward
            assert info["status"] == Status.ACCEPTED.value
            assert simulator.state is not None
            assert len(simulator.state.active_services_by_id) == 1
        if step_index == 1:
            assert info["status"] == Status.ACCEPTED.value
            assert simulator.state is not None
            assert len(simulator.state.active_services_by_id) == 0
        if step_index == 2:
            assert terminated is True
            assert truncated is False
            assert observation.shape == (simulator.observation_builder.schema.total_size,)
            assert np.count_nonzero(observation) == 0

    assert first_reward > 0.0
    assert final_info["episode_services_processed"] == 3
    assert final_info["episode_services_accepted"] == 3


def test_simulator_measure_disruptions_refreshes_established_services_with_fresh_recompute() -> None:
    simulator = Simulator(_disruption_config(), _topology(), episode_length=4)
    _, info = simulator.reset(seed=7)

    assert simulator.state is not None
    assert simulator.current_analysis is not None

    service_num_slots = int(simulator.current_analysis.required_slots_by_path_mod[0, 0])
    slot_spacing = service_num_slots + 1
    planned_slots = tuple(index * slot_spacing for index in range(4))
    assert planned_slots[-1] + service_num_slots <= simulator.config.num_spectrum_resources

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=planned_slots[0],
    )
    assert terminated is False
    assert truncated is False
    assert info["status"] == Status.ACCEPTED.value
    initial_s1_osnr = simulator.state.active_services_by_id[0].osnr
    _assert_fresh_qot_matches_state(simulator, (0,))

    for provisioned_count, initial_slot in enumerate(planned_slots[1:], start=2):
        previous_s1_osnr = simulator.state.active_services_by_id[0].osnr
        _, _, terminated, truncated, info = _accept_at_slot(
            simulator,
            info["mask"],
            path_index=0,
            modulation_offset=0,
            initial_slot=initial_slot,
        )

        assert terminated is (provisioned_count == 4)
        assert truncated is False
        assert info["status"] == Status.ACCEPTED.value

        active_service_ids = tuple(sorted(simulator.state.active_services_by_id))
        assert active_service_ids == tuple(range(provisioned_count))
        _assert_fresh_qot_matches_state(simulator, active_service_ids)

        updated_s1_osnr = simulator.state.active_services_by_id[0].osnr
        assert updated_s1_osnr != pytest.approx(previous_s1_osnr, abs=1e-9)

    assert simulator.state.active_services_by_id[0].osnr != pytest.approx(initial_s1_osnr, abs=1e-9)


def test_simulator_measure_disruptions_final_qot_is_order_independent() -> None:
    baseline_snapshot = _run_disruption_sequence((0, 1, 2, 3))
    permuted_snapshot = _run_disruption_sequence((2, 0, 3, 1))

    assert tuple(sorted(baseline_snapshot)) == (0, 1, 2, 3)
    assert tuple(sorted(permuted_snapshot)) == (0, 1, 2, 3)

    for service_id, baseline_metrics in baseline_snapshot.items():
        permuted_metrics = permuted_snapshot[service_id]
        assert permuted_metrics[0] == pytest.approx(baseline_metrics[0], abs=1e-9)
        assert permuted_metrics[1] == pytest.approx(baseline_metrics[1], abs=1e-9)
        assert permuted_metrics[2] == pytest.approx(baseline_metrics[2], abs=1e-9)


def test_simulator_drop_on_disruption_moves_service_to_limbo_and_stabilizes(monkeypatch) -> None:
    simulator = Simulator(_drop_on_disruption_config(), _topology(), episode_length=3)
    _, info = simulator.reset(seed=7)

    assert simulator.state is not None

    def evaluate_candidate_high(*_args, **_kwargs) -> QoTResult:
        return QoTResult(osnr=25.0, ase=30.0, nli=20.0, meets_threshold=True)

    refresh_calls: list[tuple[int, ...]] = []

    def refresh_services_controlled(state, service_ids):
        refresh_calls.append(tuple(service_ids))
        updates: list[ServiceQoTUpdate] = []
        for service_id in service_ids:
            if service_id == 0 and 1 in state.active_services_by_id:
                updates.append(ServiceQoTUpdate(service_id=0, osnr=5.0, ase=30.0, nli=20.0))
            else:
                updates.append(ServiceQoTUpdate(service_id=service_id, osnr=25.0, ase=30.0, nli=20.0))
        return tuple(updates)

    monkeypatch.setattr(simulator.qot_engine, "evaluate_candidate", evaluate_candidate_high)
    monkeypatch.setattr(simulator.qot_engine, "refresh_services", refresh_services_controlled)

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=0,
    )
    assert terminated is False
    assert truncated is False
    assert tuple(sorted(simulator.state.active_services_by_id)) == (0,)

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=3,
    )

    assert terminated is False
    assert truncated is False
    assert refresh_calls[0] == (0,)
    assert 0 not in simulator.state.active_services_by_id
    assert 0 in simulator.state.disrupted_services_by_id
    disrupted_service = simulator.state.disrupted_services_by_id[0]
    assert disrupted_service.is_disrupted is True
    assert disrupted_service.disruption_count == 1
    assert all(service_id != 0 for _release_time, service_id in simulator.state.release_queue_snapshot())
    assert not np.any(simulator.state.slot_allocation == 0)
    assert tuple(sorted(simulator.state.active_services_by_id)) == ()

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=6,
    )

    assert terminated is True
    assert truncated is False
    assert tuple(sorted(simulator.state.active_services_by_id)) == (2,)
    assert 0 in simulator.state.disrupted_services_by_id
    assert info["blocked_due_to_osnr"] == 0
    assert info["disrupted_or_dropped_services"] == 1


def test_simulator_drop_on_disruption_recomputes_until_state_stabilizes(monkeypatch) -> None:
    simulator = Simulator(_cascading_drop_config(), _topology(), episode_length=3)
    _, info = simulator.reset(seed=7)

    assert simulator.state is not None

    def evaluate_candidate_high(*_args, **_kwargs) -> QoTResult:
        return QoTResult(osnr=25.0, ase=30.0, nli=20.0, meets_threshold=True)

    refresh_calls: list[tuple[int, ...]] = []

    def refresh_services_cascading(state, service_ids):
        call = tuple(service_ids)
        refresh_calls.append(call)
        if call == (0,):
            return (ServiceQoTUpdate(service_id=0, osnr=25.0, ase=30.0, nli=20.0),)
        if call == (0, 1):
            return (
                ServiceQoTUpdate(service_id=0, osnr=5.0, ase=30.0, nli=20.0),
                ServiceQoTUpdate(service_id=1, osnr=25.0, ase=30.0, nli=20.0),
            )
        if call == (1, 2):
            return (
                ServiceQoTUpdate(service_id=1, osnr=5.0, ase=30.0, nli=20.0),
                ServiceQoTUpdate(service_id=2, osnr=25.0, ase=30.0, nli=20.0),
            )
        if call == (2,):
            return (ServiceQoTUpdate(service_id=2, osnr=25.0, ase=30.0, nli=20.0),)
        raise AssertionError(f"unexpected refresh call {call!r}")

    monkeypatch.setattr(simulator.qot_engine, "evaluate_candidate", evaluate_candidate_high)
    monkeypatch.setattr(simulator.qot_engine, "refresh_services", refresh_services_cascading)

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=0,
    )
    assert terminated is False
    assert truncated is False

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=3,
    )
    assert terminated is False
    assert truncated is False

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=6,
    )

    assert terminated is True
    assert truncated is False
    assert refresh_calls == [(0,), (0, 1), (1, 2), (2,)]
    assert tuple(sorted(simulator.state.active_services_by_id)) == (2,)
    assert tuple(sorted(simulator.state.disrupted_services_by_id)) == (0, 1)
    assert info["blocked_due_to_osnr"] == 0
    assert info["disrupted_or_dropped_services"] == 2


def test_simulator_drop_on_disruption_drops_worst_margin_first_then_recomputes(monkeypatch) -> None:
    simulator = Simulator(_cascading_drop_config(), _topology(), episode_length=3)
    _, info = simulator.reset(seed=7)

    assert simulator.state is not None

    def evaluate_candidate_high(*_args, **_kwargs) -> QoTResult:
        return QoTResult(osnr=25.0, ase=30.0, nli=20.0, meets_threshold=True)

    refresh_calls: list[tuple[int, ...]] = []

    def refresh_services_one_by_one(state, service_ids):
        call = tuple(service_ids)
        refresh_calls.append(call)
        if call == (0,):
            return (ServiceQoTUpdate(service_id=0, osnr=25.0, ase=30.0, nli=20.0),)
        if call == (0, 1):
            return (
                ServiceQoTUpdate(service_id=0, osnr=4.0, ase=30.0, nli=20.0),
                ServiceQoTUpdate(service_id=1, osnr=5.5, ase=30.0, nli=20.0),
            )
        if call == (1, 2):
            return (
                ServiceQoTUpdate(service_id=1, osnr=25.0, ase=30.0, nli=20.0),
                ServiceQoTUpdate(service_id=2, osnr=25.0, ase=30.0, nli=20.0),
            )
        raise AssertionError(f"unexpected refresh call {call!r}")

    monkeypatch.setattr(simulator.qot_engine, "evaluate_candidate", evaluate_candidate_high)
    monkeypatch.setattr(simulator.qot_engine, "refresh_services", refresh_services_one_by_one)

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=0,
    )
    assert terminated is False
    assert truncated is False

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=3,
    )
    assert terminated is False
    assert truncated is False

    _, _, terminated, truncated, info = _accept_at_slot(
        simulator,
        info["mask"],
        path_index=0,
        modulation_offset=0,
        initial_slot=6,
    )

    assert terminated is True
    assert truncated is False
    assert refresh_calls == [(0,), (0, 1), (1, 2)]
    assert tuple(sorted(simulator.state.active_services_by_id)) == (1, 2)
    assert tuple(sorted(simulator.state.disrupted_services_by_id)) == (0,)
    assert info["disrupted_or_dropped_services"] == 1


def test_simulator_reject_action_records_rejection_and_negative_reward() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=1)
    simulator.reset(seed=7)

    observation, reward, terminated, truncated, info = simulator.step(simulator.total_actions - 1)

    assert terminated is True
    assert truncated is False
    assert info["status"] == Status.REJECTED_BY_AGENT.value
    assert reward < 0.0
    assert info["reward_profile"] == "balanced"
    assert info["episode_services_accepted"] == 0
    assert info["rejected"] == 1
    assert np.count_nonzero(observation) == 0


def test_simulator_only_episode_counters_reset_preserves_current_request() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=3)
    simulator.reset(seed=7)
    action = _first_valid_action(simulator.action_masks())
    simulator.step(action)

    observation, info = simulator.reset(options={"only_episode_counters": True})

    assert observation.shape == (simulator.observation_builder.schema.total_size,)
    assert info["mask"].shape == (simulator.total_actions,)
    assert simulator.current_request is not None


def test_simulator_accepts_reverse_direction_request() -> None:
    simulator = Simulator(_reverse_config(), _topology(), episode_length=1)
    _, info = simulator.reset(seed=7)

    _, reward, terminated, truncated, info = simulator.step(_first_valid_action(info["mask"]))

    assert terminated is True
    assert truncated is False
    assert reward > 0.0
    assert info["status"] == Status.ACCEPTED.value
    assert info["episode_services_accepted"] == 1
