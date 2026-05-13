from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from optical_networking_gym_v2 import (
    ActionMask,
    MaskMode,
    Modulation,
    QoTEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def _config(*, mask_mode: MaskMode = MaskMode.RESOURCE_AND_QOT, margin: float = 0.0) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=f"action_mask_{mask_mode.value}",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        mask_mode=mask_mode,
        margin=margin,
        modulations=_modulations(),
        modulations_to_consider=2,
    )


def _request(service_id: int = 0) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=2,
        bit_rate=40,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def test_action_mask_reject_action_is_always_valid() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    builder = ActionMask(config, topology, QoTEngine(config, topology))

    mask = builder.build(state, _request(5))

    assert mask.shape == (97,)
    assert mask.dtype == np.uint8
    assert mask[-1] == 1


def test_action_mask_resource_only_keeps_resource_valid_slots_when_qot_would_fail() -> None:
    topology = _topology()
    resource_only_config = _config(mask_mode=MaskMode.RESOURCE_ONLY, margin=100.0)
    qot_config = _config(mask_mode=MaskMode.RESOURCE_AND_QOT, margin=100.0)
    state_resource = RuntimeState(resource_only_config, topology)
    state_qot = RuntimeState(qot_config, topology)

    resource_builder = ActionMask(
        resource_only_config,
        topology,
        QoTEngine(resource_only_config, topology),
    )
    qot_builder = ActionMask(qot_config, topology, QoTEngine(qot_config, topology))

    resource_mask = resource_builder.build(state_resource, _request(7))
    qot_mask = qot_builder.build(state_qot, _request(7))

    assert resource_mask[:-1].sum() > 0
    assert qot_mask[:-1].sum() == 0
    assert resource_mask[-1] == 1
    assert qot_mask[-1] == 1


def test_action_mask_decode_uses_same_dynamic_modulation_subset_used_by_mask() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    builder = ActionMask(config, topology, QoTEngine(config, topology))
    request = _request(9)

    mask = builder.build(state, request)
    valid_action = int(np.flatnonzero(mask[:-1])[0])
    decoded = builder.decode_action(valid_action, state, request)

    assert decoded.path_index == 0
    assert decoded.modulation_index in (0, 1)
    assert decoded.initial_slot >= 0
    assert mask[valid_action] == 1


def test_action_mask_marks_busy_ranges_invalid() -> None:
    topology = _topology()
    config = _config()
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    for link_id in path.link_ids:
        state.slot_allocation[link_id, 0:4] = 99
    builder = ActionMask(config, topology, QoTEngine(config, topology))

    mask = builder.build(state, _request(11))
    first_path_qpsk_slice = mask[:24]

    assert first_path_qpsk_slice[:4].sum() == 0


def test_action_mask_returns_none_when_output_is_disabled() -> None:
    topology = _topology()
    config = replace(_config(), enable_action_mask=False)
    state = RuntimeState(config, topology)
    builder = ActionMask(config, topology, QoTEngine(config, topology))

    mask = builder.build(state, _request(12))

    assert mask is None
