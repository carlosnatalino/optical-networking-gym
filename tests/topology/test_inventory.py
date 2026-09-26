"""Equipment library (GNPy format) and per-span network inventory."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym import QoTEngine, RuntimeState, ScenarioConfig, ServiceRequest
from optical_networking_gym.contracts import Modulation
from optical_networking_gym.defaults import BUILTIN_TOPOLOGY_DIR
from optical_networking_gym.network import (
    AmplifierRecord,
    EquipmentLibrary,
    GainRipple,
    LinkRecord,
    NetworkInventory,
    SpanRecord,
    TopologyModel,
    apply_inventory,
)

GNPY_EQPT = Path(__file__).resolve().parents[1] / "data" / "gnpy" / "eqpt_config.json"
RING_4 = BUILTIN_TOPOLOGY_DIR / "ring_4.txt"


@pytest.fixture(scope="module")
def equipment() -> EquipmentLibrary:
    return EquipmentLibrary.from_json(GNPY_EQPT)


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4, topology_id="ring_4", k_paths=2, max_span_length_km=80.0)


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="inventory_ring_4",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=320,
    )


def _osnr_by_slot(topology: TopologyModel, slots: list[int]) -> np.ndarray:
    config = _config()
    engine = QoTEngine(config, topology)
    state = RuntimeState(config, topology)
    path = topology.get_paths("1", "3")[0]
    modulation = Modulation("QPSK", 100_000.0, 2, minimum_osnr=6.72)
    request = ServiceRequest(
        request_index=0,
        service_id=0,
        source_id=0,
        destination_id=2,
        bit_rate=100,
        arrival_time=0.0,
        holding_time=1.0,
    )
    return np.array(
        [
            engine.evaluate_candidate(
                state, engine.build_candidate(request, path, modulation, slot, 4)
            ).osnr
            for slot in slots
        ]
    )


def test_gnpy_equipment_file_is_parsed(equipment: EquipmentLibrary) -> None:
    assert {"std_medium_gain", "std_fixed_gain", "high_detail_model_example"} <= set(
        equipment.amplifiers
    )
    assert equipment.roadm_add_drop_osnr_db == pytest.approx(38.0)
    assert equipment.transceiver_tx_osnr_db == pytest.approx(40.0)
    ripple = equipment.amplifier("high_detail_model_example").gain_ripple
    assert ripple is not None and len(ripple.values_db) == 96


def test_fixed_gain_nf_is_constant(equipment: EquipmentLibrary) -> None:
    amplifier = equipment.amplifier("std_fixed_gain")
    assert amplifier.noise_figure_db(20.5) == pytest.approx(5.5)


def test_variable_gain_two_coil_model_hits_datasheet_corners(equipment: EquipmentLibrary) -> None:
    amplifier = equipment.amplifier("std_medium_gain")  # gain 15..26 dB, NF 6..10 dB
    assert amplifier.noise_figure_db(26.0) == pytest.approx(6.0, abs=0.01)
    assert amplifier.noise_figure_db(15.0) == pytest.approx(10.0, abs=0.01)
    # NF decreases monotonically with gain in between.
    gains = np.linspace(15.0, 26.0, 12)
    nfs = [amplifier.noise_figure_db(g) for g in gains]
    assert all(b < a for a, b in zip(nfs, nfs[1:]))
    # Below gain_min an input pad is used and its loss adds to the NF.
    assert amplifier.noise_figure_db(13.0) == pytest.approx(amplifier.noise_figure_db(15.0) + 2.0)


def test_unsupported_nf_model_raises(equipment: EquipmentLibrary) -> None:
    with pytest.raises(ValueError):
        equipment.amplifier("openroadm_ila_standard").noise_figure_db(20.0)


def test_gain_ripple_transform_and_interpolation() -> None:
    ripple = GainRipple.uniform(191e12, 196e12, [0.0, 1.0, 0.0])
    assert ripple.at(np.array([193.5e12]))[0] == pytest.approx(1.0)
    shifted = ripple.transformed(scale=2.0, shift_hz=0.5e12)
    assert shifted.at(np.array([194.0e12]))[0] == pytest.approx(2.0)
    assert ripple.at(np.array([150e12]))[0] == pytest.approx(0.0)  # edge value
    with pytest.raises(ValueError):
        GainRipple((1.0, 1.0), (0.0, 0.0))


def test_inventory_json_roundtrip(tmp_path: Path) -> None:
    inventory = NetworkInventory.from_topology(_topology(), noise_figure_db=5.0, con_in_db=0.5)
    path = inventory.to_json(tmp_path / "inventory.json")
    assert NetworkInventory.from_json(path) == inventory


def test_nominal_inventory_reproduces_the_plain_topology() -> None:
    topology = _topology()
    nominal = apply_inventory(topology, NetworkInventory.from_topology(topology))
    slots = [10, 100, 200]
    np.testing.assert_array_equal(_osnr_by_slot(nominal, slots), _osnr_by_slot(topology, slots))


def test_connector_losses_lower_osnr() -> None:
    topology = _topology()
    lossy = apply_inventory(topology, NetworkInventory.from_topology(topology, con_in_db=1.0))
    assert np.all(_osnr_by_slot(lossy, [100]) < _osnr_by_slot(topology, [100]))


def test_nf_is_computed_from_equipment_at_span_loss(equipment: EquipmentLibrary) -> None:
    topology = _topology()
    inventory = NetworkInventory.from_topology(topology, amplifier_type="std_medium_gain")
    resolved = apply_inventory(topology, inventory, equipment)
    span = resolved.links[0].spans[0]
    expected = equipment.amplifier("std_medium_gain").noise_figure_db(span.total_loss_db)
    assert span.noise_figure_db == pytest.approx(expected)
    assert span.amplifier_type == "std_medium_gain"
    with pytest.raises(ValueError):
        apply_inventory(topology, inventory)  # type needs an equipment library


def test_gain_ripple_makes_osnr_channel_dependent(equipment: EquipmentLibrary) -> None:
    topology = _topology()
    flat = apply_inventory(
        topology,
        NetworkInventory.from_topology(
            topology, amplifier_type="high_detail_model_example", noise_figure_db=5.5
        ),
        equipment,
    )
    rippled = apply_inventory(
        topology,
        NetworkInventory.from_topology(
            topology,
            amplifier_type="high_detail_model_example",
            noise_figure_db=5.5,
            gain_ripple_scale=3.0,
        ),
        equipment,
    )
    assert all(span.gain_ripple is None for link in flat.links for span in link.spans)
    slots = list(range(0, 300, 25))
    difference = _osnr_by_slot(rippled, slots) - _osnr_by_slot(flat, slots)
    assert np.ptp(difference) > 0.05  # the penalty varies across the band


def test_batch_and_scalar_qot_agree_on_heterogeneous_inventory(equipment: EquipmentLibrary) -> None:
    topology = _topology()
    inventory = NetworkInventory.from_topology(
        topology,
        amplifier_type="high_detail_model_example",
        noise_figure_db=5.5,
        con_in_db=0.7,
        con_out_db=0.3,
        gain_ripple_scale=2.0,
    )
    resolved = apply_inventory(topology, inventory, equipment)
    config = _config()
    engine = QoTEngine(config, resolved)
    state = RuntimeState(config, resolved)
    path = resolved.get_paths("1", "3")[0]
    modulation = Modulation("QPSK", 100_000.0, 2, minimum_osnr=6.72)
    starts = [0, 57, 123, 250]
    batch = engine.summarize_candidate_starts(
        state=state,
        service_id=0,
        path=path,
        modulation=modulation,
        service_num_slots=4,
        candidate_starts=starts,
    )
    scalar = [
        engine.summarize_candidate_at(
            state=state,
            service_id=0,
            path=path,
            modulation=modulation,
            service_slot_start=start,
            service_num_slots=4,
        ).osnr_margin
        for start in starts
    ]
    np.testing.assert_allclose(batch.osnr_margin, scalar, rtol=1e-10)


def test_inventory_link_errors() -> None:
    topology = _topology()
    inventory = NetworkInventory.from_topology(topology)
    with pytest.raises(ValueError, match="misses"):
        apply_inventory(topology, replace(inventory, links=inventory.links[1:]))
    with pytest.raises(ValueError, match="twice"):
        apply_inventory(topology, replace(inventory, links=inventory.links + inventory.links[:1]))
    bogus = LinkRecord("1", "99", (SpanRecord(10.0, 0.2, amplifier=AmplifierRecord(noise_figure_db=5.0)),))
    with pytest.raises((ValueError, KeyError)):
        apply_inventory(topology, replace(inventory, links=inventory.links + (bogus,)))
