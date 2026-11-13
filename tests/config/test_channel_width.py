from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from optical_networking_gym import (
    Modulation,
    QoTEngine,
    RequestAnalysisEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
    build_scenario,
    list_scenarios,
)
from optical_networking_gym.utils.experiment_scenarios import (
    build_legacy_benchmark_scenario,
    build_nobel_eu_graph_load_scenario,
    build_nobel_eu_ofc_v1_scenario,
)
from optical_networking_gym.utils.experiment_utils import SimulationUtils


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym" / "topologies" / "ring_4.txt"
MODULATIONS = (
    Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
    Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
)


def _config(**overrides: object) -> ScenarioConfig:
    fields: dict[str, object] = {
        "scenario_id": "channel_width",
        "topology_id": "ring_4",
        "k_paths": 2,
        "num_spectrum_resources": 320,
        "modulations": MODULATIONS,
        "modulations_to_consider": 2,
    }
    fields.update(overrides)
    return ScenarioConfig(**fields)  # type: ignore[arg-type]


def _assert_consistent(config: ScenarioConfig) -> None:
    assert config.channel_width is not None
    assert config.bandwidth is not None
    assert math.isclose(config.channel_width * 1e9, config.frequency_slot_bandwidth, rel_tol=1e-9)
    assert math.isclose(
        config.bandwidth,
        config.num_spectrum_resources * config.frequency_slot_bandwidth,
        rel_tol=1e-9,
    )


def test_defaults_are_derived_from_the_slot_grid() -> None:
    config = _config()

    assert config.channel_width == 12.5
    assert config.bandwidth == 4e12
    assert config.resolved_channel_width == 12.5


def test_consistent_explicit_values_are_accepted() -> None:
    config = _config(channel_width=12.5, bandwidth=4e12)

    assert config.channel_width == 12.5
    assert config.bandwidth == 4e12
    assert config.runtime_structure_key() == _config().runtime_structure_key()


def test_narrow_slots_drive_slot_counts_and_qot_bandwidth() -> None:
    config = _config(frequency_slot_bandwidth=6.25e9, num_spectrum_resources=64)
    assert config.channel_width == 6.25
    assert config.bandwidth == 64 * 6.25e9

    topology = TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)
    qot_engine = QoTEngine(config, topology)
    engine = RequestAnalysisEngine(config, topology, qot_engine)
    request = ServiceRequest(
        request_index=0,
        service_id=0,
        source_id=0,
        destination_id=1,
        bit_rate=100,
        arrival_time=1.0,
        holding_time=10.0,
    )
    analysis = engine.build(RuntimeState(config, topology), request)

    offset_16qam = analysis.modulation_offset_for_index(1)
    assert offset_16qam is not None
    # ceil(100 / (4 x 6.25)) = 4 slots of 6.25 GHz, not 2 slots of 12.5 GHz.
    assert analysis.required_slots_by_path_mod[0, offset_16qam] == 4
    candidate = qot_engine.build_candidate(
        request=request,
        path=analysis.paths[0],
        modulation=MODULATIONS[1],
        service_slot_start=0,
        service_num_slots=4,
    )
    assert candidate.bandwidth == 4 * 6.25e9


def test_inconsistent_channel_width_raises() -> None:
    with pytest.raises(ValueError, match=r"channel_width \(GHz\) must equal frequency_slot_bandwidth"):
        _config(channel_width=12.5, frequency_slot_bandwidth=6.25e9)


def test_inconsistent_bandwidth_raises() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        _config(bandwidth=4e12, num_spectrum_resources=160)


def test_replace_of_the_grid_requires_rederiving() -> None:
    # ``replace`` carries the resolved values over, so changing the grid with it
    # is caught instead of silently keeping a stale width or bandwidth.
    with pytest.raises(ValueError, match="channel_width"):
        replace(_config(), frequency_slot_bandwidth=6.25e9)
    narrow = replace(_config(), frequency_slot_bandwidth=6.25e9, channel_width=None, bandwidth=None)
    assert narrow.channel_width == 6.25
    assert narrow.bandwidth == 320 * 6.25e9


@pytest.mark.parametrize("name", list_scenarios())
def test_every_preset_is_consistent(name: str) -> None:
    _assert_consistent(build_scenario(name))


@pytest.mark.parametrize("name", list_scenarios())
def test_preset_grid_overrides_rederive_the_width_and_bandwidth(name: str) -> None:
    fewer_slots = build_scenario(name, num_spectrum_resources=160)
    _assert_consistent(fewer_slots)
    assert fewer_slots.bandwidth == 160 * fewer_slots.frequency_slot_bandwidth

    narrow = build_scenario(name, frequency_slot_bandwidth=6.25e9)
    _assert_consistent(narrow)
    assert narrow.channel_width == 6.25

    with pytest.raises(ValueError, match="channel_width"):
        build_scenario(name, frequency_slot_bandwidth=6.25e9, channel_width=12.5)


@pytest.mark.parametrize("num_spectrum_resources", [160, 320])
def test_experiment_scenario_builders_are_consistent(num_spectrum_resources: int) -> None:
    configs = [
        build_legacy_benchmark_scenario(num_spectrum_resources=num_spectrum_resources),
        build_nobel_eu_ofc_v1_scenario(num_spectrum_resources=num_spectrum_resources),
        build_nobel_eu_graph_load_scenario(
            episode_length=10,
            load=100.0,
            num_spectrum_resources=num_spectrum_resources,
            k_paths=3,
            launch_power_dbm=0.0,
            modulations_to_consider=3,
        ),
        *SimulationUtils.create_environment(num_spectrum_resources=num_spectrum_resources, load=(100.0, 200.0)),
    ]
    for config in configs:
        _assert_consistent(config)
        assert config.num_spectrum_resources == num_spectrum_resources
