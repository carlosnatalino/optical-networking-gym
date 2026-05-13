from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from optical_networking_gym_v2 import MaskMode, Modulation, ScenarioConfig, TrafficMode


def test_scenario_config_defaults_follow_architecture() -> None:
    config = ScenarioConfig(
        scenario_id="smoke_ring_default",
        topology_id="ring_4",
        k_paths=3,
        num_spectrum_resources=80,
    )

    assert config.mask_mode is MaskMode.RESOURCE_AND_QOT
    assert config.traffic_mode is TrafficMode.DYNAMIC
    assert config.qot_constraint == "ASE+NLI"
    assert config.measure_disruptions is False
    assert config.drop_on_disruption is False
    assert config.channel_width == pytest.approx(12.5)
    assert config.frequency_slot_bandwidth == pytest.approx(12.5e9)
    assert config.launch_power_dbm == pytest.approx(0.0)
    assert config.margin == pytest.approx(0.0)
    assert config.bandwidth == pytest.approx(80 * 12.5e9)
    assert config.episode_length == 1_000
    assert config.enable_observation is True
    assert config.enable_action_mask is True
    assert config.include_mask_in_info is True
    assert config.traffic_source["bit_rates"] == (10, 40, 100, 400)
    assert config.traffic_source["load"] == pytest.approx(300.0)
    assert config.traffic_source["mean_holding_time"] == pytest.approx(100.0)


def test_scenario_config_is_immutable() -> None:
    config = ScenarioConfig(
        scenario_id="smoke_ring_default",
        topology_id="ring_4",
        k_paths=3,
        num_spectrum_resources=80,
    )

    with pytest.raises(FrozenInstanceError):
        config.k_paths = 5


def test_static_traffic_requires_source() -> None:
    with pytest.raises(ValueError, match="traffic_source"):
        ScenarioConfig(
            scenario_id="static_fixture",
            topology_id="ring_4",
            k_paths=2,
            num_spectrum_resources=24,
            traffic_mode=TrafficMode.STATIC,
        )


def test_static_traffic_path_populates_static_source() -> None:
    config = ScenarioConfig(
        scenario_id="static_fixture_path",
        topology_id="ring_4",
        topology_dir="examples/topologies",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        static_traffic_path="traffic/table.jsonl",
    )

    assert config.topology_dir == Path("examples/topologies")
    assert config.static_traffic_path == Path("traffic/table.jsonl")
    assert config.traffic_source == Path("traffic/table.jsonl")


def test_invalid_qot_constraint_fails_fast() -> None:
    with pytest.raises(ValueError, match="qot_constraint"):
        ScenarioConfig(
            scenario_id="invalid_qot",
            topology_id="ring_4",
            k_paths=2,
            num_spectrum_resources=24,
            qot_constraint="INVALID",
        )


def test_scenario_config_clamps_modulations_to_consider_to_catalog_size() -> None:
    config = ScenarioConfig(
        scenario_id="mod_catalog",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2),
            Modulation("16QAM", 500.0, 4),
        ),
        modulations_to_consider=10,
    )

    assert len(config.modulations) == 2
    assert config.modulations_to_consider == 2


def test_disabling_action_mask_forces_mask_info_off() -> None:
    config = ScenarioConfig(
        scenario_id="mask_disabled",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        enable_action_mask=False,
        include_mask_in_info=True,
    )

    assert config.enable_action_mask is False
    assert config.include_mask_in_info is False
