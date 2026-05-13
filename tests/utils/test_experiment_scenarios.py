from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import (
    BUILTIN_TOPOLOGY_DIR,
    ScenarioConfig,
    build_scenario,
    build_scenario_grid,
    list_scenarios,
)
from optical_networking_gym_v2.utils.experiment_scenarios import (
    build_legacy_benchmark_scenario,
    build_nobel_eu_graph_load_scenario,
    build_nobel_eu_ofc_v1_scenario,
)
from optical_networking_gym_v2.utils.experiment_utils import DEFAULT_BIT_RATES, SimulationUtils


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_list_scenarios_returns_stable_names() -> None:
    assert list_scenarios() == (
        "ring4_quickstart",
        "nobel_eu_baseline",
        "nobel_eu_legacy_benchmark",
        "nobel_eu_publication",
        "nobel_eu_disruptions",
        "jocn_benchmark",
    )


def test_build_scenario_returns_typed_config() -> None:
    scenario = build_scenario("ring4_quickstart")

    assert isinstance(scenario, ScenarioConfig)
    assert scenario.topology_id == "ring_4"
    assert scenario.topology_dir == BUILTIN_TOPOLOGY_DIR


def test_build_scenario_applies_overrides() -> None:
    scenario = build_scenario("nobel_eu_baseline", load=400, margin=2.0, seed=10)

    assert scenario.load == pytest.approx(400.0)
    assert scenario.margin == pytest.approx(2.0)
    assert scenario.seed == 10


def test_build_scenario_accepts_modulation_names() -> None:
    scenario = build_scenario("nobel_eu_baseline", modulations="BPSK,QPSK,16QAM")

    assert tuple(modulation.name for modulation in scenario.modulations) == (
        "BPSK",
        "QPSK",
        "16QAM",
    )
    assert scenario.modulations_to_consider == 3


def test_build_scenario_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Available scenarios"):
        build_scenario("missing")


def test_build_scenario_grid_expands_axes() -> None:
    scenarios = build_scenario_grid(
        "nobel_eu_baseline",
        axes={
            "load": (100, 150),
            "margin": (0.0, 1.0),
            "topology_id": ("ring_4",),
            "qot_constraint": ("DIST", "ASE+NLI"),
        },
    )

    assert len(scenarios) == 8
    assert {scenario.load for scenario in scenarios} == {100.0, 150.0}
    assert {scenario.margin for scenario in scenarios} == {0.0, 1.0}
    assert {scenario.topology_id for scenario in scenarios} == {"ring_4"}
    assert {scenario.qot_constraint for scenario in scenarios} == {"DIST", "ASE+NLI"}


def test_simulation_utils_expands_inclusive_range_values() -> None:
    assert SimulationUtils.normalize_values((100, 400, 100)) == (100.0, 200.0, 300.0, 400.0)
    assert SimulationUtils.normalize_values(210) == (210.0,)


def test_simulation_utils_create_environment_matches_jocn_setup() -> None:
    scenarios = SimulationUtils.create_environment(
        topology_name="nobel-eu",
        modulation_names="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
        seed=42,
        bit_rates=DEFAULT_BIT_RATES,
        load=(100, 400, 100),
        num_spectrum_resources=320,
        episode_length=100,
        modulations_to_consider=3,
        defragmentation=False,
        k_paths=3,
        gen_observation=True,
    )

    assert tuple(scenario.load for scenario in scenarios) == (100.0, 200.0, 300.0, 400.0)
    assert scenarios[0].topology_id == "nobel-eu"
    assert scenarios[0].topology_dir == BUILTIN_TOPOLOGY_DIR
    assert scenarios[0].bit_rates == (10, 40, 100, 400)
    assert scenarios[0].num_spectrum_resources == 320
    assert scenarios[0].episode_length == 100
    assert scenarios[0].modulations_to_consider == 3
    assert scenarios[0].k_paths == 3
    assert scenarios[0].enable_observation is True
    assert scenarios[0].seed == 42
    assert tuple(modulation.name for modulation in scenarios[0].modulations) == (
        "BPSK",
        "QPSK",
        "8QAM",
        "16QAM",
        "32QAM",
        "64QAM",
    )


def test_build_nobel_eu_graph_load_scenario_requires_episode_length() -> None:
    with pytest.raises(TypeError):
        build_nobel_eu_graph_load_scenario(
            PROJECT_ROOT.parent,
            seed=50,
            load=300.0,
            mean_holding_time=10800.0,
            num_spectrum_resources=320,
            k_paths=5,
            launch_power_dbm=1.0,
            modulations_to_consider=3,
        )


def test_build_nobel_eu_graph_load_scenario_returns_typed_config() -> None:
    scenario = build_nobel_eu_graph_load_scenario(
        PROJECT_ROOT.parent,
        topology_id="ring_4",
        episode_length=8,
        seed=7,
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        k_paths=2,
        launch_power_dbm=1.0,
        modulations_to_consider=2,
    )

    assert isinstance(scenario, ScenarioConfig)
    assert scenario.topology_id == "ring_4"
    assert scenario.episode_length == 8
    assert scenario.seed == 7
    assert scenario.topology_dir == BUILTIN_TOPOLOGY_DIR
    assert scenario.measure_disruptions is True
    assert scenario.drop_on_disruption is False


def test_build_nobel_eu_ofc_v1_scenario_uses_legacy_thresholds() -> None:
    scenario = build_nobel_eu_ofc_v1_scenario(seed=11)

    assert isinstance(scenario, ScenarioConfig)
    assert scenario.topology_id == "nobel-eu"
    assert scenario.episode_length == 1000
    assert scenario.seed == 11
    assert scenario.topology_dir == BUILTIN_TOPOLOGY_DIR
    assert scenario.max_span_length_km == pytest.approx(80.0)
    assert scenario.default_attenuation_db_per_km == pytest.approx(0.2)
    assert scenario.default_noise_figure_db == pytest.approx(4.5)
    assert scenario.bit_rates == (10, 40, 100, 400)
    assert scenario.measure_disruptions is False
    assert scenario.drop_on_disruption is False
    assert tuple(modulation.name for modulation in scenario.modulations) == (
        "BPSK",
        "QPSK",
        "8QAM",
        "16QAM",
        "32QAM",
        "64QAM",
    )
    assert tuple(modulation.minimum_osnr for modulation in scenario.modulations) == pytest.approx(
        (
            3.71925646843142,
            6.72955642507124,
            10.8453935345953,
            13.2406469649752,
            16.1608982942870,
            19.0134649345090,
        )
    )


def test_build_legacy_benchmark_scenario_matches_requested_profile() -> None:
    scenario = build_legacy_benchmark_scenario(seed=10)

    assert isinstance(scenario, ScenarioConfig)
    assert scenario.topology_id == "nobel-eu"
    assert scenario.episode_length == 1000
    assert scenario.seed == 10
    assert scenario.topology_dir == BUILTIN_TOPOLOGY_DIR
    assert scenario.load == 300.0
    assert scenario.mean_holding_time == 10800.0
    assert scenario.num_spectrum_resources == 320
    assert scenario.k_paths == 5
    assert scenario.launch_power_dbm == 2.0
    assert scenario.frequency_slot_bandwidth == 12.5e9
    assert scenario.frequency_start == 3e8 / 1565e-9
    assert scenario.bandwidth == 4e12
    assert scenario.bit_rates == (10, 40, 100, 400)
    assert scenario.modulations_to_consider == 6
    assert scenario.measure_disruptions is False
    assert scenario.drop_on_disruption is False
    assert tuple(modulation.name for modulation in scenario.modulations) == (
        "BPSK",
        "QPSK",
        "8QAM",
        "16QAM",
        "32QAM",
        "64QAM",
    )
    assert tuple(modulation.minimum_osnr for modulation in scenario.modulations) == pytest.approx(
        (
            3.71925646843142,
            6.72955642507124,
            10.8453935345953,
            13.2406469649752,
            16.1608982942870,
            19.0134649345090,
        )
    )


def test_jocn_benchmark_scenario_matches_article_profile() -> None:
    scenario = build_scenario("jocn_benchmark")

    assert scenario.scenario_id == "jocn_benchmark"
    assert scenario.topology_id == "nobel-eu"
    assert scenario.k_paths == 5
    assert scenario.num_spectrum_resources == 320
    assert scenario.episode_length == 1000
    assert scenario.max_span_length_km == pytest.approx(80.0)
    assert scenario.default_attenuation_db_per_km == pytest.approx(0.2)
    assert scenario.default_noise_figure_db == pytest.approx(4.5)
    assert scenario.bit_rates == (10, 40, 100, 400)
    assert scenario.mean_holding_time == pytest.approx(10_800.0)
    assert scenario.frequency_slot_bandwidth == pytest.approx(12.5e9)
    assert scenario.bandwidth == pytest.approx(4e12)
    assert scenario.modulations_to_consider == 6
    assert tuple(modulation.minimum_osnr for modulation in scenario.modulations) == pytest.approx(
        (3.71, 6.72, 10.84, 13.24, 16.16, 19.01)
    )
