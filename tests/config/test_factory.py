from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2 import OpticalEnv, ScenarioConfig, get_modulations, make_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_ROOT / "src" / "optical_networking_gym_v2" / "topologies"


def test_make_env_builds_optical_env_with_dynamic_defaults() -> None:
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        topology_dir=TOPOLOGY_DIR,
        seed=7,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=4,
        modulations_to_consider=2,
        k_paths=2,
    )

    assert isinstance(env, OpticalEnv)
    assert env.simulator.config.topology_id == "ring_4"
    assert env.simulator.config.seed == 7
    assert env.simulator.config.traffic_source["load"] == 10.0
    assert tuple(modulation.name for modulation in env.simulator.config.modulations) == ("QPSK", "16QAM")


def test_make_env_maps_output_flags_into_config() -> None:
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        topology_dir=TOPOLOGY_DIR,
        seed=9,
        num_spectrum_resources=24,
        episode_length=4,
        modulations_to_consider=2,
        k_paths=2,
        enable_observation=False,
        enable_action_mask=False,
        include_mask_in_info=True,
        capture_step_trace=True,
        drop_on_disruption=True,
    )

    assert env.simulator.config.enable_observation is False
    assert env.simulator.config.enable_action_mask is False
    assert env.simulator.config.include_mask_in_info is False
    assert env.simulator.config.capture_step_trace is True
    assert env.simulator.config.drop_on_disruption is True


def test_make_env_accepts_explicit_config() -> None:
    config = ScenarioConfig(
        scenario_id="factory_config_path",
        topology_id="ring_4",
        topology_dir=TOPOLOGY_DIR,
        k_paths=2,
        num_spectrum_resources=24,
        episode_length=4,
        modulations=get_modulations("QPSK,16QAM"),
        modulations_to_consider=2,
        seed=11,
        enable_action_mask=True,
        include_mask_in_info=False,
    )

    env = make_env(config=config)

    assert isinstance(env, OpticalEnv)
    assert env.simulator.config == config
    assert env.simulator.config.include_mask_in_info is False


def test_make_env_accepts_scenario_name() -> None:
    env = make_env(scenario="ring4_quickstart", episode_length=2)

    assert isinstance(env, OpticalEnv)
    assert env.simulator.config.scenario_id == "ring4_quickstart"
    assert env.simulator.config.topology_id == "ring_4"
    assert env.simulator.config.episode_length == 2


def test_make_env_scenario_overrides_do_not_apply_flat_defaults() -> None:
    env = make_env(scenario="nobel_eu_legacy_benchmark")

    assert env.simulator.config.load == 300.0
    assert env.simulator.config.launch_power_dbm == 2.0
    assert env.simulator.config.modulations_to_consider == 6


def test_make_env_accepts_scenario_config_alias() -> None:
    config = ScenarioConfig(
        scenario_id="factory_scenario_alias",
        topology_id="ring_4",
        topology_dir=TOPOLOGY_DIR,
        k_paths=2,
        num_spectrum_resources=24,
        episode_length=4,
        modulations=get_modulations("QPSK,16QAM"),
        modulations_to_consider=2,
        seed=11,
    )

    env = make_env(scenario=config)

    assert env.simulator.config == config


def test_make_env_rejects_ambiguous_inputs() -> None:
    config = ScenarioConfig(
        scenario_id="factory_ambiguous",
        topology_id="ring_4",
        topology_dir=TOPOLOGY_DIR,
        k_paths=2,
        num_spectrum_resources=24,
        episode_length=4,
        modulations=get_modulations("QPSK,16QAM"),
        modulations_to_consider=2,
    )

    import pytest

    with pytest.raises(ValueError, match="config cannot be combined with scenario"):
        make_env(config=config, scenario="ring4_quickstart")

    with pytest.raises(ValueError, match="topology_name cannot be combined with scenario"):
        make_env("ring_4", scenario="ring4_quickstart")
