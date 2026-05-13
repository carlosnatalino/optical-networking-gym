from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2 import (
    BUILTIN_TOPOLOGY_DIR,
    Modulation,
    ScenarioConfig,
    build_scenario,
    get_modulations,
)

from .experiment_utils import DEFAULT_MODULATION_NAMES


LEGACY_BENCHMARK_LOAD = 300.0
LEGACY_BENCHMARK_K_PATHS = 5
LEGACY_BENCHMARK_LAUNCH_POWER_DBM = 2.0
LEGACY_BENCHMARK_MODULATIONS_TO_CONSIDER = 6


def _build_legacy_qrmsa_modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("BPSK", 100_000.0, 1, minimum_osnr=3.71925646843142, inband_xt=-14.0),
        Modulation("QPSK", 2_000.0, 2, minimum_osnr=6.72955642507124, inband_xt=-17.0),
        Modulation("8QAM", 1_000.0, 3, minimum_osnr=10.8453935345953, inband_xt=-20.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.2406469649752, inband_xt=-23.0),
        Modulation("32QAM", 250.0, 5, minimum_osnr=16.1608982942870, inband_xt=-26.0),
        Modulation("64QAM", 125.0, 6, minimum_osnr=19.0134649345090, inband_xt=-29.0),
    )


def _build_legacy_qrmsa_scenario(
    *,
    episode_length: int,
    seed: int,
    load: float,
    mean_holding_time: float,
    num_spectrum_resources: int,
    k_paths: int,
    launch_power_dbm: float,
    modulations_to_consider: int,
    margin: float = 0.0,
    measure_disruptions: bool = False,
    drop_on_disruption: bool = False,
    topology_id: str,
    scenario_suffix: str,
) -> ScenarioConfig:
    return build_scenario(
        "nobel_eu_legacy_benchmark",
        scenario_id=f"{topology_id}_{scenario_suffix}_seed{seed}",
        topology_id=topology_id,
        episode_length=episode_length,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
        num_spectrum_resources=num_spectrum_resources,
        k_paths=k_paths,
        launch_power_dbm=launch_power_dbm,
        margin=margin,
        measure_disruptions=measure_disruptions,
        drop_on_disruption=drop_on_disruption,
        modulations=_build_legacy_qrmsa_modulations(),
        modulations_to_consider=modulations_to_consider,
    )


def build_nobel_eu_ofc_v1_scenario(
    *,
    episode_length: int = 1_000,
    seed: int = 10,
    load: float = 400.0,
    mean_holding_time: float = 10_800.0,
    num_spectrum_resources: int = 320,
    k_paths: int = 3,
    launch_power_dbm: float = 0.0,
    modulations_to_consider: int = 6,
    margin: float = 0.0,
    measure_disruptions: bool = False,
    drop_on_disruption: bool = False,
    topology_id: str = "nobel-eu",
) -> ScenarioConfig:
    return _build_legacy_qrmsa_scenario(
        episode_length=episode_length,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
        num_spectrum_resources=num_spectrum_resources,
        k_paths=k_paths,
        launch_power_dbm=launch_power_dbm,
        modulations_to_consider=modulations_to_consider,
        margin=margin,
        measure_disruptions=measure_disruptions,
        drop_on_disruption=drop_on_disruption,
        topology_id=topology_id,
        scenario_suffix="ofc_v1",
    )


def build_legacy_benchmark_scenario(
    *,
    episode_length: int = 1_000,
    seed: int = 10,
    load: float = LEGACY_BENCHMARK_LOAD,
    mean_holding_time: float = 10_800.0,
    num_spectrum_resources: int = 320,
    k_paths: int = LEGACY_BENCHMARK_K_PATHS,
    launch_power_dbm: float = LEGACY_BENCHMARK_LAUNCH_POWER_DBM,
    modulations_to_consider: int = LEGACY_BENCHMARK_MODULATIONS_TO_CONSIDER,
    margin: float = 0.0,
    measure_disruptions: bool = False,
    drop_on_disruption: bool = False,
    topology_id: str = "nobel-eu",
) -> ScenarioConfig:
    return _build_legacy_qrmsa_scenario(
        episode_length=episode_length,
        seed=seed,
        load=load,
        mean_holding_time=mean_holding_time,
        num_spectrum_resources=num_spectrum_resources,
        k_paths=k_paths,
        launch_power_dbm=launch_power_dbm,
        modulations_to_consider=modulations_to_consider,
        margin=margin,
        measure_disruptions=measure_disruptions,
        drop_on_disruption=drop_on_disruption,
        topology_id=topology_id,
        scenario_suffix="legacy_benchmark",
    )


def build_nobel_eu_graph_load_scenario(
    repo_root: Path | None = None,
    *,
    episode_length: int,
    seed: int = 50,
    load: float,
    mean_holding_time: float = 10800.0,
    num_spectrum_resources: int,
    k_paths: int,
    launch_power_dbm: float,
    modulations_to_consider: int,
    margin: float = 0.0,
    measure_disruptions: bool = True,
    drop_on_disruption: bool = False,
    topology_id: str = "nobel-eu",
) -> ScenarioConfig:
    topology_dir = BUILTIN_TOPOLOGY_DIR
    return ScenarioConfig(
        scenario_id=f"{topology_id}_graph_load_seed{seed}",
        topology_id=topology_id,
        topology_dir=topology_dir,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        episode_length=episode_length,
        max_span_length_km=80.0,
        default_attenuation_db_per_km=0.2,
        default_noise_figure_db=4.5,
        bit_rates=(10, 40, 100, 400),
        load=load,
        mean_holding_time=mean_holding_time,
        qot_constraint="ASE+NLI",
        measure_disruptions=measure_disruptions,
        drop_on_disruption=drop_on_disruption,
        frequency_start=(3e8 / 1565e-9),
        frequency_slot_bandwidth=12.5e9,
        launch_power_dbm=launch_power_dbm,
        margin=margin,
        bandwidth=4e12,
        modulations=get_modulations(DEFAULT_MODULATION_NAMES),
        modulations_to_consider=modulations_to_consider,
        seed=seed,
    )


__all__ = [
    "LEGACY_BENCHMARK_K_PATHS",
    "LEGACY_BENCHMARK_LAUNCH_POWER_DBM",
    "LEGACY_BENCHMARK_LOAD",
    "LEGACY_BENCHMARK_MODULATIONS_TO_CONSIDER",
    "build_legacy_benchmark_scenario",
    "build_nobel_eu_graph_load_scenario",
    "build_nobel_eu_ofc_v1_scenario",
]
