from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation,
)
from optical_networking_gym.topology import Modulation as LegacyModulation
from optical_networking_gym.topology import get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym_v2.contracts.enums import TrafficMode
from optical_networking_gym_v2.envs.optical_env import OpticalEnv
from optical_networking_gym_v2.heuristics import select_first_fit_action
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.utils import build_nobel_eu_ofc_v1_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_DIR = PROJECT_ROOT / "examples" / "topologies"
NOBEL_EU_PATH = TOPOLOGY_DIR / "nobel-eu.xml"
REQUEST_COUNT = 250
SEED = 10


def _legacy_modulations() -> tuple[LegacyModulation, ...]:
    return (
        LegacyModulation("BPSK", 100_000, 1, minimum_osnr=3.71925646843142, inband_xt=-14),
        LegacyModulation("QPSK", 2_000, 2, minimum_osnr=6.72955642507124, inband_xt=-17),
        LegacyModulation("8QAM", 1_000, 3, minimum_osnr=10.8453935345953, inband_xt=-20),
        LegacyModulation("16QAM", 500, 4, minimum_osnr=13.2406469649752, inband_xt=-23),
        LegacyModulation("32QAM", 250, 5, minimum_osnr=16.1608982942870, inband_xt=-26),
        LegacyModulation("64QAM", 125, 6, minimum_osnr=19.0134649345090, inband_xt=-29),
    )


def _build_legacy_env() -> QRMSAEnvWrapper:
    topology = get_topology(
        str(NOBEL_EU_PATH),
        topology_name="nobel-eu",
        modulations=_legacy_modulations(),
        max_span_length=80.0,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=3,
    )
    return QRMSAEnvWrapper(
        topology=topology,
        seed=SEED,
        allow_rejection=True,
        load=300.0,
        episode_length=REQUEST_COUNT + 1,
        num_spectrum_resources=320,
        launch_power_dbm=0.0,
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8 / 1565e-9,
        bandwidth=320 * 12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(10, 40, 100, 400),
        margin=0.0,
        measure_disruptions=False,
        file_name="",
        k_paths=3,
        modulations_to_consider=6,
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=False,
        qot_constraint="ASE+NLI",
        rl_mode=True,
        capture_traffic_table=True,
        capture_step_trace=True,
    )


def _build_v2_env(*, traffic_path: Path) -> OpticalEnv:
    base_scenario = build_nobel_eu_ofc_v1_scenario(
        episode_length=REQUEST_COUNT + 1,
        seed=SEED,
        measure_disruptions=False,
        drop_on_disruption=False,
    )
    config = replace(
        base_scenario,
        scenario_id="nobel_eu_ofc_v1_trace_static_seed10_250",
        traffic_mode=TrafficMode.STATIC,
        traffic_source=str(traffic_path),
        capture_step_trace=True,
    )
    topology = TopologyModel.from_file(
        NOBEL_EU_PATH,
        topology_id=config.topology_id,
        k_paths=config.k_paths,
        max_span_length_km=config.max_span_length_km,
        default_attenuation_db_per_km=config.default_attenuation_db_per_km,
        default_noise_figure_db=config.default_noise_figure_db,
    )
    return OpticalEnv(
        config,
        topology,
        episode_length=REQUEST_COUNT + 1,
        capture_step_trace=True,
    )


def _assert_trace_values_match(expected, observed, *, abs_tol: float = 1e-9) -> None:
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        assert tuple(expected) == tuple(observed)
        for key, expected_value in expected.items():
            _assert_trace_values_match(expected_value, observed[key], abs_tol=abs_tol)
        return

    if isinstance(expected, (list, tuple)):
        assert isinstance(observed, type(expected))
        assert len(expected) == len(observed)
        for expected_item, observed_item in zip(expected, observed):
            _assert_trace_values_match(expected_item, observed_item, abs_tol=abs_tol)
        return

    if isinstance(expected, float):
        assert observed == pytest.approx(expected, abs=abs_tol)
        return

    assert observed == expected


def test_v1_nobel_eu_ofc_traffic_table_replays_exactly_for_250_requests(tmp_path: Path) -> None:
    legacy_env = _build_legacy_env()
    legacy_env.reset(seed=SEED)

    for _ in range(REQUEST_COUNT):
        mask = legacy_env.get_trace_action_mask()
        action = shortest_available_path_first_fit_best_modulation(mask)
        legacy_env.step(action)

    traffic_path = tmp_path / "nobel_eu_ofc_v1_seed10_250.jsonl"
    legacy_env.save_captured_traffic_table_jsonl(traffic_path)
    legacy_trace = legacy_env.export_step_trace()

    assert legacy_trace["header"]["topology_id"] == "nobel-eu"
    assert len(legacy_trace["steps"]) == REQUEST_COUNT

    v2_env = _build_v2_env(traffic_path=traffic_path)
    v2_env.reset(seed=SEED)

    for _ in range(REQUEST_COUNT):
        mask = v2_env.get_trace_action_mask()
        action = select_first_fit_action(mask)
        v2_env.step(action)

    v2_trace = v2_env.export_step_trace()

    _assert_trace_values_match(legacy_trace, v2_trace)
