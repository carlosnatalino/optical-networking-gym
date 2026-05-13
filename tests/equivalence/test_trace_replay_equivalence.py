from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation,
)
from optical_networking_gym.topology import Modulation as LegacyModulation
from optical_networking_gym.topology import get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym_v2.contracts.enums import TrafficMode
from optical_networking_gym_v2.contracts.modulation import Modulation
from optical_networking_gym_v2.envs.optical_env import OpticalEnv
from optical_networking_gym_v2.heuristics import select_first_fit_action
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.simulation.scenario import ScenarioConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_DIR = PROJECT_ROOT / "examples" / "topologies"
RING6_PATH = TOPOLOGY_DIR / "ring6.txt"


def _legacy_modulations() -> tuple[LegacyModulation, ...]:
    return (
        LegacyModulation("BPSK", 100_000, 1, minimum_osnr=3.71, inband_xt=-14),
        LegacyModulation("QPSK", 2_000, 2, minimum_osnr=6.72, inband_xt=-17),
        LegacyModulation("8QAM", 1_000, 3, minimum_osnr=10.84, inband_xt=-20),
        LegacyModulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
        LegacyModulation("32QAM", 250, 5, minimum_osnr=16.16, inband_xt=-26),
        LegacyModulation("64QAM", 125, 6, minimum_osnr=19.01, inband_xt=-29),
    )


def _v2_modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("BPSK", 100_000.0, 1, minimum_osnr=3.71, inband_xt=-14.0),
        Modulation("QPSK", 2_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("8QAM", 1_000.0, 3, minimum_osnr=10.84, inband_xt=-20.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        Modulation("32QAM", 250.0, 5, minimum_osnr=16.16, inband_xt=-26.0),
        Modulation("64QAM", 125.0, 6, minimum_osnr=19.01, inband_xt=-29.0),
    )


def _build_legacy_env() -> QRMSAEnvWrapper:
    topology = get_topology(
        str(RING6_PATH),
        topology_name="ring6",
        modulations=_legacy_modulations(),
        max_span_length=80.0,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=3,
    )
    return QRMSAEnvWrapper(
        topology=topology,
        seed=42,
        allow_rejection=True,
        load=300.0,
        episode_length=101,
        num_spectrum_resources=50,
        launch_power_dbm=0.0,
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8 / 1565e-9,
        bandwidth=50 * 12.5e9,
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
    topology = TopologyModel.from_file(
        RING6_PATH,
        topology_id="ring6",
        k_paths=3,
        max_span_length_km=80.0,
        default_attenuation_db_per_km=0.2,
        default_noise_figure_db=4.5,
    )
    config = ScenarioConfig(
        scenario_id="ring6_trace_static",
        topology_id="ring6",
        k_paths=3,
        num_spectrum_resources=50,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=str(traffic_path),
        modulations=_v2_modulations(),
        modulations_to_consider=6,
        seed=42,
        qot_constraint="ASE+NLI",
        measure_disruptions=False,
        margin=0.0,
        bandwidth=50 * 12.5e9,
    )
    return OpticalEnv(
        config,
        topology,
        episode_length=101,
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


def test_v1_dynamic_capture_exports_v2_compatible_static_table_and_exact_trace_parity(
    tmp_path: Path,
) -> None:
    legacy_env = _build_legacy_env()
    legacy_env.reset(seed=42)

    for _ in range(100):
        mask = legacy_env.get_trace_action_mask()
        action = shortest_available_path_first_fit_best_modulation(mask)
        legacy_env.step(action)

    traffic_path = tmp_path / "ring6__seed_42__traffic.jsonl"
    legacy_env.save_captured_traffic_table_jsonl(traffic_path)
    legacy_trace = legacy_env.export_step_trace()

    assert legacy_trace["header"]["topology_id"] == "ring6"
    assert len(legacy_trace["steps"]) == 100

    v2_env = _build_v2_env(traffic_path=traffic_path)
    v2_env.reset(seed=42)

    for _ in range(100):
        mask = v2_env.get_trace_action_mask()
        action = select_first_fit_action(mask)
        v2_env.step(action)

    v2_trace = v2_env.export_step_trace()

    _assert_trace_values_match(legacy_trace, v2_trace)
