from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from optical_networking_gym_v2 import (
    Modulation,
    OpticalEnv,
    ScenarioConfig,
    TopologyModel,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RING_4_PATH = PROJECT_ROOT / "optical_networking_gym_v2" / "src" / "optical_networking_gym_v2" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _config() -> ScenarioConfig:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="optical_env_static",
        scenario_id="optical_env_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=2,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 0, 2, 40, 1.0, 8.0, table_id=table.table_id, row_index=0),
        TrafficRecord(1, 1, 0, 2, 40, 2.0, 8.0, table_id=table.table_id, row_index=1),
    )
    return ScenarioConfig(
        scenario_id="optical_env_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source={"table": table, "records": records},
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )


def _first_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask[:-1])
    return int(valid[0]) if valid.size > 0 else int(mask.shape[0] - 1)


def test_optical_env_exposes_spaces_and_action_masks() -> None:
    env = OpticalEnv(_config(), _topology(), episode_length=2)

    observation, info = env.reset(seed=7)

    assert env.action_space.n == 97
    assert env.observation_space.shape == observation.shape
    assert env.action_masks() is info["mask"]
    assert info["mask"] is not None
    assert info["mask"].flags.writeable is False

    action = _first_valid_action(info["mask"])
    _, _, _, _, step_info = env.step(action)

    assert env.action_masks() is not None
    assert env.action_masks() is step_info["mask"]
    assert step_info["mask"] is not None
    assert step_info["mask"].flags.writeable is False


def test_optical_env_returns_empty_observation_when_disabled() -> None:
    config = replace(_config(), enable_observation=False)
    env = OpticalEnv(config, _topology(), episode_length=2)

    observation, info = env.reset(seed=7)
    next_observation, _, terminated, truncated, _ = env.step(_first_valid_action(info["mask"]))

    assert env.observation_space.shape == (0,)
    assert observation.shape == (0,)
    assert observation.dtype == np.float32
    if not terminated and not truncated:
        assert next_observation.shape == (0,)
        assert next_observation.dtype == np.float32


def test_optical_env_can_disable_action_mask_everywhere() -> None:
    config = replace(_config(), enable_action_mask=False)
    env = OpticalEnv(config, _topology(), episode_length=2)

    _, info = env.reset(seed=7)

    assert info["mask"] is None
    assert env.action_masks() is None


def test_optical_env_can_hide_mask_from_info_but_keep_action_masks() -> None:
    config = replace(_config(), include_mask_in_info=False)
    env = OpticalEnv(config, _topology(), episode_length=2)

    _, info = env.reset(seed=7)

    assert info["mask"] is None
    assert env.action_masks() is not None
