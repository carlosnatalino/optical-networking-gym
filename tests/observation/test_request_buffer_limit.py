from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optical_networking_gym import (
    MaskMode,
    Modulation,
    QoTEngine,
    RequestAnalysisEngine,
    RuntimeState,
    ScenarioConfig,
    ServiceRequest,
    TopologyModel,
    build_scenario,
    make_env,
)
from optical_networking_gym.heuristics.dispatch import select_heuristic_action
from optical_networking_gym.heuristics.runtime_heuristics import select_first_fit_action


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RING_4_PATH = PROJECT_ROOT / "src" / "optical_networking_gym" / "topologies" / "ring_4.txt"


def _ring_config(**overrides: object) -> ScenarioConfig:
    config = ScenarioConfig(
        scenario_id="request_buffer_limit",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )
    return replace(config, **overrides)


def _request(service_id: int, destination_id: int, bit_rate: int = 40) -> ServiceRequest:
    return ServiceRequest(
        request_index=service_id,
        service_id=service_id,
        source_id=0,
        destination_id=destination_id,
        bit_rate=bit_rate,
        arrival_time=1.0 + service_id,
        holding_time=10.0,
    )


def _lean_env(arrivals: int, **overrides: object):
    # Resource-only analyses without observation or mask: the cheapest loop
    # that still builds one analysis per arrival on nobel-eu.
    return make_env(
        scenario="jocn_benchmark",
        episode_length=arrivals,
        overrides={
            "mask_mode": MaskMode.RESOURCE_ONLY,
            "enable_observation": False,
            "enable_action_mask": False,
            **overrides,
        },
    )


def _run_lean(arrivals: int, **overrides: object) -> RequestAnalysisEngine:
    env = _lean_env(arrivals, **overrides)
    env.reset(seed=3)
    for _ in range(arrivals):
        _, _, terminated, truncated, _ = env.step(select_first_fit_action(env))
        if terminated or truncated:
            break
    return env.unwrapped.simulator.analysis_engine


def test_default_limit_bounds_the_cache() -> None:
    assert build_scenario("jocn_benchmark").request_buffer_limit == 8
    engine = _run_lean(1_000)

    assert engine.cache_misses == 1_000
    assert engine.cache_hits == 0
    assert engine.cache_size <= 8
    assert engine.cache_evictions == 1_000 - engine.cache_size


def test_unlimited_keeps_every_analysis() -> None:
    engine = _run_lean(1_000, request_buffer_limit=-1)

    assert engine.cache_size == 1_000
    assert engine.cache_evictions == 0


def test_zero_disables_the_cache() -> None:
    engine = _run_lean(200, request_buffer_limit=0)

    assert engine.cache_size == 0
    assert engine.cache_misses == 200
    assert engine.cache_evictions == 0


def test_lru_eviction_order() -> None:
    topology = TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)
    config = _ring_config(request_buffer_limit=2)
    state = RuntimeState(config, topology)
    engine = RequestAnalysisEngine(config, topology, QoTEngine(config, topology))
    request_a, request_b, request_c = _request(1, 1), _request(2, 2), _request(3, 3)

    analysis_a = engine.build(state, request_a)
    engine.build(state, request_b)
    assert engine.build(state, request_a) is analysis_a
    analysis_c = engine.build(state, request_c)

    assert engine.cache_hits == 1
    assert engine.cache_misses == 3
    assert engine.cache_evictions == 1
    assert engine.cache_size == 2
    # A and C are kept, B was evicted.
    assert engine.build(state, request_a) is analysis_a
    assert engine.build(state, request_c) is analysis_c
    assert engine.cache_hits == 3
    engine.build(state, request_b)
    assert engine.cache_misses == 4

    engine.clear_cache()
    assert engine.cache_size == 0
    assert engine.cache_evictions == 0


def test_limit_is_read_at_every_insert() -> None:
    topology = TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)
    config = _ring_config(request_buffer_limit=-1)
    state = RuntimeState(config, topology)
    engine = RequestAnalysisEngine(config, topology, QoTEngine(config, topology))
    for service_id, destination_id in enumerate((1, 2, 3)):
        engine.build(state, _request(service_id, destination_id))
    assert engine.cache_size == 3

    engine.config = replace(config, request_buffer_limit=1)
    engine.build(state, _request(9, 1, bit_rate=100))
    assert engine.cache_size == 1
    assert engine.cache_evictions == 3


def test_limit_does_not_change_results() -> None:
    def run(limit: int) -> list[object]:
        env = make_env(scenario="jocn_benchmark", episode_length=120, request_buffer_limit=limit)
        observation, info = env.reset(seed=11)
        record: list[object] = [observation.copy(), info["mask"].copy()]
        for _ in range(120):
            action = select_heuristic_action("ksp-lb-bm", env, info)
            observation, reward, terminated, truncated, info = env.step(action)
            mask = info["mask"]
            record.append(
                (
                    observation.copy(),
                    None if mask is None else np.asarray(mask).copy(),
                    reward,
                    {key: value for key, value in info.items() if key != "mask"},
                )
            )
            if terminated or truncated:
                break
        return record

    reference = run(-1)
    for limit in (0, 8):
        other = run(limit)
        assert len(other) == len(reference)
        np.testing.assert_array_equal(other[0], reference[0])
        np.testing.assert_array_equal(other[1], reference[1])
        for (obs, mask, reward, info), (ref_obs, ref_mask, ref_reward, ref_info) in zip(
            other[2:], reference[2:]
        ):
            assert obs.tobytes() == ref_obs.tobytes()
            if ref_mask is None:
                assert mask is None
            else:
                np.testing.assert_array_equal(mask, ref_mask)
            assert reward == ref_reward
            assert info == ref_info


def test_limit_is_not_part_of_the_runtime_structure_key() -> None:
    config = build_scenario("ring4_quickstart")
    assert (
        replace(config, request_buffer_limit=0).runtime_structure_key()
        == config.runtime_structure_key()
    )


@pytest.mark.parametrize("value", [-2, 1.5, True, "8"])
def test_invalid_limits_raise(value: object) -> None:
    with pytest.raises(ValueError, match="request_buffer_limit must be -1"):
        build_scenario("ring4_quickstart", request_buffer_limit=value)
