from __future__ import annotations

from optical_networking_gym_v2.bench import (
    benchmark_action_mask,
    benchmark_allocation,
    benchmark_observation,
    benchmark_qot_engine,
    benchmark_request_analysis,
    benchmark_reward_function,
    benchmark_runtime_state,
)


def test_runtime_state_benchmark_returns_expected_keys() -> None:
    result = benchmark_runtime_state(iterations=5, warmup=1)

    assert result["component"] == "RuntimeState"
    assert result["iterations"] == 5
    assert result["cycle_mean_us"] >= 0.0
    assert result["provision_mean_us"] >= 0.0
    assert result["release_mean_us"] >= 0.0
    assert result["cycle_p95_us"] >= 0.0


def test_allocation_benchmark_returns_expected_keys() -> None:
    result = benchmark_allocation(iterations=5, warmup=1)

    assert result["component"] == "Allocation"
    assert result["iterations"] == 5
    assert result["mean_us"] >= 0.0
    assert result["p95_us"] >= 0.0
    assert result["candidate_count"] >= 0


def test_qot_engine_benchmark_returns_expected_keys() -> None:
    result = benchmark_qot_engine(iterations=5, warmup=1)

    assert result["component"] == "QoTEngine"
    assert result["iterations"] == 5
    assert result["evaluate_candidate_mean_us"] >= 0.0
    assert result["evaluate_candidate_p95_us"] >= 0.0
    assert result["refresh_service_mean_us"] >= 0.0
    assert result["refresh_service_p95_us"] >= 0.0


def test_action_mask_benchmark_returns_expected_keys() -> None:
    result = benchmark_action_mask(iterations=5, warmup=1)

    assert result["component"] == "ActionMask"
    assert result["iterations"] == 5
    assert result["valid_actions"] >= 0
    assert result["warm_mean_us"] >= 0.0
    assert result["warm_p95_us"] >= 0.0
    assert result["cold_mean_us"] >= 0.0
    assert result["cold_p95_us"] >= 0.0
    assert result["cold_mean_us"] >= result["warm_mean_us"]


def test_observation_benchmark_returns_expected_keys() -> None:
    result = benchmark_observation(iterations=5, warmup=1)

    assert result["component"] == "Observation"
    assert result["iterations"] == 5
    assert result["feature_size"] > 0
    assert result["warm_mean_us"] >= 0.0
    assert result["warm_p95_us"] >= 0.0
    assert result["cold_mean_us"] >= 0.0
    assert result["cold_p95_us"] >= 0.0
    assert result["cold_mean_us"] >= result["warm_mean_us"]


def test_request_analysis_benchmark_returns_expected_keys() -> None:
    result = benchmark_request_analysis(iterations=5, warmup=1)

    assert result["component"] == "RequestAnalysis"
    assert result["iterations"] == 5
    assert result["candidate_count"] >= 0
    assert result["warm_mean_us"] >= 0.0
    assert result["warm_p95_us"] >= 0.0
    assert result["cold_mean_us"] >= 0.0
    assert result["cold_p95_us"] >= 0.0
    assert result["cold_mean_us"] >= result["warm_mean_us"]


def test_reward_function_benchmark_returns_expected_keys() -> None:
    result = benchmark_reward_function(iterations=5, warmup=1)

    assert result["component"] == "RewardFunction"
    assert result["iterations"] == 5
    assert result["mean_us"] >= 0.0
    assert result["p95_us"] >= 0.0
