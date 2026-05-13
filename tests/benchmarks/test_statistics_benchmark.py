from __future__ import annotations

from optical_networking_gym_v2.bench import benchmark_statistics_step_info


def test_statistics_step_info_benchmark_returns_expected_keys() -> None:
    result = benchmark_statistics_step_info(iterations=5, warmup=1)

    assert result["component"] == "StatisticsStepInfo"
    assert result["iterations"] == 5
    assert result["mean_us"] >= 0.0
    assert result["p95_us"] >= 0.0
