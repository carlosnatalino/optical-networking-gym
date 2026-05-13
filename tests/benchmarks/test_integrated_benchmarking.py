from __future__ import annotations

from importlib import import_module

from optical_networking_gym_v2.bench import (
    benchmark_integrated_episode_vs_legacy,
    profile_simulator_episode,
)


def test_integrated_benchmark_builds_legacy_env_with_disruptions_disabled(monkeypatch) -> None:
    module = import_module("optical_networking_gym_v2.bench.integrated_benchmarking")
    captured: dict[str, object] = {}

    class DummyLegacyEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "QRMSAEnv", DummyLegacyEnv)

    module._build_legacy_env(
        topology_id="ring_4",
        num_spectrum_resources=24,
        k_paths=2,
        seed=7,
        load=10.0,
        mean_holding_time=100.0,
        episode_length=8,
    )

    assert "measure_disruptions" in captured
    assert captured["measure_disruptions"] is False


def test_integrated_benchmark_returns_expected_keys() -> None:
    result = benchmark_integrated_episode_vs_legacy(request_count=4, repeats=1, warmup=0)

    assert result["component"] == "IntegratedEpisodeReplay"
    assert result["request_count"] == 4
    assert result["v2_reset_mean_us"] >= 0.0
    assert result["v2_step_mean_us"] >= 0.0
    assert result["legacy_step_mean_us"] >= 0.0
    assert result["legacy_apply_mean_us"] >= 0.0
    assert result["v2_episode_mean_us"] >= 0.0
    assert result["legacy_episode_mean_us"] >= 0.0
    assert result["v2_episode_speedup_vs_legacy"] >= 0.0
    assert result["legacy_step_definition"].startswith("full request cycle")
    assert result["parity"]["status_matches"] is True
    assert result["parity"]["mask_matches"] is True
    assert result["parity"]["slot_matches"] is True
    assert "records" not in result["parity"]


def test_simulator_profile_returns_ranked_entries() -> None:
    result = profile_simulator_episode(request_count=4, top_n=5)

    assert result["component"] == "SimulatorProfile"
    assert result["request_count"] == 4
    assert result["elapsed_ms"] >= 0.0
    assert len(result["top_entries"]) == 5
    assert result["top_entries"][0]["cumulative_time_s"] >= result["top_entries"][-1]["cumulative_time_s"]
    assert "simulator.py" in result["rendered_stats"] or "request_analysis.py" in result["rendered_stats"]
