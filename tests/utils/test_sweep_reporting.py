from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from optical_networking_gym_v2.utils.sweep_reporting import aggregate_summary_metrics
from optical_networking_gym_v2.utils.sweep_reporting import (
    Parallelism,
    create_experiment_run,
    run_cases,
)


def test_aggregate_summary_metrics_keeps_only_requested_kpis() -> None:
    rows = [
        {
            "services_accepted": 10.0,
            "services_served": 9.0,
            "service_blocking_rate": 0.1,
            "bit_rate_blocking_rate": 0.2,
            "mean_osnr_final": 17.0,
            "blocked_due_to_resources": 2.0,
        },
        {
            "services_accepted": 14.0,
            "services_served": 12.0,
            "service_blocking_rate": 0.2,
            "bit_rate_blocking_rate": 0.3,
            "mean_osnr_final": 19.0,
            "blocked_due_to_resources": 5.0,
        },
    ]

    aggregated = aggregate_summary_metrics(
        rows,
        metric_names=(
            "services_accepted",
            "services_served",
            "service_blocking_rate",
            "bit_rate_blocking_rate",
            "mean_osnr_final",
        ),
    )

    assert aggregated["services_accepted_mean"] == 12.0
    assert aggregated["services_accepted_std"] == 2.0
    assert aggregated["mean_osnr_final_mean"] == 18.0
    assert "blocked_due_to_resources_mean" not in aggregated


def _double(value: int) -> int:
    return value * 2


def test_create_experiment_run_writes_standard_layout(tmp_path: Path) -> None:
    run = create_experiment_run(
        script_path=Path("examples/SBRT2026/osnr_margin_sweep.py"),
        base_dir=tmp_path,
        family="SBRT2026",
        now=datetime(2026, 5, 12, 14, 30),
        scenario_name="nobel_eu_baseline",
        scenario_id="nobel_eu_baseline_load400",
        overrides={"load": 400},
        parallelism=Parallelism(workers=2, envs_per_worker=1),
    )

    assert run.run_dir == tmp_path / "SBRT2026" / "osnr_margin_sweep" / "20260512-143000"
    assert run.artifact_path("episodes") == run.run_dir / "episodes.csv"
    assert run.artifact_path("summary") == run.run_dir / "summary.csv"
    assert run.metadata_path == run.run_dir / "metadata.json"
    assert run.metadata["script_name"] == "osnr_margin_sweep.py"
    assert run.metadata["workers"] == 2
    assert run.metadata["envs_per_worker"] == 1
    assert run.metadata["max_active_envs"] == 2
    assert run.metadata_path.exists()


def test_parallelism_auto_uses_cpu_count_minus_two(monkeypatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 8)

    assert Parallelism.auto().resolve().workers == 6


def test_parallelism_validates_positive_values() -> None:
    with pytest.raises(ValueError, match="workers"):
        Parallelism(workers=0).resolve()

    with pytest.raises(ValueError, match="envs_per_worker"):
        Parallelism(envs_per_worker=0).resolve()


def test_run_cases_sequential() -> None:
    assert run_cases((1, 2, 3), _double, parallelism=Parallelism(workers=1)) == [2, 4, 6]
