from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import runpy

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "legacy_benchmark" / "osnr_margin_sweep.py"
NOTEBOOK_PATH = (
    PROJECT_ROOT / "examples" / "legacy_benchmark" / "osnr_margin_sweep_analysis.ipynb"
)


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE_PATH))


def test_legacy_benchmark_margin_sweep_defaults_enable_disruption_tracking() -> None:
    module = _load_module()
    experiment = module["LegacyBenchmarkMarginSweepExperiment"]()
    scenario = module["build_base_scenario"](experiment=experiment, load=100.0, margin=0.5, seed=11)

    assert scenario.measure_disruptions is True
    assert scenario.drop_on_disruption is True
    assert experiment.processes == 1
    assert scenario.margin == pytest.approx(0.5)
    assert scenario.load == pytest.approx(100.0)


def test_legacy_benchmark_margin_sweep_rejects_non_positive_process_count() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="processes must be positive"):
        module["LegacyBenchmarkMarginSweepExperiment"](processes=0)


def test_legacy_benchmark_margin_sweep_selects_higher_load_on_tie() -> None:
    module = _load_module()
    selected = module["select_single_load_from_pilot_rows"](
        [
            {"load": 300.0, "effective_blocking_rate_mean": 0.011},
            {"load": 310.0, "effective_blocking_rate_mean": 0.009},
            {"load": 320.0, "effective_blocking_rate_mean": 0.014},
        ],
        target_rate=0.010,
    )

    assert selected == pytest.approx(310.0)


def test_legacy_benchmark_margin_sweep_smoke_generates_expected_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    experiment = module["LegacyBenchmarkMarginSweepExperiment"](
        topology_id="ring_4",
        loads=(10.0,),
        margins=(0.0,),
        episodes_per_point=1,
        episode_length=8,
        seed=7,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        k_paths=2,
        modulations_to_consider=2,
        pilot_load_start=10.0,
        pilot_load_end=10.0,
        pilot_load_step=10.0,
        pilot_episodes=1,
        output_dir=tmp_path,
    )

    outputs = module["run_margin_sweep"](experiment=experiment, now=datetime(2026, 3, 19, 0, 0))

    assert outputs.run_dir == tmp_path / "19-03-00h00-osnr-margin-sweep"
    assert outputs.pilot_summary_csv == outputs.run_dir / "pilot-summary.csv"
    assert outputs.episodes_csv == outputs.run_dir / "episodes.csv"
    assert outputs.summary_csv == outputs.run_dir / "summary.csv"
    assert outputs.requests_csv == outputs.run_dir / "requests.csv"
    assert outputs.pilot_summary_csv.exists()
    assert outputs.episodes_csv.exists()
    assert outputs.summary_csv.exists()
    assert outputs.requests_csv.exists()

    with outputs.pilot_summary_csv.open("r", encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    with outputs.episodes_csv.open("r", encoding="utf-8", newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    with outputs.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with outputs.requests_csv.open("r", encoding="utf-8", newline="") as handle:
        request_rows = list(csv.DictReader(handle))

    assert len(pilot_rows) == 1
    assert len(episode_rows) == 1
    assert len(summary_rows) == 1
    assert len(request_rows) == 8

    summary_row = summary_rows[0]
    episode_row = episode_rows[0]
    request_row = request_rows[0]

    assert summary_row["analysis_group"] == "multi_load"
    assert summary_row["is_single_load_focus"] == "True"
    assert "effective_blocking_rate_mean" in summary_row
    assert "accepted_osnr_margin_p95" in summary_row
    assert "final_osnr_margin_p95" in summary_row
    assert "fragmentation_shannon_entropy_mean" in summary_row
    assert "is_best_margin_for_load" in summary_row
    assert "best_margin_rank_within_load" in summary_row

    assert float(episode_row["effective_blocking_rate"]) == pytest.approx(
        1.0 - float(episode_row["service_served_rate"]),
        abs=1e-9,
    )
    if int(episode_row["services_accepted"]) > 0:
        assert float(episode_row["drop_after_accept_rate"]) == pytest.approx(
            float(episode_row["disrupted_or_dropped_services"]) / float(episode_row["services_accepted"]),
            abs=1e-9,
        )

    assert "osnr_margin" in request_row
    assert "fragmentation_shannon_entropy" in request_row
    assert "fragmentation_route_cuts" in request_row
    assert "fragmentation_route_rss" in request_row


def test_legacy_benchmark_margin_sweep_notebook_exists_and_references_artifacts() -> None:
    assert NOTEBOOK_PATH.exists()

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    joined_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    assert "pilot-summary.csv" in joined_source
    assert "summary.csv" in joined_source
    assert "episodes.csv" in joined_source
    assert "requests.csv" in joined_source
    assert "effective_blocking_rate_mean" in joined_source
