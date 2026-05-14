from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import runpy
import tomllib

import pytest
from optical_networking_gym_v2.utils import Parallelism, SimulationUtils, compare_summary_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "JOCN_Benchmark_2024"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


def _load_script(name: str) -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE_DIR / name))


def _test_scenarios(*, load=10.0, margin=0.0, launch_power_dbm=-4.0):
    return SimulationUtils.create_environment(
        topology_name="ring_4",
        seed=7,
        load=load,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=8,
        k_paths=2,
        launch_power_dbm=launch_power_dbm,
        margin=margin,
        modulations_to_consider=2,
    )


def test_topology_assets_are_declared_as_package_data() -> None:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

    package_data = data["tool"]["setuptools"]["package-data"]
    assert "optical_networking_gym_v2" in package_data
    assert "topologies/*.xml" in package_data["optical_networking_gym_v2"]
    assert "topologies/*.txt" in package_data["optical_networking_gym_v2"]


def test_jocn_scripts_do_not_import_root_package() -> None:
    for script_name in ("graph_load.py", "graph_margin.py", "graph_launch_power.py"):
        source = (EXAMPLE_DIR / script_name).read_text(encoding="utf-8")

        assert "from optical_networking_gym " not in source
        assert "import optical_networking_gym " not in source
        assert "optical_networking_gym.wrappers" not in source
        assert "sys.path" not in source


def test_jocn_load_sweep_rebuilds_traffic_source_for_each_load() -> None:
    module = _load_script("graph_load.py")

    scenarios = module["create_env"]((50.0, 650.0), episode_length=8)

    assert [scenario.load for scenario in scenarios] == [50.0, 650.0]
    assert [scenario.traffic_source["load"] for scenario in scenarios] == [50.0, 650.0]
    assert [scenario.traffic_source["mean_holding_time"] for scenario in scenarios] == [10800.0, 10800.0]


@pytest.mark.parametrize(
    ("script_name", "experiment_name", "kwargs", "expected_rows"),
    [
        (
            "graph_load.py",
            "LoadSweepExperiment",
            {"loads": (10.0,)},
            4,
        ),
        (
            "graph_margin.py",
            "MarginSweepExperiment",
            {"margins": (0.0, 1.0)},
            2,
        ),
        (
            "graph_launch_power.py",
            "LaunchPowerSweepExperiment",
            {"launch_powers_dbm": (-4.0,)},
            1,
        ),
    ],
)
def test_jocn_benchmark_smoke_generates_standard_artifacts(
    tmp_path: Path,
    script_name: str,
    experiment_name: str,
    kwargs: dict[str, object],
    expected_rows: int,
    ) -> None:
    module = _load_script(script_name)
    experiment_type = module[experiment_name]
    if script_name == "graph_load.py":
        experiment_type.scenarios_by_value.__globals__["create_env"] = lambda load, **_kwargs: _test_scenarios(
            load=load
        )
    elif script_name == "graph_margin.py":
        experiment_type.scenarios_by_value.__globals__["create_env"] = lambda margin, **_kwargs: _test_scenarios(
            margin=margin
        )[0]
    else:
        experiment_type.scenarios_by_value.__globals__["create_env"] = (
            lambda launch_power_dbm, **_kwargs: _test_scenarios(launch_power_dbm=launch_power_dbm)[0]
        )
    experiment = experiment_type(
        episodes_per_point=1,
        output_dir=tmp_path,
        parallelism=Parallelism(workers=1),
        **kwargs,
    )

    outputs = module["run_sweep"](experiment=experiment, now=datetime(2026, 5, 13))

    expected_run_dir = tmp_path / "JOCN_Benchmark_2024" / Path(script_name).stem / "20260513-000000"
    assert outputs.run_dir == expected_run_dir
    assert outputs.episodes_csv == expected_run_dir / "episodes.csv"
    assert outputs.summary_csv == expected_run_dir / "summary.csv"
    assert outputs.metadata_json == expected_run_dir / "metadata.json"
    assert outputs.episodes_csv.exists()
    assert outputs.summary_csv.exists()
    assert outputs.metadata_json.exists()

    with outputs.episodes_csv.open("r", encoding="utf-8", newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    with outputs.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))

    assert len(episode_rows) == expected_rows
    assert len(summary_rows) == expected_rows
    assert metadata["script_name"] == script_name
    assert metadata["scenario_name"] == "JOCN_Benchmark_2024"
    assert episode_rows[0]["topology_id"] == "ring_4"
    assert episode_rows[0]["requests_per_episode"] == "8"
    assert "effective_blocking_rate" in episode_rows[0]
    assert "effective_blocking_rate_mean" in summary_rows[0]
    if script_name == "graph_load.py":
        assert {row["policy"] for row in summary_rows} == {
            "KSP-FF-BM",
            "LS-BM-KSP",
            "BM-KSP-LB",
            "KSP-LB-BM",
        }


def test_jocn_capture_services_writes_required_service_fields(tmp_path: Path) -> None:
    module = _load_script("graph_launch_power.py")
    experiment_type = module["LaunchPowerSweepExperiment"]
    experiment_type.scenarios_by_value.__globals__["create_env"] = (
        lambda launch_power_dbm, **_kwargs: _test_scenarios(launch_power_dbm=launch_power_dbm)[0]
    )
    experiment = experiment_type(
        launch_powers_dbm=(-4.0,),
        episodes_per_point=1,
        capture_services=True,
        output_dir=tmp_path,
        parallelism=Parallelism(workers=1),
    )

    outputs = module["run_sweep"](experiment=experiment, now=datetime(2026, 5, 13))

    assert outputs.services_csv is not None
    assert outputs.services_csv.exists()
    with outputs.services_csv.open("r", encoding="utf-8", newline="") as handle:
        service_rows = list(csv.DictReader(handle))

    assert service_rows
    required_fields = {
        "osnr",
        "ase",
        "nli",
        "modulation",
        "path_length",
        "path_k",
        "load",
        "margin",
        "launch_power_dbm",
        "episode_index",
        "policy",
    }
    assert required_fields <= set(service_rows[0])


def test_jocn_summary_comparison_uses_explicit_tolerances() -> None:
    reference = [{"load": 10.0, "effective_blocking_rate_mean": 0.10}]
    candidate = [{"load": 10.0, "effective_blocking_rate_mean": 0.105}]

    results = compare_summary_rows(
        reference,
        candidate,
        key_fields=("load",),
        metric_tolerances={"effective_blocking_rate_mean": 0.01},
    )

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].metric == "effective_blocking_rate_mean"
    assert results[0].absolute_delta == pytest.approx(0.005)


def test_jocn_plot_notebook_is_lightweight_and_references_standard_csvs() -> None:
    notebook_path = EXAMPLE_DIR / "plots.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4

    joined_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert "summary.csv" in joined_source
    assert "episodes.csv" in joined_source
    assert "services.csv" in joined_source
    assert "jocn_benchmark" in joined_source
    assert "ax.set_yscale('log')" in joined_source
    assert "Request blocking rate" in joined_source
    assert "Bit rate blocking rate" in joined_source
    assert "PercentFormatter" not in joined_source
