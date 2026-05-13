from __future__ import annotations

from pathlib import Path
import runpy

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKED_FIRST_FIT_PATH = PROJECT_ROOT / "examples" / "heuristics" / "masked_first_fit.py"
MASKED_RANDOM_PATH = PROJECT_ROOT / "examples" / "heuristics" / "masked_random.py"
RUNTIME_FIRST_FIT_PATH = PROJECT_ROOT / "examples" / "heuristics" / "runtime_first_fit.py"
RUNTIME_RANDOM_PATH = PROJECT_ROOT / "examples" / "heuristics" / "runtime_random.py"


def _run_example(path: Path) -> dict[str, object]:
    module = runpy.run_path(str(path))
    return module["run_episode"](seed=7)


def test_masked_first_fit_example_runs() -> None:
    summary = _run_example(MASKED_FIRST_FIT_PATH)

    assert summary["mode"] == "masked"
    assert summary["policy"] == "first_fit"
    assert summary["steps"] == 1000


def test_masked_first_fit_example_uses_ofc_v1_defaults() -> None:
    module = runpy.run_path(str(MASKED_FIRST_FIT_PATH))
    scenario = module["build_default_scenario"](seed=7)

    assert scenario.topology_id == "nobel-eu"
    assert scenario.episode_length == 1000
    assert scenario.measure_disruptions is False
    assert scenario.max_span_length_km == pytest.approx(80.0)
    assert tuple(modulation.minimum_osnr for modulation in scenario.modulations) == pytest.approx(
        (
            3.71925646843142,
            6.72955642507124,
            10.8453935345953,
            13.2406469649752,
            16.1608982942870,
            19.0134649345090,
        )
    )


def test_masked_random_example_runs() -> None:
    summary = _run_example(MASKED_RANDOM_PATH)

    assert summary["mode"] == "masked"
    assert summary["policy"] == "random"
    assert summary["steps"] == 12


def test_runtime_first_fit_example_runs_without_action_mask() -> None:
    summary = _run_example(RUNTIME_FIRST_FIT_PATH)

    assert summary["mode"] == "runtime"
    assert summary["policy"] == "first_fit"
    assert summary["steps"] == 1000


def test_runtime_random_example_runs_without_action_mask() -> None:
    summary = _run_example(RUNTIME_RANDOM_PATH)

    assert summary["mode"] == "runtime"
    assert summary["policy"] == "random"
    assert summary["steps"] == 12
