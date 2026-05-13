from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "env_test.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("env_test_example", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load example module at {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_env_test_example_writes_visual_report(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "env_test_report.html"

    suite = module.generate_visual_report(
        output_path=output_path,
        topology_name="ring_4",
        steps=3,
        seed=7,
        num_slots=24,
        episode_length=6,
        k_paths=2,
        load=10.0,
    )

    payload = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    html_text = output_path.read_text(encoding="utf-8")

    assert suite.variant == "v2"
    assert suite.executed_steps > 0
    assert all(test.passed for test in suite.tests)
    assert output_path.exists()
    assert "Environment Visual Test Report" in html_text
    assert "Spectrum Occupancy" in html_text
    assert payload["suite"]["variant"] == "v2"
    assert payload["suite"]["executed_steps"] == suite.executed_steps
