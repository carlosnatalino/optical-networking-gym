from __future__ import annotations

import json
from pathlib import Path

from optical_networking_gym_v2 import set_topology_dir
from optical_networking_gym_v2.optical.first_fit_example import run_episode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_ROOT.parent / "examples" / "topologies"


def test_basic_first_fit_example_writes_results_file(tmp_path: Path) -> None:
    set_topology_dir(TOPOLOGY_DIR)
    summary = run_episode(
        topology_name="ring_4",
        seed=7,
        episode_length=12,
        num_spectrum_resources=24,
        k_paths=2,
        load=10.0,
    )
    output_path = tmp_path / "basic_first_fit.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["topology_name"] == "ring_4"
    assert payload["steps"] == 12
    assert "episode_service_blocking_rate" in payload
    assert "episode_service_served_rate" in payload
