from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys

from optical_networking_gym import TopologyModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "create_topology.py"


def test_create_topology_script_pickles_a_topology_model(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--topology",
            "nsfnet_chen",
            "-k",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0, result.stderr

    pickle_path = tmp_path / "nsfnet_chen_2-paths.pkl"
    assert pickle_path.exists()

    with pickle_path.open("rb") as handle:
        topology = pickle.load(handle)

    assert isinstance(topology, TopologyModel)
    assert topology.node_count == 14  # NSFNET
