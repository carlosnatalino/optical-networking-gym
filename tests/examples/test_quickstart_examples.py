from __future__ import annotations

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART_PATH = PROJECT_ROOT / "examples" / "quickstart" / "basic_first_fit.py"


def test_quickstart_basic_first_fit_runs_canonical_loop() -> None:
    module = runpy.run_path(str(QUICKSTART_PATH))
    summary = module["run_episode"](
        topology_name="ring_4",
        seed=7,
        load=10.0,
        episode_length=4,
        num_spectrum_resources=24,
        modulations_to_consider=2,
        k_paths=2,
        bit_rates=(40,),
        modulation_names="QPSK,16QAM",
    )

    assert summary["steps"] == 4
    assert summary["episode_services_processed"] == 4
    assert summary["episode_services_accepted"] >= 1
