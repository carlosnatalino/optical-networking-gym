from __future__ import annotations

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_AGING_PATH = PROJECT_ROOT / "examples" / "heuristics" / "network_aging.py"


def test_network_aging_example_ages_the_network() -> None:
    module = runpy.run_path(str(NETWORK_AGING_PATH))

    aged = module["run_episode"](seed=7, episode_length=400, aging_interval=100)
    not_aged = module["run_episode"](seed=7, episode_length=400, aging_interval=10**9)

    assert aged["steps"] == not_aged["steps"] == 400
    assert aged["physical_layer_version"] == 3
    assert not_aged["physical_layer_version"] == 0
    assert not_aged["disrupted_services_rate"] == 0.0
    assert aged["disrupted_services_rate"] > 0.0


def test_network_aging_example_with_zero_steps_matches_no_aging() -> None:
    module = runpy.run_path(str(NETWORK_AGING_PATH))

    zero_steps = module["run_episode"](
        seed=7,
        episode_length=400,
        aging_interval=100,
        attenuation_step_db_per_km=0.0,
        noise_figure_step_db=0.0,
    )
    not_aged = module["run_episode"](seed=7, episode_length=400, aging_interval=10**9)

    assert zero_steps["physical_layer_version"] == 3
    assert zero_steps["blocking_rate"] == not_aged["blocking_rate"]
    assert zero_steps["disrupted_services_rate"] == not_aged["disrupted_services_rate"] == 0.0
