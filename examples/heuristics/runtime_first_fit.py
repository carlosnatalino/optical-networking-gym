from __future__ import annotations

from optical_networking_gym_v2 import make_env
from optical_networking_gym_v2.defaults import (
    DEFAULT_K_PATHS,
    DEFAULT_LAUNCH_POWER_DBM,
    DEFAULT_LOAD,
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_MODULATIONS_TO_CONSIDER,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    DEFAULT_SEED,
)
from optical_networking_gym_v2.heuristics.runtime_heuristics import select_first_fit_action
from optical_networking_gym_v2.utils import experiment_scenarios as scenario_utils


def run_episode(seed: int = DEFAULT_SEED) -> dict[str, float | int | str]:
    scenario = scenario_utils.build_nobel_eu_graph_load_scenario(
        topology_id="nobel-eu",
        episode_length=1000,
        seed=seed,
        load=DEFAULT_LOAD,
        mean_holding_time=DEFAULT_MEAN_HOLDING_TIME,
        num_spectrum_resources=DEFAULT_NUM_SPECTRUM_RESOURCES,
        k_paths=DEFAULT_K_PATHS,
        launch_power_dbm=DEFAULT_LAUNCH_POWER_DBM,
        modulations_to_consider=DEFAULT_MODULATIONS_TO_CONSIDER,
        measure_disruptions=False,
        drop_on_disruption=False,
    )
    env = make_env(config=scenario)
    _, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    last_status = "unknown"

    while True:
        action = select_first_fit_action(env.heuristic_context())
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        last_status = str(info.get("status", "unknown"))
        if terminated or truncated:
            break

    return {
        "mode": "runtime",
        "policy": "first_fit",
        "steps": steps,
        "total_reward": total_reward,
        "last_status": last_status,
        "blocking_rate": float(info.get("episode_service_blocking_rate", 0.0)),
    }


def main() -> None:
    print(run_episode())


if __name__ == "__main__":
    main()
