from __future__ import annotations

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, ScenarioConfig, make_env
from optical_networking_gym_v2.heuristics.masked_heuristics import select_first_fit_action
from optical_networking_gym_v2.utils import build_nobel_eu_ofc_v1_scenario


TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR


def build_default_scenario(seed: int = 10) -> ScenarioConfig:
    return build_nobel_eu_ofc_v1_scenario(
        episode_length=1000,
        seed=seed,
        load=300.0,
        mean_holding_time=10800.0,
        num_spectrum_resources=320,
        k_paths=3,
        launch_power_dbm=0.0,
        modulations_to_consider=6,
        measure_disruptions=False,
        drop_on_disruption=False,
    )


def run_episode(seed: int = 10) -> dict[str, float | int | str]:
    env = make_env(config=build_default_scenario(seed=seed))
    _, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    last_status = "unknown"

    while True:
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()
        action = select_first_fit_action(mask)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        last_status = str(info.get("status", "unknown"))
        if terminated or truncated:
            break

    return {
        "mode": "masked",
        "policy": "first_fit",
        "steps": steps,
        "total_reward": total_reward,
        "last_status": last_status,
        "Info": info,
    }


def main() -> None:
    print(run_episode())


if __name__ == "__main__":
    main()
