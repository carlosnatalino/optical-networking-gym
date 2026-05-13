from __future__ import annotations

from pathlib import Path

import numpy as np

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, make_env, set_topology_dir
from optical_networking_gym_v2.heuristics.runtime_heuristics import select_random_action


TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR


def run_episode(seed: int = 7) -> dict[str, float | int | str]:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        topology_dir=TOPOLOGY_DIR,
        seed=seed,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=12,
        modulations_to_consider=2,
        k_paths=2,
        enable_action_mask=False,
    )
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    last_status = "unknown"

    while True:
        action = select_random_action(env.heuristic_context(), rng=rng)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        last_status = str(info.get("status", "unknown"))
        if terminated or truncated:
            break

    return {
        "mode": "runtime",
        "policy": "random",
        "steps": steps,
        "total_reward": total_reward,
        "last_status": last_status,
    }


def main() -> None:
    print(run_episode())


if __name__ == "__main__":
    main()
