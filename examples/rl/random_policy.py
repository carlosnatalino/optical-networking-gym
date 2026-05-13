from __future__ import annotations

from pathlib import Path

import numpy as np

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, make_env, set_topology_dir
from optical_networking_gym_v2.heuristics.masked_heuristics import select_random_action


TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR


def run_episode(seed: int = 42) -> dict[str, float | int]:
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
        episode_length=20,
        modulations_to_consider=2,
        k_paths=2,
    )
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()
        action = select_random_action(mask, rng=rng)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "episode_service_blocking_rate": float(info.get("episode_service_blocking_rate", 0.0)),
    }


def main() -> None:
    summary = run_episode()
    print(summary)


if __name__ == "__main__":
    main()
