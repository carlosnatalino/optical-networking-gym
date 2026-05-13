from __future__ import annotations

import cProfile
from pathlib import Path

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, make_env, select_first_fit_action, set_topology_dir
from optical_networking_gym_v2.instrumentation.profiling import write_cprofile_stats


TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR


def run_episode() -> None:
    env = make_env(
        topology_name="nobel-eu",
        modulation_names="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
        seed=10,
        load=400.0,
        episode_length=100,
        modulations_to_consider=3,
        topology_dir=TOPOLOGY_DIR,
    )
    _, info = env.reset(seed=10)

    while True:
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()

        action = select_first_fit_action(mask)
        _, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break


def main() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    profiler = cProfile.Profile()
    profiler.enable()
    run_episode()
    profiler.disable()

    output_path = write_cprofile_stats(profiler, Path("perf") / "profile_v2.txt")
    print(f"Profile saved to {output_path}")


if __name__ == "__main__":
    main()
