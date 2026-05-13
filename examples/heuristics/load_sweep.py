from __future__ import annotations

import csv
from pathlib import Path

from optical_networking_gym_v2 import make_env, select_first_fit_action, set_topology_dir


REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_DIR = REPO_ROOT / "examples" / "topologies"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_load_sweep(loads: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0)) -> Path:
    set_topology_dir(TOPOLOGY_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "load_sweep__first_fit__nobel-eu.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file_handler:
        writer = csv.writer(file_handler)
        writer.writerow(
            [
                "load",
                "episode_services_processed",
                "episode_services_accepted",
                "episode_service_blocking_rate",
                "episode_bit_rate_blocking_rate",
                "total_reward",
            ]
        )
        for load in loads:
            env = make_env(
                topology_name="nobel-eu",
                modulation_names="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
                seed=42,
                load=load,
                episode_length=100,
                modulations_to_consider=4,
                topology_dir=TOPOLOGY_DIR,
            )
            _, info = env.reset(seed=42)
            total_reward = 0.0
            while True:
                mask = info.get("mask")
                if mask is None:
                    mask = env.action_masks()
                action = select_first_fit_action(mask)
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            writer.writerow(
                [
                    load,
                    int(info.get("episode_services_processed", 0)),
                    int(info.get("episode_services_accepted", 0)),
                    float(info.get("episode_service_blocking_rate", 0.0)),
                    float(info.get("episode_bit_rate_blocking_rate", 0.0)),
                    total_reward,
                ]
            )
    return output_path


def main() -> None:
    output_path = run_load_sweep()
    print(f"Load sweep saved to {output_path}")


if __name__ == "__main__":
    main()
