from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from optical_networking_gym_v2 import make_env, select_first_fit_action


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_episode(
    *,
    topology_name: str = "nobel-eu",
    seed: int = 42,
    load: float = 300.0,
    episode_length: int = 100,
    num_spectrum_resources: int = 320,
    modulations_to_consider: int = 4,
    k_paths: int = 3,
    bit_rates: tuple[int, ...] = (10, 40, 100, 400),
    modulation_names: str = "BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
) -> dict[str, object]:
    scenario_name = "ring4_quickstart" if topology_name == "ring_4" else "nobel_eu_baseline"
    env = make_env(
        scenario=scenario_name,
        modulations=modulation_names,
        seed=seed,
        bit_rates=bit_rates,
        load=load,
        num_spectrum_resources=num_spectrum_resources,
        episode_length=episode_length,
        modulations_to_consider=modulations_to_consider,
        k_paths=k_paths,
        overrides={"topology_id": topology_name},
    )
    _, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    statuses: Counter[str] = Counter()

    while True:
        mask = info.get("mask")
        if mask is None:
            mask = env.action_masks()

        action = select_first_fit_action(mask)
        _, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        statuses[str(info.get("status", "unknown"))] += 1

        if terminated or truncated:
            break

    return {
        "topology_name": topology_name,
        "seed": seed,
        "load": load,
        "episode_length": episode_length,
        "steps": steps,
        "total_reward": total_reward,
        "episode_service_blocking_rate": float(info.get("episode_service_blocking_rate", 0.0)),
        "episode_bit_rate_blocking_rate": float(info.get("episode_bit_rate_blocking_rate", 0.0)),
        "episode_services_processed": int(info.get("episode_services_processed", steps)),
        "episode_services_accepted": int(info.get("episode_services_accepted", 0)),
        "status_counts": dict(statuses),
        "blocked_due_to_resources_decisions": int(statuses.get("blocked_resources", 0)),
        "blocked_due_to_qot_decisions": int(statuses.get("blocked_qot", 0)),
    }


def save_results(summary: dict[str, object]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        f"basic_first_fit__{summary['topology_name']}__seed_{summary['seed']}__ep_{summary['steps']}.json"
    )
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    summary = run_episode()
    output_path = save_results(summary)
    print(f"Steps: {summary['steps']}  |  Total reward: {summary['total_reward']:.2f}")
    print(f"Blocking rate: {summary['episode_service_blocking_rate']:.4f}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
