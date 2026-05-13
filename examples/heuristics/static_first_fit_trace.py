from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from optical_networking_gym_v2 import (
    OpticalEnv,
    ScenarioConfig,
    TopologyModel,
    TrafficMode,
    get_modulations,
    select_first_fit_action,
)
from optical_networking_gym_v2.instrumentation.traces import write_step_trace_jsonl


from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR
TOPOLOGY_DIR = BUILTIN_TOPOLOGY_DIR
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def build_env(
    *,
    traffic_table_path: str | Path,
    topology_name: str = "ring6",
    seed: int = 42,
    episode_length: int = 100,
    num_spectrum_resources: int = 50,
    k_paths: int = 3,
) -> OpticalEnv:
    topology_path = TOPOLOGY_DIR / f"{topology_name}.txt"
    topology = TopologyModel.from_file(
        topology_path,
        topology_id=topology_name,
        k_paths=k_paths,
        max_span_length_km=80.0,
        default_attenuation_db_per_km=0.2,
        default_noise_figure_db=4.5,
    )
    config = ScenarioConfig(
        scenario_id=f"{topology_name}_static_trace_seed{seed}",
        topology_id=topology_name,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        traffic_mode=TrafficMode.STATIC,
        static_traffic_path=traffic_table_path,
        modulations=get_modulations("BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM"),
        modulations_to_consider=6,
        seed=seed,
        qot_constraint="ASE+NLI",
        measure_disruptions=False,
        margin=0.0,
        bandwidth=num_spectrum_resources * 12.5e9,
        capture_step_trace=True,
    )
    return OpticalEnv(
        config,
        topology,
        episode_length=episode_length,
        capture_step_trace=True,
    )


def run_episode(
    *,
    traffic_table_path: str | Path,
    topology_name: str = "ring6",
    seed: int = 42,
    episode_length: int = 100,
    num_spectrum_resources: int = 50,
    k_paths: int = 3,
) -> tuple[dict[str, object], dict[str, object]]:
    env = build_env(
        traffic_table_path=traffic_table_path,
        topology_name=topology_name,
        seed=seed,
        episode_length=episode_length,
        num_spectrum_resources=num_spectrum_resources,
        k_paths=k_paths,
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

    summary = {
        "topology_name": topology_name,
        "seed": seed,
        "episode_length": episode_length,
        "steps": steps,
        "traffic_table_path": str(Path(traffic_table_path)),
        "total_reward": total_reward,
        "episode_service_blocking_rate": float(info.get("episode_service_blocking_rate", 0.0)),
        "episode_bit_rate_blocking_rate": float(info.get("episode_bit_rate_blocking_rate", 0.0)),
        "episode_services_processed": int(info.get("episode_services_processed", steps)),
        "episode_services_accepted": int(info.get("episode_services_accepted", 0)),
        "status_counts": dict(statuses),
    }
    return summary, env.export_step_trace()


def save_results(summary: dict[str, object], trace: dict[str, object]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"static_first_fit_trace__{summary['topology_name']}__seed_{summary['seed']}__ep_{summary['steps']}"
    )
    summary_path = RESULTS_DIR / f"{base_name}.json"
    trace_path = RESULTS_DIR / f"{base_name}.trace.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path, write_step_trace_jsonl(trace, trace_path)


def main() -> None:
    traffic_table_path = RESULTS_DIR / "ring6__seed_42__traffic.jsonl"
    summary, trace = run_episode(traffic_table_path=traffic_table_path)
    summary_path, trace_path = save_results(summary, trace)
    print(f"Steps: {summary['steps']}  |  Total reward: {summary['total_reward']:.2f}")
    print(f"Blocking rate: {summary['episode_service_blocking_rate']:.4f}")
    print(f"Summary saved to: {summary_path}")
    print(f"Trace saved to: {trace_path}")


if __name__ == "__main__":
    main()
