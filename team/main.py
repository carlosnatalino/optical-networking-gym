import json
from datetime import datetime, timezone
from pathlib import Path

from team.agents.first_fit import FirstFitAgent
from team.env_factory import create_env
from team.runner import run_episode


def main():
    config = {
        "seed": 42,
        "load": 300.0,
        "episode_length": 100,
    }

    agent = FirstFitAgent()
    env = create_env(config)

    try:
        metrics = run_episode(env, agent, seed=config["seed"])
    finally:
        env.close()

    result = {
        "algorithm": agent.name,
        "config": config,
        "metrics": metrics,
    }

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_path = results_dir / f"{agent.name}_{timestamp}.json"

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nKết quả:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu: {output_path}")


if __name__ == "__main__":
    main()