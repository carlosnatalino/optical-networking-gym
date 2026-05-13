from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optical_networking_gym_v2.bench.integrated_benchmarking import (
    benchmark_integrated_episode_vs_legacy,
    benchmark_simulator_episode,
    profile_simulator_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run integrated v2 simulator benchmarks and profile")
    parser.add_argument("--topology-id", default="ring_4")
    parser.add_argument("--k-paths", type=int, default=2)
    parser.add_argument("--num-spectrum-resources", type=int, default=24)
    parser.add_argument("--request-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--load", type=float, default=10.0)
    parser.add_argument("--mean-holding-time", type=float, default=100.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    results: dict[str, object] = {
        "simulator_benchmark": benchmark_simulator_episode(
            topology_id=args.topology_id,
            k_paths=args.k_paths,
            num_spectrum_resources=args.num_spectrum_resources,
            request_count=args.request_count,
            seed=args.seed,
            load=args.load,
            mean_holding_time=args.mean_holding_time,
            repeats=args.repeats,
            warmup=args.warmup,
        ),
        "simulator_profile": profile_simulator_episode(
            topology_id=args.topology_id,
            k_paths=args.k_paths,
            num_spectrum_resources=args.num_spectrum_resources,
            request_count=args.request_count,
            seed=args.seed,
            load=args.load,
            mean_holding_time=args.mean_holding_time,
            top_n=args.top_n,
        ),
    }
    if not args.skip_legacy:
        results["legacy_comparison"] = benchmark_integrated_episode_vs_legacy(
            topology_id=args.topology_id,
            k_paths=args.k_paths,
            num_spectrum_resources=args.num_spectrum_resources,
            request_count=args.request_count,
            seed=args.seed,
            load=args.load,
            mean_holding_time=args.mean_holding_time,
            repeats=args.repeats,
            warmup=args.warmup,
        )

    rendered = json.dumps(results, indent=2)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
