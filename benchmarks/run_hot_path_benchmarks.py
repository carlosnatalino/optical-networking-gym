from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hot_path_benchmarks import run_hot_path_benchmarks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v2 hot-path benchmarks against legacy baselines")
    parser.add_argument("--runtime-iterations", type=int, default=1_000)
    parser.add_argument("--runtime-warmup", type=int, default=100)
    parser.add_argument("--allocation-iterations", type=int, default=5_000)
    parser.add_argument("--allocation-warmup", type=int, default=500)
    parser.add_argument("--qot-iterations", type=int, default=1_000)
    parser.add_argument("--qot-warmup", type=int, default=100)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    results = run_hot_path_benchmarks(
        runtime_iterations=args.runtime_iterations,
        runtime_warmup=args.runtime_warmup,
        allocation_iterations=args.allocation_iterations,
        allocation_warmup=args.allocation_warmup,
        qot_iterations=args.qot_iterations,
        qot_warmup=args.qot_warmup,
    )
    rendered = json.dumps(results, indent=2)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
