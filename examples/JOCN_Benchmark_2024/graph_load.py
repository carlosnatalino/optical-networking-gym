from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from optical_networking_gym_v2 import build_scenario
from optical_networking_gym_v2.utils.experiment_utils import SimulationUtils
from optical_networking_gym_v2.utils.experiment_utils import build_standard_sweep_parser, run_first_fit_sweep
from optical_networking_gym_v2.utils.sweep_reporting import Parallelism


SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY = "JOCN_Benchmark_2024"
DEFAULT_LOAD = (50.0, 650.0, 50.0)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
LEGACY_POLICIES = ("KSP-FF-BM", "LS-BM-KSP", "BM-KSP-LB", "KSP-LB-BM")


def default_loads_for_topology(topology_id: str) -> tuple[float, float, float]:
    if topology_id == "nobel-eu":
        return (50.0, 650.0, 50.0)
    if topology_id == "germany50":
        return (300.0, 800.0, 50.0)
    if topology_id == "janos-us":
        return (100.0, 600.0, 50.0)
    if topology_id == "nsfnet_chen":
        return (100.0, 600.0, 50.0)
    raise ValueError(f"unknown topology id: {topology_id}")


def create_env(
    load: float | tuple[float, ...] = DEFAULT_LOAD,
    *,
    topology_id: str = "nobel-eu",
    episode_length: int = 1_000,
):
    return tuple(
        build_scenario(
            "jocn_benchmark",
            scenario_id=f"jocn_benchmark_{topology_id}_load_{current_load:g}",
            topology_id=topology_id,
            episode_length=episode_length,
            load=float(current_load),
        )
        for current_load in SimulationUtils.normalize_values(load)
    )


@dataclass(frozen=True, slots=True)
class LoadSweepExperiment:
    topology_id: str = "nobel-eu"
    loads: float | tuple[float, ...] = DEFAULT_LOAD
    episodes_per_point: int = 10
    episode_length: int = 1_000
    policy_names: tuple[str, ...] = LEGACY_POLICIES
    capture_services: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    parallelism: Parallelism = Parallelism.auto()
    progress: bool = True
    progress_interval: int = 100

    def scenarios_by_value(self):
        scenarios = create_env(
            load=self.loads,
            topology_id=self.topology_id,
            episode_length=self.episode_length,
        )
        return tuple((scenario.load, scenario) for scenario in scenarios)


def run_sweep(experiment: LoadSweepExperiment | None = None, *, now: datetime | None = None):
    resolved = LoadSweepExperiment() if experiment is None else experiment
    return run_first_fit_sweep(
        script_path=SCRIPT_DIR / "graph_load.py",
        family=FAMILY,
        sweep_name="load",
        scenarios_by_value=resolved.scenarios_by_value(),
        output_dir=resolved.output_dir,
        parallelism=resolved.parallelism,
        episodes_per_point=resolved.episodes_per_point,
        policy_name=resolved.policy_names,
        now=now,
        progress=resolved.progress,
        progress_interval=resolved.progress_interval,
        capture_services=resolved.capture_services,
    )


def build_parser():
    parser = build_standard_sweep_parser(
        description="JOCN 2024 load sweep.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--request-count", type=int, default=1_000)
    parser.add_argument("--loads", type=float, nargs="*", default=None)
    parser.add_argument("--policies", nargs="*", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    loads = default_loads_for_topology(args.topology_id) if args.loads is None else tuple(args.loads)
    outputs = run_sweep(
        LoadSweepExperiment(
            topology_id=args.topology_id,
            loads=loads,
            episodes_per_point=args.episodes_per_point,
            episode_length=args.request_count,
            output_dir=args.output_dir,
            parallelism=Parallelism(workers=args.workers),
            progress=args.progress,
            progress_interval=args.progress_interval,
            capture_services=args.capture_services,
            policy_names=LEGACY_POLICIES if args.policies is None else tuple(args.policies),
        )
    )
    print(f"JOCN load sweep episodes saved to: {outputs.episodes_csv}")
    print(f"JOCN load sweep summary saved to: {outputs.summary_csv}")
    if outputs.services_csv is not None:
        print(f"JOCN load sweep services saved to: {outputs.services_csv}")


if __name__ == "__main__":
    main()
