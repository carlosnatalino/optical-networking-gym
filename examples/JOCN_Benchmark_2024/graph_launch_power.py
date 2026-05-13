from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from optical_networking_gym_v2 import build_scenario
from optical_networking_gym_v2.utils.experiment_utils import SimulationUtils
from optical_networking_gym_v2.utils.experiment_utils import build_standard_sweep_parser, run_first_fit_sweep
from optical_networking_gym_v2.utils.sweep_reporting import Parallelism


SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY = "JOCN_Benchmark_2024"
DEFAULT_LAUNCH_POWERS_DBM = (-8.0, 8.0, 2.0)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


def create_env(
    launch_power_dbm: float,
    *,
    topology_id: str = "nobel-eu",
    load: float = 210.0,
    episode_length: int = 1_000,
):
    return replace(
        build_scenario(
            "jocn_benchmark",
            topology_id=topology_id,
            load=load,
            episode_length=episode_length,
        ),
        scenario_id=f"jocn_benchmark_{topology_id}_launch_power_{launch_power_dbm:g}",
        launch_power_dbm=float(launch_power_dbm),
    )


@dataclass(frozen=True, slots=True)
class LaunchPowerSweepExperiment:
    topology_id: str = "nobel-eu"
    load: float = 210.0
    launch_powers_dbm: float | tuple[float, ...] = DEFAULT_LAUNCH_POWERS_DBM
    episodes_per_point: int = 10
    episode_length: int = 1_000
    policy_name: str = "first_fit"
    capture_services: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    parallelism: Parallelism = Parallelism.auto()
    progress: bool = True
    progress_interval: int = 100

    def scenarios_by_value(self):
        scenarios = []
        for launch_power_dbm in SimulationUtils.normalize_values(self.launch_powers_dbm):
            scenario = create_env(
                launch_power_dbm=launch_power_dbm,
                topology_id=self.topology_id,
                load=self.load,
                episode_length=self.episode_length,
            )
            scenarios.append((launch_power_dbm, scenario))
        return tuple(scenarios)


def run_sweep(experiment: LaunchPowerSweepExperiment | None = None, *, now: datetime | None = None):
    resolved = LaunchPowerSweepExperiment() if experiment is None else experiment
    return run_first_fit_sweep(
        script_path=SCRIPT_DIR / "graph_launch_power.py",
        family=FAMILY,
        sweep_name="launch_power_dbm",
        scenarios_by_value=resolved.scenarios_by_value(),
        output_dir=resolved.output_dir,
        parallelism=resolved.parallelism,
        episodes_per_point=resolved.episodes_per_point,
        policy_name=resolved.policy_name,
        now=now,
        progress=resolved.progress,
        progress_interval=resolved.progress_interval,
        capture_services=resolved.capture_services,
    )


def build_parser():
    parser = build_standard_sweep_parser(
        description="JOCN 2024 launch-power sweep.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--load", type=float, default=210.0)
    parser.add_argument("--request-count", type=int, default=1_000)
    parser.add_argument("--launch-powers-dbm", type=float, nargs="*", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    launch_powers = DEFAULT_LAUNCH_POWERS_DBM if args.launch_powers_dbm is None else tuple(args.launch_powers_dbm)
    outputs = run_sweep(
        LaunchPowerSweepExperiment(
            topology_id=args.topology_id,
            load=args.load,
            launch_powers_dbm=launch_powers,
            episodes_per_point=args.episodes_per_point,
            episode_length=args.request_count,
            output_dir=args.output_dir,
            parallelism=Parallelism(workers=args.workers),
            progress=args.progress,
            progress_interval=args.progress_interval,
            capture_services=args.capture_services,
        )
    )
    print(f"JOCN launch-power sweep episodes saved to: {outputs.episodes_csv}")
    print(f"JOCN launch-power sweep summary saved to: {outputs.summary_csv}")
    if outputs.services_csv is not None:
        print(f"JOCN launch-power sweep services saved to: {outputs.services_csv}")


if __name__ == "__main__":
    main()
