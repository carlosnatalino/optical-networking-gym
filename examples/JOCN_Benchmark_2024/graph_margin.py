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
DEFAULT_MARGINS = (0.0, 2.0, 0.5)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


def create_env(
    margin: float,
    *,
    topology_id: str = "nobel-eu",
    load: float = 210.0,
    episode_length: int = 1_000,
    measure_disruptions: bool = True,
    drop_on_disruption: bool = False,
):
    return build_scenario(
        "jocn_benchmark",
        scenario_id=f"jocn_benchmark_{topology_id}_margin_{margin:g}",
        topology_id=topology_id,
        load=load,
        episode_length=episode_length,
        margin=float(margin),
        measure_disruptions=measure_disruptions,
        drop_on_disruption=drop_on_disruption,
    )


@dataclass(frozen=True, slots=True)
class MarginSweepExperiment:
    topology_id: str = "nobel-eu"
    load: float = 210.0
    margins: float | tuple[float, ...] = DEFAULT_MARGINS
    episodes_per_point: int = 10
    episode_length: int = 1_000
    policy_name: str = "first_fit"
    measure_disruptions: bool = True
    drop_on_disruption: bool = False
    capture_services: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    parallelism: Parallelism = Parallelism.auto()
    progress: bool = True
    progress_interval: int = 100

    def scenarios_by_value(self):
        scenarios = []
        for margin in SimulationUtils.normalize_values(self.margins):
            scenario = create_env(
                margin=margin,
                topology_id=self.topology_id,
                load=self.load,
                episode_length=self.episode_length,
                measure_disruptions=self.measure_disruptions,
                drop_on_disruption=self.drop_on_disruption,
            )
            scenarios.append((margin, scenario))
        return tuple(scenarios)


def run_sweep(experiment: MarginSweepExperiment | None = None, *, now: datetime | None = None):
    resolved = MarginSweepExperiment() if experiment is None else experiment
    return run_first_fit_sweep(
        script_path=SCRIPT_DIR / "graph_margin.py",
        family=FAMILY,
        sweep_name="margin",
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
        description="JOCN 2024 margin sweep.",
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--load", type=float, default=210.0)
    parser.add_argument("--request-count", type=int, default=1_000)
    parser.add_argument("--margins", type=float, nargs="*", default=None)
    parser.add_argument("--no-measure-disruptions", dest="measure_disruptions", action="store_false")
    parser.add_argument("--drop-on-disruption", action="store_true", default=False)
    parser.set_defaults(measure_disruptions=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    margins = DEFAULT_MARGINS if args.margins is None else tuple(args.margins)
    outputs = run_sweep(
        MarginSweepExperiment(
            topology_id=args.topology_id,
            load=args.load,
            margins=margins,
            episodes_per_point=args.episodes_per_point,
            episode_length=args.request_count,
            output_dir=args.output_dir,
            parallelism=Parallelism(workers=args.workers),
            progress=args.progress,
            progress_interval=args.progress_interval,
            capture_services=args.capture_services,
            measure_disruptions=args.measure_disruptions,
            drop_on_disruption=args.drop_on_disruption,
        )
    )
    print(f"JOCN margin sweep episodes saved to: {outputs.episodes_csv}")
    print(f"JOCN margin sweep summary saved to: {outputs.summary_csv}")
    if outputs.services_csv is not None:
        print(f"JOCN margin sweep services saved to: {outputs.services_csv}")


if __name__ == "__main__":
    main()
