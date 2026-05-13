from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import math
import multiprocessing
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "optical_networking_gym_v2" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optical_networking_gym_v2 import ScenarioConfig, make_env
from optical_networking_gym_v2.defaults import DEFAULT_MEAN_HOLDING_TIME, DEFAULT_NUM_SPECTRUM_RESOURCES, DEFAULT_SEED
import optical_networking_gym_v2.utils.experiment_scenarios as scenario_utils
import optical_networking_gym_v2.utils.experiment_utils as sweep_utils
import optical_networking_gym_v2.utils.sweep_reporting as report_utils


SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_NAME = "osnr-margin-sweep"
DEFAULT_POLICY_NAME = "first_fit"
DEFAULT_LOADS = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
DEFAULT_MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_EPISODES_PER_POINT = 10
DEFAULT_EPISODE_LENGTH = 1000
DEFAULT_PILOT_LOAD_START = 200.0
DEFAULT_PILOT_LOAD_END = 400.0
DEFAULT_PILOT_LOAD_STEP = 10.0
DEFAULT_PILOT_EPISODES = 5
DEFAULT_TARGET_EFFECTIVE_BLOCKING_RATE = 0.01

PILOT_SEED_OFFSET = 0
MAIN_SWEEP_SEED_OFFSET = 100_000
SINGLE_LOAD_SEED_OFFSET = 200_000

ANALYSIS_GROUP_MULTI_LOAD = "multi_load"
ANALYSIS_GROUP_SINGLE_LOAD_FOCUS = "single_load_focus"

MODULATION_INDEX_TO_NAME = sweep_utils.build_modulation_index_to_name(
    sweep_utils.DEFAULT_MODULATION_NAMES
)
MODULATION_NAMES = tuple(name.strip() for name in sweep_utils.DEFAULT_MODULATION_NAMES.split(","))
MODULATION_RATIO_FIELDNAMES = tuple(
    f"accepted_modulation_{name.lower()}_ratio".replace("-", "_") for name in MODULATION_NAMES
)

PILOT_FIELDNAMES = [
    "date",
    "topology_id",
    "policy",
    "load",
    "episodes",
    "requests_per_episode",
    "seed_base",
    "mean_holding_time",
    "num_spectrum_resources",
    "k_paths",
    "launch_power_dbm",
    "modulations_to_consider",
    "measure_disruptions",
    "drop_on_disruption",
    "margin",
    "service_blocking_rate_mean",
    "service_blocking_rate_std",
    "service_served_rate_mean",
    "service_served_rate_std",
    "effective_blocking_rate_mean",
    "effective_blocking_rate_std",
    "target_effective_blocking_rate",
    "selection_distance",
    "is_selected_single_load",
]

EPISODE_BASE_FIELDS = [
    "date",
    "topology_id",
    "policy",
    "analysis_group",
    "is_single_load_focus",
    "selected_single_load",
    "target_effective_blocking_rate",
    "load",
    "margin",
    "episode_index",
    "episode_seed",
    "episodes_per_point",
    "requests_per_episode",
    "seed_base",
    "mean_holding_time",
    "num_spectrum_resources",
    "k_paths",
    "launch_power_dbm",
    "modulations_to_consider",
    "measure_disruptions",
    "drop_on_disruption",
]
EPISODE_METRIC_FIELDS = [
    "services_processed",
    "services_accepted",
    "services_served",
    "service_blocking_rate",
    "service_served_rate",
    "effective_blocking_rate",
    "bit_rate_blocking_rate",
    "blocked_due_to_resources",
    "blocked_due_to_osnr",
    "rejected",
    "episode_disrupted_services_count",
    "episode_disrupted_services_rate",
    "disrupted_or_dropped_services",
    "drop_after_accept_rate",
    "accepted_osnr_margin_mean",
    "final_osnr_margin_mean",
    "fragmentation_shannon_entropy_mean",
    "fragmentation_route_cuts_mean",
    "fragmentation_route_rss_mean",
    "episode_time_s",
    *MODULATION_INDEX_TO_NAME.values(),
]
EPISODE_FIELDNAMES = EPISODE_BASE_FIELDS + EPISODE_METRIC_FIELDS

SUMMARY_BASE_FIELDS = [
    "date",
    "topology_id",
    "policy",
    "analysis_group",
    "is_single_load_focus",
    "selected_single_load",
    "target_effective_blocking_rate",
    "load",
    "margin",
    "episodes",
    "requests_per_episode",
    "seed_base",
    "mean_holding_time",
    "num_spectrum_resources",
    "k_paths",
    "launch_power_dbm",
    "modulations_to_consider",
    "measure_disruptions",
    "drop_on_disruption",
]
SUMMARY_AGGREGATIONS = (
    ("services_processed", "services_processed"),
    ("services_accepted", "services_accepted"),
    ("services_served", "services_served"),
    ("service_blocking_rate", "service_blocking_rate"),
    ("service_served_rate", "service_served_rate"),
    ("effective_blocking_rate", "effective_blocking_rate"),
    ("bit_rate_blocking_rate", "bit_rate_blocking_rate"),
    ("blocked_due_to_resources", "blocked_due_to_resources"),
    ("blocked_due_to_osnr", "blocked_due_to_osnr"),
    ("rejected", "rejected"),
    ("episode_disrupted_services_count", "episode_disrupted_services_count"),
    ("episode_disrupted_services_rate", "episode_disrupted_services_rate"),
    ("disrupted_or_dropped_services", "disrupted_or_dropped_services"),
    ("drop_after_accept_rate", "drop_after_accept_rate"),
    ("accepted_osnr_margin_mean", "accepted_osnr_margin"),
    ("final_osnr_margin_mean", "final_osnr_margin"),
    ("fragmentation_shannon_entropy_mean", "fragmentation_shannon_entropy"),
    ("fragmentation_route_cuts_mean", "fragmentation_route_cuts"),
    ("fragmentation_route_rss_mean", "fragmentation_route_rss"),
    ("episode_time_s", "episode_time_s"),
)
SUMMARY_FIELDNAMES = SUMMARY_BASE_FIELDS + [
    fieldname
    for _episode_field, summary_name in SUMMARY_AGGREGATIONS
    for fieldname in (f"{summary_name}_mean", f"{summary_name}_std")
] + [
    "accepted_osnr_margin_p50",
    "accepted_osnr_margin_p95",
    "final_osnr_margin_p50",
    "final_osnr_margin_p95",
    *MODULATION_RATIO_FIELDNAMES,
    "is_best_margin_for_load",
    "best_margin_rank_within_load",
]

REQUEST_FIELDNAMES = [
    "date",
    "topology_id",
    "policy",
    "analysis_group",
    "is_single_load_focus",
    "selected_single_load",
    "load",
    "margin",
    "episode_index",
    "episode_seed",
    "request_index",
    "service_id",
    "source_id",
    "destination_id",
    "bit_rate",
    "status",
    "accepted",
    "osnr",
    "osnr_req",
    "osnr_margin",
    "chosen_path_index",
    "chosen_slot",
    "chosen_modulation_index",
    "modulation_name",
    "fragmentation_shannon_entropy",
    "fragmentation_route_cuts",
    "fragmentation_route_rss",
]


@dataclass(frozen=True, slots=True)
class LegacyBenchmarkMarginSweepExperiment:
    topology_id: str = "nobel-eu"
    policy_name: str = DEFAULT_POLICY_NAME
    loads: tuple[float, ...] = DEFAULT_LOADS
    margins: tuple[float, ...] = DEFAULT_MARGINS
    episodes_per_point: int = DEFAULT_EPISODES_PER_POINT
    episode_length: int = DEFAULT_EPISODE_LENGTH
    seed: int = DEFAULT_SEED
    mean_holding_time: float = DEFAULT_MEAN_HOLDING_TIME
    num_spectrum_resources: int = DEFAULT_NUM_SPECTRUM_RESOURCES
    k_paths: int = scenario_utils.LEGACY_BENCHMARK_K_PATHS
    launch_power_dbm: float = scenario_utils.LEGACY_BENCHMARK_LAUNCH_POWER_DBM
    modulations_to_consider: int = scenario_utils.LEGACY_BENCHMARK_MODULATIONS_TO_CONSIDER
    pilot_load_start: float = DEFAULT_PILOT_LOAD_START
    pilot_load_end: float = DEFAULT_PILOT_LOAD_END
    pilot_load_step: float = DEFAULT_PILOT_LOAD_STEP
    pilot_episodes: int = DEFAULT_PILOT_EPISODES
    target_effective_blocking_rate: float = DEFAULT_TARGET_EFFECTIVE_BLOCKING_RATE
    processes: int = 1
    measure_disruptions: bool = True
    drop_on_disruption: bool = True
    output_dir: Path = SCRIPT_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "loads", tuple(float(load) for load in self.loads))
        object.__setattr__(self, "margins", tuple(float(margin) for margin in self.margins))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if not self.topology_id:
            raise ValueError("topology_id must be a non-empty string")
        if self.policy_name != DEFAULT_POLICY_NAME:
            raise ValueError(f"unsupported policy {self.policy_name!r}; supported values: {DEFAULT_POLICY_NAME}")
        if not self.loads:
            raise ValueError("loads must be a non-empty sequence")
        if any(load <= 0 for load in self.loads):
            raise ValueError("loads must contain only positive values")
        if not self.margins:
            raise ValueError("margins must be a non-empty sequence")
        if any(margin < 0 for margin in self.margins):
            raise ValueError("margins must contain only non-negative values")
        if self.episodes_per_point <= 0:
            raise ValueError("episodes_per_point must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.mean_holding_time <= 0:
            raise ValueError("mean_holding_time must be positive")
        if self.num_spectrum_resources <= 0:
            raise ValueError("num_spectrum_resources must be positive")
        if self.k_paths <= 0:
            raise ValueError("k_paths must be positive")
        if self.modulations_to_consider <= 0:
            raise ValueError("modulations_to_consider must be positive")
        if self.pilot_load_start <= 0 or self.pilot_load_end <= 0:
            raise ValueError("pilot loads must be positive")
        if self.pilot_load_end < self.pilot_load_start:
            raise ValueError("pilot_load_end must be >= pilot_load_start")
        if self.pilot_load_step <= 0:
            raise ValueError("pilot_load_step must be positive")
        if self.pilot_episodes <= 0:
            raise ValueError("pilot_episodes must be positive")
        if self.target_effective_blocking_rate < 0:
            raise ValueError("target_effective_blocking_rate must be non-negative")
        if self.processes <= 0:
            raise ValueError("processes must be positive")

    def pilot_loads(self) -> tuple[float, ...]:
        return build_load_sequence(
            start=self.pilot_load_start,
            end=self.pilot_load_end,
            step=self.pilot_load_step,
        )


@dataclass(frozen=True, slots=True)
class LegacyBenchmarkMarginSweepOutputs:
    run_dir: Path
    pilot_summary_csv: Path
    episodes_csv: Path
    summary_csv: Path
    requests_csv: Path
    selected_single_load: float


@dataclass(frozen=True, slots=True)
class EpisodeRunResult:
    episode_row: dict[str, report_utils.Scalar]
    request_rows: list[dict[str, report_utils.Scalar]]
    accepted_osnr_margins: list[float]
    final_osnr_margins: list[float]


@dataclass(frozen=True, slots=True)
class PointRunResult:
    sort_index: int
    load: float
    margin: float
    analysis_group: str
    is_single_load_focus: bool
    episode_rows: list[dict[str, report_utils.Scalar]]
    request_rows: list[dict[str, report_utils.Scalar]]
    accepted_osnr_margins: list[float]
    final_osnr_margins: list[float]


@dataclass(frozen=True, slots=True)
class PointTask:
    sort_index: int
    experiment: LegacyBenchmarkMarginSweepExperiment
    load: float
    margin: float
    load_index: int
    margin_index: int
    episodes_per_point: int
    analysis_group: str
    is_single_load_focus: bool
    selected_single_load: float
    seed_offset: int
    date_label: str


def build_load_sequence(*, start: float, end: float, step: float) -> tuple[float, ...]:
    if step <= 0:
        raise ValueError("step must be positive")
    values: list[float] = []
    current = float(start)
    while current <= float(end) + 1e-9:
        values.append(round(current, 10))
        current += float(step)
    if not values:
        raise ValueError("load sweep produced no values")
    return tuple(values)


def select_single_load_from_pilot_rows(
    rows: list[dict[str, report_utils.Scalar]],
    *,
    target_rate: float,
) -> float:
    if not rows:
        raise ValueError("pilot rows must be non-empty")
    selected = min(
        rows,
        key=lambda row: (
            round(abs(float(row["effective_blocking_rate_mean"]) - float(target_rate)), 12),
            -float(row["load"]),
        ),
    )
    return float(selected["load"])


def build_base_scenario(
    *,
    experiment: LegacyBenchmarkMarginSweepExperiment,
    load: float,
    margin: float,
    seed: int,
) -> ScenarioConfig:
    scenario = scenario_utils.build_legacy_benchmark_scenario(
        topology_id=experiment.topology_id,
        episode_length=experiment.episode_length,
        seed=seed,
        load=load,
        mean_holding_time=experiment.mean_holding_time,
        num_spectrum_resources=experiment.num_spectrum_resources,
        k_paths=experiment.k_paths,
        launch_power_dbm=experiment.launch_power_dbm,
        modulations_to_consider=experiment.modulations_to_consider,
        margin=margin,
        measure_disruptions=experiment.measure_disruptions,
        drop_on_disruption=experiment.drop_on_disruption,
    )
    return replace(
        scenario,
        scenario_id=f"{experiment.topology_id}_legacy_margin_{float(margin):g}_load_{float(load):g}_seed{seed}",
    )


def build_env(*, scenario: ScenarioConfig):
    return make_env(config=scenario)


def _date_prefix(now: datetime | None = None) -> str:
    current = datetime.now() if now is None else now
    return current.strftime("%d-%m-%Hh%M")


def _build_output_paths(
    *,
    output_dir: Path,
    now: datetime | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    run_dir = Path(output_dir) / f"{_date_prefix(now)}-{SWEEP_NAME}"
    return (
        run_dir,
        run_dir / "pilot-summary.csv",
        run_dir / "episodes.csv",
        run_dir / "summary.csv",
        run_dir / "requests.csv",
    )


def _episode_seed(
    *,
    experiment: LegacyBenchmarkMarginSweepExperiment,
    load_index: int,
    margin_index: int,
    episode_index: int,
    episodes_per_point: int,
    margin_count: int,
    seed_offset: int,
) -> int:
    return int(
        experiment.seed
        + seed_offset
        + load_index * margin_count * episodes_per_point
        + margin_index * episodes_per_point
        + episode_index
    )


def _same_load(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _modulation_name_for_index(scenario: ScenarioConfig, modulation_index: int | None) -> str:
    if modulation_index is None:
        return ""
    if modulation_index < 0 or modulation_index >= len(scenario.modulations):
        return ""
    return str(scenario.modulations[modulation_index].name)


def _accepted_modulation_ratios(
    request_rows: list[dict[str, report_utils.Scalar]],
) -> dict[str, float]:
    counts = {fieldname: 0.0 for fieldname in MODULATION_RATIO_FIELDNAMES}
    accepted_rows = [row for row in request_rows if bool(row["accepted"])]
    if not accepted_rows:
        return counts

    for row in accepted_rows:
        modulation_name = str(row["modulation_name"]).strip().lower()
        fieldname = f"accepted_modulation_{modulation_name}_ratio".replace("-", "_")
        if fieldname in counts:
            counts[fieldname] += 1.0

    total = float(len(accepted_rows))
    return {fieldname: count / total for fieldname, count in counts.items()}


def run_single_episode(
    *,
    experiment: LegacyBenchmarkMarginSweepExperiment,
    scenario: ScenarioConfig,
    analysis_group: str,
    is_single_load_focus: bool,
    selected_single_load: float,
    episode_index: int,
    episodes_per_point: int,
    date_label: str,
) -> EpisodeRunResult:
    if experiment.policy_name != DEFAULT_POLICY_NAME:
        raise ValueError(f"unsupported policy {experiment.policy_name!r}; supported values: {DEFAULT_POLICY_NAME}")

    env = build_env(scenario=scenario)
    episode_seed = int(scenario.seed or 0)
    _, info = env.reset(seed=episode_seed)

    request_rows: list[dict[str, report_utils.Scalar]] = []
    accepted_osnr_margins: list[float] = []
    fragmentation_shannon_values: list[float] = []
    fragmentation_route_cuts_values: list[float] = []
    fragmentation_route_rss_values: list[float] = []

    started_at = time.perf_counter()
    while True:
        simulator = env.simulator
        current_request = simulator.current_request
        if current_request is None:
            raise RuntimeError("simulator current_request is unexpectedly unavailable")

        action = sweep_utils.select_masked_first_fit_policy(env, info)
        _, _, terminated, truncated, info = env.step(action)

        accepted = bool(info.get("accepted", False))
        osnr = float(info.get("osnr", 0.0))
        osnr_req = float(info.get("osnr_req", 0.0))
        osnr_margin = float(osnr - osnr_req)
        modulation_index_raw = info.get("chosen_modulation_index")
        modulation_index = "" if modulation_index_raw is None else int(modulation_index_raw)
        modulation_name = _modulation_name_for_index(
            scenario,
            None if modulation_index_raw is None else int(modulation_index_raw),
        )
        fragmentation_shannon_entropy = float(info.get("fragmentation_shannon_entropy", 0.0))
        fragmentation_route_cuts = float(info.get("fragmentation_route_cuts", 0.0))
        fragmentation_route_rss = float(info.get("fragmentation_route_rss", 0.0))

        request_rows.append(
            {
                "date": date_label,
                "topology_id": scenario.topology_id,
                "policy": experiment.policy_name,
                "analysis_group": analysis_group,
                "is_single_load_focus": bool(is_single_load_focus),
                "selected_single_load": float(selected_single_load),
                "load": float(scenario.load),
                "margin": float(scenario.margin),
                "episode_index": int(episode_index),
                "episode_seed": episode_seed,
                "request_index": int(current_request.request_index),
                "service_id": int(current_request.service_id),
                "source_id": int(current_request.source_id),
                "destination_id": int(current_request.destination_id),
                "bit_rate": int(current_request.bit_rate),
                "status": str(info.get("status", "")),
                "accepted": bool(accepted),
                "osnr": float(osnr),
                "osnr_req": float(osnr_req),
                "osnr_margin": float(osnr_margin),
                "chosen_path_index": info.get("chosen_path_index", ""),
                "chosen_slot": info.get("chosen_slot", ""),
                "chosen_modulation_index": modulation_index,
                "modulation_name": modulation_name,
                "fragmentation_shannon_entropy": float(fragmentation_shannon_entropy),
                "fragmentation_route_cuts": float(fragmentation_route_cuts),
                "fragmentation_route_rss": float(fragmentation_route_rss),
            }
        )

        if accepted:
            accepted_osnr_margins.append(osnr_margin)
            fragmentation_shannon_values.append(fragmentation_shannon_entropy)
            fragmentation_route_cuts_values.append(fragmentation_route_cuts)
            fragmentation_route_rss_values.append(fragmentation_route_rss)

        if terminated or truncated:
            break

    episode_time_s = time.perf_counter() - started_at
    simulator = env.simulator
    if simulator.statistics is None:
        raise RuntimeError("simulator statistics are not available after the episode")

    snapshot = simulator.statistics.snapshot()
    active_services = () if simulator.state is None else simulator.state.active_services_by_id.values()
    final_osnr_margins = [
        float(service.osnr - (float(service.modulation.minimum_osnr) + float(scenario.margin)))
        for service in active_services
        if service.modulation is not None
    ]

    services_accepted = int(snapshot.episode_services_accepted)
    disrupted_or_dropped_services = int(snapshot.episode_services_dropped_qot)
    episode_row: dict[str, report_utils.Scalar] = {
        "date": date_label,
        "topology_id": scenario.topology_id,
        "policy": experiment.policy_name,
        "analysis_group": analysis_group,
        "is_single_load_focus": bool(is_single_load_focus),
        "selected_single_load": float(selected_single_load),
        "target_effective_blocking_rate": float(experiment.target_effective_blocking_rate),
        "load": float(scenario.load),
        "margin": float(scenario.margin),
        "episode_index": int(episode_index),
        "episode_seed": episode_seed,
        "episodes_per_point": int(episodes_per_point),
        "requests_per_episode": int(scenario.episode_length),
        "seed_base": int(experiment.seed),
        "mean_holding_time": float(scenario.mean_holding_time),
        "num_spectrum_resources": int(scenario.num_spectrum_resources),
        "k_paths": int(scenario.k_paths),
        "launch_power_dbm": float(scenario.launch_power_dbm),
        "modulations_to_consider": int(scenario.modulations_to_consider),
        "measure_disruptions": bool(scenario.measure_disruptions),
        "drop_on_disruption": bool(scenario.drop_on_disruption),
        "services_processed": int(snapshot.episode_services_processed),
        "services_accepted": services_accepted,
        "services_served": int(snapshot.episode_services_served),
        "service_blocking_rate": float(snapshot.episode_service_blocking_rate),
        "service_served_rate": float(snapshot.episode_service_served_rate),
        "effective_blocking_rate": float(1.0 - snapshot.episode_service_served_rate),
        "bit_rate_blocking_rate": float(snapshot.episode_bit_rate_blocking_rate),
        "blocked_due_to_resources": int(snapshot.episode_services_blocked_resources),
        "blocked_due_to_osnr": int(snapshot.episode_services_blocked_qot),
        "rejected": int(snapshot.episode_services_rejected_by_agent),
        "episode_disrupted_services_count": int(snapshot.episode_disrupted_services),
        "episode_disrupted_services_rate": float(snapshot.episode_disrupted_services_rate),
        "disrupted_or_dropped_services": disrupted_or_dropped_services,
        "drop_after_accept_rate": _safe_ratio(disrupted_or_dropped_services, services_accepted),
        "accepted_osnr_margin_mean": sweep_utils.float_mean(accepted_osnr_margins),
        "final_osnr_margin_mean": sweep_utils.float_mean(final_osnr_margins),
        "fragmentation_shannon_entropy_mean": sweep_utils.float_mean(fragmentation_shannon_values),
        "fragmentation_route_cuts_mean": sweep_utils.float_mean(fragmentation_route_cuts_values),
        "fragmentation_route_rss_mean": sweep_utils.float_mean(fragmentation_route_rss_values),
        "episode_time_s": float(episode_time_s),
    }
    episode_row.update(
        sweep_utils.episode_modulation_counts(
            snapshot,
            modulation_index_to_name=MODULATION_INDEX_TO_NAME,
        )
    )

    return EpisodeRunResult(
        episode_row=episode_row,
        request_rows=request_rows,
        accepted_osnr_margins=accepted_osnr_margins,
        final_osnr_margins=final_osnr_margins,
    )


def run_point(
    *,
    experiment: LegacyBenchmarkMarginSweepExperiment,
    load: float,
    margin: float,
    load_index: int,
    margin_index: int,
    episodes_per_point: int,
    analysis_group: str,
    is_single_load_focus: bool,
    selected_single_load: float,
    seed_offset: int,
    date_label: str,
) -> PointRunResult:
    episode_rows: list[dict[str, report_utils.Scalar]] = []
    request_rows: list[dict[str, report_utils.Scalar]] = []
    accepted_osnr_margins: list[float] = []
    final_osnr_margins: list[float] = []

    for episode_index in range(episodes_per_point):
        episode_seed = _episode_seed(
            experiment=experiment,
            load_index=load_index,
            margin_index=margin_index,
            episode_index=episode_index,
            episodes_per_point=episodes_per_point,
            margin_count=max(1, len(experiment.margins)),
            seed_offset=seed_offset,
        )
        scenario = build_base_scenario(
            experiment=experiment,
            load=load,
            margin=margin,
            seed=episode_seed,
        )
        episode_result = run_single_episode(
            experiment=experiment,
            scenario=scenario,
            analysis_group=analysis_group,
            is_single_load_focus=is_single_load_focus,
            selected_single_load=selected_single_load,
            episode_index=episode_index,
            episodes_per_point=episodes_per_point,
            date_label=date_label,
        )
        episode_rows.append(episode_result.episode_row)
        request_rows.extend(episode_result.request_rows)
        accepted_osnr_margins.extend(episode_result.accepted_osnr_margins)
        final_osnr_margins.extend(episode_result.final_osnr_margins)

    return PointRunResult(
        sort_index=load_index * max(1, len(experiment.margins)) + margin_index,
        load=float(load),
        margin=float(margin),
        analysis_group=analysis_group,
        is_single_load_focus=bool(is_single_load_focus),
        episode_rows=episode_rows,
        request_rows=request_rows,
        accepted_osnr_margins=accepted_osnr_margins,
        final_osnr_margins=final_osnr_margins,
    )


def _run_point_task(task: PointTask) -> PointRunResult:
    point_result = run_point(
        experiment=task.experiment,
        load=task.load,
        margin=task.margin,
        load_index=task.load_index,
        margin_index=task.margin_index,
        episodes_per_point=task.episodes_per_point,
        analysis_group=task.analysis_group,
        is_single_load_focus=task.is_single_load_focus,
        selected_single_load=task.selected_single_load,
        seed_offset=task.seed_offset,
        date_label=task.date_label,
    )
    return replace(point_result, sort_index=task.sort_index)


def _run_point_tasks(tasks: list[PointTask], *, processes: int) -> list[PointRunResult]:
    if not tasks:
        return []
    if processes <= 1:
        return [_run_point_task(task) for task in tasks]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=processes) as pool:
        results = pool.map(_run_point_task, tasks)
    return list(results)


def _build_pilot_row(
    *,
    experiment: LegacyBenchmarkMarginSweepExperiment,
    load: float,
    date_label: str,
    episode_rows: list[dict[str, report_utils.Scalar]],
) -> dict[str, report_utils.Scalar]:
    aggregated = report_utils.aggregate_summary_metrics(
        episode_rows,
        metric_names=("service_blocking_rate", "service_served_rate", "effective_blocking_rate"),
    )
    return {
        "date": date_label,
        "topology_id": experiment.topology_id,
        "policy": experiment.policy_name,
        "load": float(load),
        "episodes": int(experiment.pilot_episodes),
        "requests_per_episode": int(experiment.episode_length),
        "seed_base": int(experiment.seed),
        "mean_holding_time": float(experiment.mean_holding_time),
        "num_spectrum_resources": int(experiment.num_spectrum_resources),
        "k_paths": int(experiment.k_paths),
        "launch_power_dbm": float(experiment.launch_power_dbm),
        "modulations_to_consider": int(experiment.modulations_to_consider),
        "measure_disruptions": bool(experiment.measure_disruptions),
        "drop_on_disruption": bool(experiment.drop_on_disruption),
        "margin": 0.0,
        "service_blocking_rate_mean": float(aggregated["service_blocking_rate_mean"]),
        "service_blocking_rate_std": float(aggregated["service_blocking_rate_std"]),
        "service_served_rate_mean": float(aggregated["service_served_rate_mean"]),
        "service_served_rate_std": float(aggregated["service_served_rate_std"]),
        "effective_blocking_rate_mean": float(aggregated["effective_blocking_rate_mean"]),
        "effective_blocking_rate_std": float(aggregated["effective_blocking_rate_std"]),
        "target_effective_blocking_rate": float(experiment.target_effective_blocking_rate),
        "selection_distance": 0.0,
        "is_selected_single_load": False,
    }


def _build_summary_row(
    *,
    experiment: LegacyBenchmarkMarginSweepExperiment,
    point_result: PointRunResult,
    date_label: str,
    selected_single_load: float,
) -> dict[str, report_utils.Scalar]:
    row: dict[str, report_utils.Scalar] = {
        "date": date_label,
        "topology_id": experiment.topology_id,
        "policy": experiment.policy_name,
        "analysis_group": point_result.analysis_group,
        "is_single_load_focus": bool(point_result.is_single_load_focus),
        "selected_single_load": float(selected_single_load),
        "target_effective_blocking_rate": float(experiment.target_effective_blocking_rate),
        "load": float(point_result.load),
        "margin": float(point_result.margin),
        "episodes": int(len(point_result.episode_rows)),
        "requests_per_episode": int(experiment.episode_length),
        "seed_base": int(experiment.seed),
        "mean_holding_time": float(experiment.mean_holding_time),
        "num_spectrum_resources": int(experiment.num_spectrum_resources),
        "k_paths": int(experiment.k_paths),
        "launch_power_dbm": float(experiment.launch_power_dbm),
        "modulations_to_consider": int(experiment.modulations_to_consider),
        "measure_disruptions": bool(experiment.measure_disruptions),
        "drop_on_disruption": bool(experiment.drop_on_disruption),
    }
    for episode_field, summary_name in SUMMARY_AGGREGATIONS:
        values = [float(episode_row[episode_field]) for episode_row in point_result.episode_rows]
        row[f"{summary_name}_mean"] = sweep_utils.float_mean(values)
        row[f"{summary_name}_std"] = sweep_utils.float_std(values)
    row["accepted_osnr_margin_p50"] = sweep_utils.float_percentile(point_result.accepted_osnr_margins, 50.0)
    row["accepted_osnr_margin_p95"] = sweep_utils.float_percentile(point_result.accepted_osnr_margins, 95.0)
    row["final_osnr_margin_p50"] = sweep_utils.float_percentile(point_result.final_osnr_margins, 50.0)
    row["final_osnr_margin_p95"] = sweep_utils.float_percentile(point_result.final_osnr_margins, 95.0)
    row.update(_accepted_modulation_ratios(point_result.request_rows))
    row["is_best_margin_for_load"] = False
    row["best_margin_rank_within_load"] = 0
    return row


def _annotate_best_margins(summary_rows: list[dict[str, report_utils.Scalar]]) -> None:
    by_load: dict[float, list[dict[str, report_utils.Scalar]]] = {}
    for row in summary_rows:
        by_load.setdefault(float(row["load"]), []).append(row)

    for load_rows in by_load.values():
        ordered = sorted(
            load_rows,
            key=lambda row: (
                float(row["effective_blocking_rate_mean"]),
                float(row["bit_rate_blocking_rate_mean"]),
                -float(row["accepted_osnr_margin_mean"]),
                float(row["margin"]),
            ),
        )
        for index, row in enumerate(ordered, start=1):
            row["is_best_margin_for_load"] = index == 1
            row["best_margin_rank_within_load"] = int(index)


def run_margin_sweep(
    experiment: LegacyBenchmarkMarginSweepExperiment | None = None,
    *,
    now: datetime | None = None,
) -> LegacyBenchmarkMarginSweepOutputs:
    resolved_experiment = LegacyBenchmarkMarginSweepExperiment() if experiment is None else experiment
    date_label = _date_prefix(now)

    pilot_tasks = [
        PointTask(
            sort_index=load_index,
            experiment=resolved_experiment,
            load=float(load),
            margin=0.0,
            load_index=load_index,
            margin_index=0,
            episodes_per_point=resolved_experiment.pilot_episodes,
            analysis_group="pilot",
            is_single_load_focus=False,
            selected_single_load=-1.0,
            seed_offset=PILOT_SEED_OFFSET,
            date_label=date_label,
        )
        for load_index, load in enumerate(resolved_experiment.pilot_loads())
    ]
    pilot_results = sorted(
        _run_point_tasks(pilot_tasks, processes=resolved_experiment.processes),
        key=lambda result: result.sort_index,
    )
    pilot_rows: list[dict[str, report_utils.Scalar]] = []
    for pilot_result in pilot_results:
        pilot_rows.append(
            _build_pilot_row(
                experiment=resolved_experiment,
                load=float(pilot_result.load),
                date_label=date_label,
                episode_rows=pilot_result.episode_rows,
            )
        )

    selected_single_load = select_single_load_from_pilot_rows(
        pilot_rows,
        target_rate=resolved_experiment.target_effective_blocking_rate,
    )
    for row in pilot_rows:
        row["selection_distance"] = abs(
            float(row["effective_blocking_rate_mean"]) - float(resolved_experiment.target_effective_blocking_rate)
        )
        row["is_selected_single_load"] = _same_load(float(row["load"]), selected_single_load)

    episode_rows: list[dict[str, report_utils.Scalar]] = []
    summary_rows: list[dict[str, report_utils.Scalar]] = []
    request_rows: list[dict[str, report_utils.Scalar]] = []

    main_tasks: list[PointTask] = []
    for load_index, load in enumerate(resolved_experiment.loads):
        for margin_index, margin in enumerate(resolved_experiment.margins):
            main_tasks.append(
                PointTask(
                    sort_index=load_index * max(1, len(resolved_experiment.margins)) + margin_index,
                    experiment=resolved_experiment,
                    load=float(load),
                    margin=float(margin),
                    load_index=load_index,
                    margin_index=margin_index,
                    episodes_per_point=resolved_experiment.episodes_per_point,
                    analysis_group=ANALYSIS_GROUP_MULTI_LOAD,
                    is_single_load_focus=_same_load(load, selected_single_load),
                    selected_single_load=selected_single_load,
                    seed_offset=MAIN_SWEEP_SEED_OFFSET,
                    date_label=date_label,
                )
            )
    main_results = sorted(
        _run_point_tasks(main_tasks, processes=resolved_experiment.processes),
        key=lambda result: result.sort_index,
    )
    for point_result in main_results:
        episode_rows.extend(point_result.episode_rows)
        summary_rows.append(
            _build_summary_row(
                experiment=resolved_experiment,
                point_result=point_result,
                date_label=date_label,
                selected_single_load=selected_single_load,
            )
        )
        request_rows.extend(point_result.request_rows)

    if not any(_same_load(load, selected_single_load) for load in resolved_experiment.loads):
        single_load_index = len(resolved_experiment.loads)
        single_load_tasks = [
            PointTask(
                sort_index=single_load_index * max(1, len(resolved_experiment.margins)) + margin_index,
                experiment=resolved_experiment,
                load=float(selected_single_load),
                margin=float(margin),
                load_index=single_load_index,
                margin_index=margin_index,
                episodes_per_point=resolved_experiment.episodes_per_point,
                analysis_group=ANALYSIS_GROUP_SINGLE_LOAD_FOCUS,
                is_single_load_focus=True,
                selected_single_load=selected_single_load,
                seed_offset=SINGLE_LOAD_SEED_OFFSET,
                date_label=date_label,
            )
            for margin_index, margin in enumerate(resolved_experiment.margins)
        ]
        single_load_results = sorted(
            _run_point_tasks(single_load_tasks, processes=resolved_experiment.processes),
            key=lambda result: result.sort_index,
        )
        for point_result in single_load_results:
            episode_rows.extend(point_result.episode_rows)
            summary_rows.append(
                _build_summary_row(
                    experiment=resolved_experiment,
                    point_result=point_result,
                    date_label=date_label,
                    selected_single_load=selected_single_load,
                )
            )
            request_rows.extend(point_result.request_rows)

    _annotate_best_margins(summary_rows)

    run_dir, pilot_summary_csv, episodes_csv, summary_csv, requests_csv = _build_output_paths(
        output_dir=resolved_experiment.output_dir,
        now=now,
    )
    report_utils.write_csv_rows(
        path=pilot_summary_csv,
        fieldnames=PILOT_FIELDNAMES,
        rows=pilot_rows,
    )
    report_utils.write_csv_rows(
        path=episodes_csv,
        fieldnames=EPISODE_FIELDNAMES,
        rows=episode_rows,
    )
    report_utils.write_csv_rows(
        path=summary_csv,
        fieldnames=SUMMARY_FIELDNAMES,
        rows=summary_rows,
    )
    report_utils.write_csv_rows(
        path=requests_csv,
        fieldnames=REQUEST_FIELDNAMES,
        rows=request_rows,
    )
    return LegacyBenchmarkMarginSweepOutputs(
        run_dir=run_dir,
        pilot_summary_csv=pilot_summary_csv,
        episodes_csv=episodes_csv,
        summary_csv=summary_csv,
        requests_csv=requests_csv,
        selected_single_load=float(selected_single_load),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legacy benchmark OSNR margin sweep with first-fit.")
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--loads", type=float, nargs="*", default=None)
    parser.add_argument("--margins", type=float, nargs="*", default=None)
    parser.add_argument("--episodes-per-point", type=int, default=DEFAULT_EPISODES_PER_POINT)
    parser.add_argument("--pilot-load-start", type=float, default=DEFAULT_PILOT_LOAD_START)
    parser.add_argument("--pilot-load-end", type=float, default=DEFAULT_PILOT_LOAD_END)
    parser.add_argument("--pilot-load-step", type=float, default=DEFAULT_PILOT_LOAD_STEP)
    parser.add_argument("--pilot-episodes", type=int, default=DEFAULT_PILOT_EPISODES)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--request-count", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser


def main() -> None:
    multiprocessing.freeze_support()
    args = build_parser().parse_args()
    experiment = LegacyBenchmarkMarginSweepExperiment(
        topology_id=args.topology_id,
        loads=DEFAULT_LOADS if args.loads is None else tuple(args.loads),
        margins=DEFAULT_MARGINS if args.margins is None else tuple(args.margins),
        episodes_per_point=args.episodes_per_point,
        episode_length=args.request_count,
        pilot_load_start=args.pilot_load_start,
        pilot_load_end=args.pilot_load_end,
        pilot_load_step=args.pilot_load_step,
        pilot_episodes=args.pilot_episodes,
        processes=args.processes,
        output_dir=args.output_dir,
    )
    outputs = run_margin_sweep(experiment=experiment)
    print(f"Legacy benchmark run directory: {outputs.run_dir}")
    print(f"Pilot summary saved to: {outputs.pilot_summary_csv}")
    print(f"Episode metrics saved to: {outputs.episodes_csv}")
    print(f"Summary metrics saved to: {outputs.summary_csv}")
    print(f"Request diagnostics saved to: {outputs.requests_csv}")
    print(f"Selected single-load focus: {outputs.selected_single_load:g}")


__all__ = [
    "LegacyBenchmarkMarginSweepExperiment",
    "LegacyBenchmarkMarginSweepOutputs",
    "build_base_scenario",
    "build_env",
    "build_load_sequence",
    "run_margin_sweep",
    "run_point",
    "run_single_episode",
    "select_single_load_from_pilot_rows",
]


if __name__ == "__main__":
    main()
