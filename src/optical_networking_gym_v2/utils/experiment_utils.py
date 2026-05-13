from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

from optical_networking_gym_v2 import (
    BUILTIN_TOPOLOGY_DIR,
    ScenarioConfig,
    get_modulations,
    make_env,
    select_disruption_aware_first_fit_action,
    select_first_fit_action,
    select_highest_snr_first_fit_runtime_action,
    select_jocn_ls_bm_ksp_action,
    select_jocn_bm_ksp_lb_action,
    select_jocn_ksp_lb_bm_action,
    select_ksp_best_mod_last_fit_runtime_action,
    select_load_balancing_runtime_action,
    select_lowest_fragmentation_runtime_action,
    select_random_runtime_action,
)


DEFAULT_MODULATION_NAMES = "BPSK,QPSK,8QAM,16QAM,32QAM,64QAM"
DEFAULT_BIT_RATES = (10, 40, 100, 400)
DEFAULT_FREQUENCY_START = 3e8 / 1565e-9
DEFAULT_FREQUENCY_SLOT_BANDWIDTH = 12.5e9
DEFAULT_MAX_SPAN_LENGTH_KM = 80.0
DEFAULT_ATTENUATION_DB_PER_KM = 0.2
DEFAULT_NOISE_FIGURE_DB = 4.5

EpisodePolicy = Callable[[object, dict[str, object]], int]


@dataclass(frozen=True, slots=True)
class StandardSweepOutputs:
    run_dir: Path
    episodes_csv: Path
    summary_csv: Path
    metadata_json: Path
    services_csv: Path | None = None


@dataclass(frozen=True, slots=True)
class SweepCaseResult:
    episode_row: dict[str, object]
    service_rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class SweepCase:
    scenario: ScenarioConfig
    sweep_name: str
    sweep_value: float
    episode_index: int
    date_label: str
    policy_name: str


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    key: tuple[str, ...]
    metric: str
    reference_value: float
    candidate_value: float
    tolerance: float
    absolute_delta: float
    passed: bool


class StepProgress(AbstractContextManager["StepProgress"]):
    def __init__(self, *, total: int, description: str, interval: int, enabled: bool) -> None:
        if interval <= 0:
            raise ValueError("progress_interval must be positive")
        self._total = int(total)
        self._description = description
        self._interval = int(interval)
        self._enabled = bool(enabled)
        self._pending = 0
        self._bar = nullcontext()
        self._progress = None

    def __enter__(self) -> "StepProgress":
        if not self._enabled:
            return self
        try:
            from tqdm import tqdm
        except ImportError:
            return self
        self._progress = tqdm(
            total=self._total,
            desc=self._description,
            unit="step",
            mininterval=0.25,
            file=sys.stderr,
        )
        self._bar = self._progress
        self._bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        self.flush()
        return self._bar.__exit__(exc_type, exc_value, traceback)

    def update(self, steps: int = 1) -> None:
        if steps <= 0:
            return
        self._pending += int(steps)
        if self._pending >= self._interval:
            self.flush()

    def flush(self) -> None:
        if self._pending <= 0:
            return
        if self._progress is not None:
            self._progress.update(self._pending)
        self._pending = 0


class SimulationUtils:
    @staticmethod
    def normalize_values(value: float | int | tuple[float, ...] | list[float]) -> tuple[float, ...]:
        if isinstance(value, int | float):
            return (float(value),)
        values = tuple(float(item) for item in value)
        if len(values) == 3:
            start, stop, step = values
            if step <= 0:
                raise ValueError("range step must be positive")
            expanded: list[float] = []
            current = start
            while current <= stop + 1e-9:
                expanded.append(round(current, 10))
                current += step
            return tuple(expanded)
        if not values:
            raise ValueError("values must be non-empty")
        return values

    @staticmethod
    def create_environment(
        *,
        topology_name: str = "nobel-eu",
        modulation_names: str = DEFAULT_MODULATION_NAMES,
        seed: int = 42,
        bit_rates: tuple[int, ...] = DEFAULT_BIT_RATES,
        load: float | int | tuple[float, ...] | list[float] = 300.0,
        mean_holding_time: float = 10_800.0,
        num_spectrum_resources: int = 320,
        episode_length: int = 100,
        modulations_to_consider: int = 3,
        defragmentation: bool = False,
        k_paths: int = 3,
        gen_observation: bool = True,
        launch_power_dbm: float = 0.0,
        margin: float = 0.0,
        measure_disruptions: bool = True,
        drop_on_disruption: bool = True,
    ) -> tuple[ScenarioConfig, ...]:
        if defragmentation:
            raise ValueError("defragmentation is not supported by this environment setup")
        loads = SimulationUtils.normalize_values(load)
        modulations = get_modulations(modulation_names)
        return tuple(
            ScenarioConfig(
                scenario_id=f"{topology_name}_load_{current_load:g}_seed{seed + index}",
                topology_id=topology_name,
                topology_dir=BUILTIN_TOPOLOGY_DIR,
                k_paths=k_paths,
                num_spectrum_resources=num_spectrum_resources,
                episode_length=episode_length,
                max_span_length_km=DEFAULT_MAX_SPAN_LENGTH_KM,
                default_attenuation_db_per_km=DEFAULT_ATTENUATION_DB_PER_KM,
                default_noise_figure_db=DEFAULT_NOISE_FIGURE_DB,
                bit_rates=bit_rates,
                load=float(current_load),
                mean_holding_time=mean_holding_time,
                qot_constraint="ASE+NLI",
                measure_disruptions=measure_disruptions,
                drop_on_disruption=drop_on_disruption,
                frequency_start=DEFAULT_FREQUENCY_START,
                frequency_slot_bandwidth=DEFAULT_FREQUENCY_SLOT_BANDWIDTH,
                launch_power_dbm=launch_power_dbm,
                margin=margin,
                bandwidth=num_spectrum_resources * DEFAULT_FREQUENCY_SLOT_BANDWIDTH,
                modulations=modulations,
                modulations_to_consider=modulations_to_consider,
                enable_observation=gen_observation,
                seed=seed + index,
            )
            for index, current_load in enumerate(loads)
        )


def add_progress_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument("--progress", dest="progress", action="store_true", default=True)
    progress_group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser


def build_standard_sweep_parser(*, description: str, default_output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--episodes-per-point", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--capture-services", action="store_true", default=False)
    add_progress_arguments(parser)
    return parser


def float_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def float_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def float_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])

    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    if lower_index == upper_index:
        return float(lower)
    weight = rank - lower_index
    return float(lower + (upper - lower) * weight)


def build_modulation_index_to_name(modulation_names: str) -> dict[int, str]:
    return {
        int(modulation.spectral_efficiency): f"modulation_{modulation.name.lower()}".replace("-", "_")
        for modulation in get_modulations(modulation_names)
    }


def select_masked_first_fit_policy(env: object, info: dict[str, object]) -> int:
    mask = info.get("mask")
    if mask is None and hasattr(env, "action_masks"):
        mask = env.action_masks()
    if mask is None:
        raise RuntimeError("first-fit policy requires an action mask")
    return int(select_first_fit_action(mask))


def select_disruption_aware_first_fit_policy(env: object, info: dict[str, object]) -> int:
    del info
    if not hasattr(env, "heuristic_context"):
        raise RuntimeError("disruption-aware first-fit requires an env with heuristic_context()")
    return int(select_disruption_aware_first_fit_action(env.heuristic_context()))


def select_policy_action(policy_name: str, env: object, info: dict[str, object]) -> int:
    key = policy_name.strip().lower()
    if key in {"jocn_ksp_ff_bm", "ksp-ff-bm", "strategy_1", "1"}:
        return select_masked_first_fit_policy(env, info)
    if key in {"jocn_ls_bm_ksp", "ls-bm-ksp", "strategy_2", "2"}:
        return int(select_jocn_ls_bm_ksp_action(env.heuristic_context()))
    if key in {"jocn_bm_ksp_lb", "bm-ksp-lb", "strategy_3", "3"}:
        return int(select_jocn_bm_ksp_lb_action(env.heuristic_context()))
    if key in {"jocn_ksp_lb_bm", "ksp-lb-bm", "strategy_4", "4"}:
        return int(select_jocn_ksp_lb_bm_action(env.heuristic_context()))
    if key in {"first_fit", "ksp-ff-bm"}:
        return select_masked_first_fit_policy(env, info)
    if key in {"disruption_aware_first_fit", "disruption-aware-first-fit"}:
        return int(select_disruption_aware_first_fit_action(env.heuristic_context()))
    if key in {"random", "random_runtime"}:
        return int(select_random_runtime_action(env.heuristic_context()))
    if key in {"load_balancing", "load-balancing"}:
        return int(select_load_balancing_runtime_action(env.heuristic_context()))
    if key in {"lowest_fragmentation", "lowest-fragmentation"}:
        return int(select_lowest_fragmentation_runtime_action(env.heuristic_context()))
    if key in {"highest_snr_first_fit", "highest-snr-first-fit"}:
        return int(select_highest_snr_first_fit_runtime_action(env.heuristic_context()))
    if key in {"ksp_best_mod_last_fit", "ksp-best-mod-last-fit"}:
        return int(select_ksp_best_mod_last_fit_runtime_action(env.heuristic_context()))
    raise ValueError(f"unsupported policy_name {policy_name!r}")


def episode_modulation_counts(
    statistics_snapshot,
    modulation_index_to_name: dict[int, str],
) -> dict[str, float]:
    counts_by_key = {column_name: 0.0 for column_name in modulation_index_to_name.values()}
    for spectral_efficiency, count in statistics_snapshot.episode_modulation_histogram:
        column_name = modulation_index_to_name.get(int(spectral_efficiency))
        if column_name is None:
            continue
        counts_by_key[column_name] = float(count)
    return counts_by_key


STANDARD_EPISODE_FIELDNAMES = [
    "date",
    "topology_id",
    "policy",
    "sweep_name",
    "sweep_value",
    "load",
    "margin",
    "launch_power_dbm",
    "episode_index",
    "episode_seed",
    "requests_per_episode",
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
    "accepted_osnr_margin_mean",
    "final_osnr_margin_mean",
    "episode_time_s",
]
STANDARD_SUMMARY_BASE_FIELDS = [
    "date",
    "topology_id",
    "policy",
    "sweep_name",
    "sweep_value",
    "episodes",
    "requests_per_episode",
    "load",
    "margin",
    "launch_power_dbm",
]
STANDARD_SUMMARY_METRIC_NAMES = [
    "services_accepted",
    "services_served",
    "service_blocking_rate",
    "service_served_rate",
    "effective_blocking_rate",
    "bit_rate_blocking_rate",
    "blocked_due_to_resources",
    "blocked_due_to_osnr",
    "accepted_osnr_margin_mean",
    "final_osnr_margin_mean",
    "episode_disrupted_services_count",
    "episode_disrupted_services_rate",
    "disrupted_or_dropped_services",
    "episode_time_s",
]

STANDARD_SERVICE_FIELDNAMES = [
    "date",
    "topology_id",
    "policy",
    "sweep_name",
    "sweep_value",
    "load",
    "margin",
    "launch_power_dbm",
    "episode_index",
    "episode_seed",
    "request_index",
    "service_id",
    "source_id",
    "destination_id",
    "bit_rate",
    "accepted",
    "status",
    "chosen_path_index",
    "path_k",
    "path_length",
    "chosen_modulation_index",
    "modulation",
    "modulation_spectral_efficiency",
    "min_osnr",
    "osnr",
    "osnr_req",
    "osnr_margin",
    "ase",
    "nli",
    "disrupted_services",
    "fragmentation_shannon_entropy",
    "fragmentation_route_cuts",
    "fragmentation_route_rss",
]


def run_first_fit_sweep(
    *,
    script_path: Path,
    family: str,
    sweep_name: str,
    scenarios_by_value: tuple[tuple[float, ScenarioConfig], ...],
    output_dir: Path,
    parallelism,
    episodes_per_point: int = 1,
    now=None,
    policy_name: str | tuple[str, ...] = "first_fit",
    progress: bool = True,
    progress_interval: int = 100,
    capture_services: bool = False,
) -> StandardSweepOutputs:
    from . import sweep_reporting as report_utils

    if not scenarios_by_value:
        raise ValueError("scenarios_by_value must be non-empty")
    if episodes_per_point <= 0:
        raise ValueError("episodes_per_point must be positive")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    date_label = report_utils.date_prefix(now)
    resolved_parallelism = parallelism.resolve()
    artifacts = None
    if capture_services:
        artifacts = {
            "episodes": "episodes.csv",
            "summary": "summary.csv",
            "services": "services.csv",
        }
    policy_names = (policy_name,) if isinstance(policy_name, str) else tuple(policy_name)
    if not policy_names:
        raise ValueError("policy_name must contain at least one policy")
    run = report_utils.create_experiment_run(
        script_path=script_path,
        base_dir=output_dir,
        family=family,
        now=now,
        scenario_name=family,
        scenario_id=f"{scenarios_by_value[0][1].topology_id}_{sweep_name}",
        overrides={
            "sweep_name": sweep_name,
            "values": tuple(value for value, _scenario in scenarios_by_value),
            "policies": policy_names,
        },
        parallelism=resolved_parallelism,
        artifacts=artifacts,
    )
    cases = tuple(
        SweepCase(
            scenario=replace(
                scenario,
                scenario_id=f"{scenario.scenario_id}_episode_{episode_index}",
                seed=int(scenario.seed or 0) + episode_index,
            ),
            sweep_name=sweep_name,
            sweep_value=float(value),
            episode_index=episode_index,
            date_label=date_label,
            policy_name=policy,
        )
        for policy in policy_names
        for value, scenario in scenarios_by_value
        for episode_index in range(episodes_per_point)
    )
    if resolved_parallelism.workers == 1:
        total_steps = sum(int(case.scenario.episode_length) for case in cases)
        with StepProgress(
            total=total_steps,
            description=f"{family} {sweep_name}",
            interval=progress_interval,
            enabled=progress,
        ) as step_progress:
            results = [
                _run_sweep_case(case, progress=step_progress, capture_services=capture_services)
                for case in cases
            ]
    else:
        worker_fn = partial(_run_sweep_case, capture_services=capture_services)
        results = report_utils.run_cases(cases, worker_fn, parallelism=resolved_parallelism)

    episode_rows = [result.episode_row for result in results]
    summary_rows = _build_standard_summary_rows(episode_rows, report_utils)

    episodes_csv = run.artifact_path("episodes")
    summary_csv = run.artifact_path("summary")
    report_utils.write_csv_rows(path=episodes_csv, fieldnames=STANDARD_EPISODE_FIELDNAMES, rows=episode_rows)
    summary_fieldnames = report_utils.build_summary_fieldnames(
        base_fields=STANDARD_SUMMARY_BASE_FIELDS,
        metric_names=STANDARD_SUMMARY_METRIC_NAMES,
    )
    report_utils.write_csv_rows(path=summary_csv, fieldnames=summary_fieldnames, rows=summary_rows)
    services_csv = None
    if capture_services:
        services_csv = run.artifact_path("services")
        service_rows = [row for result in results for row in result.service_rows]
        report_utils.write_csv_rows(
            path=services_csv,
            fieldnames=STANDARD_SERVICE_FIELDNAMES,
            rows=service_rows,
        )
    return StandardSweepOutputs(
        run_dir=run.run_dir,
        episodes_csv=episodes_csv,
        summary_csv=summary_csv,
        metadata_json=run.metadata_path,
        services_csv=services_csv,
    )


def _run_sweep_case(
    case: SweepCase,
    progress: StepProgress | None = None,
    *,
    capture_services: bool = False,
) -> SweepCaseResult:
    env = make_env(config=case.scenario)
    episode_seed = int(case.scenario.seed or 0)
    _, info = env.reset(seed=episode_seed)

    accepted_osnr_margins: list[float] = []
    service_rows: list[dict[str, object]] = []
    simulator = getattr(env, "simulator", None)
    started_at = time.perf_counter()
    while True:
        current_request = simulator.current_request if capture_services and simulator is not None else None
        current_analysis = simulator.current_analysis if capture_services and simulator is not None else None
        action = select_policy_action(case.policy_name, env, info)
        _, _, terminated, truncated, info = env.step(action)
        if progress is not None:
            progress.update()
        if capture_services and current_request is not None:
            service_rows.append(
                _build_service_row(
                    case=case,
                    episode_seed=episode_seed,
                    request=current_request,
                    analysis=current_analysis,
                    info=info,
                    simulator=simulator,
                )
            )
        if bool(info.get("accepted", False)):
            accepted_osnr_margins.append(float(info.get("osnr", 0.0)) - float(info.get("osnr_req", 0.0)))
        if terminated or truncated:
            break

    episode_time_s = time.perf_counter() - started_at
    simulator = env.simulator
    if simulator.statistics is None:
        raise RuntimeError("simulator statistics are not available after the episode")
    snapshot = simulator.statistics.snapshot()
    active_services = () if simulator.state is None else simulator.state.active_services_by_id.values()
    final_osnr_margins = [
        float(service.osnr) - (float(service.modulation.minimum_osnr) + float(case.scenario.margin))
        for service in active_services
        if service.modulation is not None
    ]
    episode_row = {
        "date": case.date_label,
        "topology_id": case.scenario.topology_id,
        "policy": case.policy_name,
        "sweep_name": case.sweep_name,
        "sweep_value": float(case.sweep_value),
        "load": float(case.scenario.load),
        "margin": float(case.scenario.margin),
        "launch_power_dbm": float(case.scenario.launch_power_dbm),
        "episode_index": int(case.episode_index),
        "episode_seed": episode_seed,
        "requests_per_episode": int(case.scenario.episode_length),
        "services_processed": int(snapshot.episode_services_processed),
        "services_accepted": int(snapshot.episode_services_accepted),
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
        "disrupted_or_dropped_services": int(snapshot.episode_services_dropped_qot),
        "accepted_osnr_margin_mean": float_mean(accepted_osnr_margins),
        "final_osnr_margin_mean": float_mean(final_osnr_margins),
        "episode_time_s": float(episode_time_s),
    }
    return SweepCaseResult(episode_row=episode_row, service_rows=service_rows)


def _build_service_row(
    *,
    case: SweepCase,
    episode_seed: int,
    request,
    analysis,
    info: dict[str, object],
    simulator,
) -> dict[str, object]:
    accepted = bool(info.get("accepted", False))
    status = info.get("status", "")
    chosen_path_index = info.get("chosen_path_index", None)
    chosen_modulation_index = info.get("chosen_modulation_index", None)
    osnr = float(info.get("osnr", 0.0))
    osnr_req = float(info.get("osnr_req", 0.0))
    osnr_margin = osnr - osnr_req

    path_k = -1
    path_length = 0.0
    modulation_name = ""
    modulation_efficiency = None
    min_osnr = 0.0
    ase = 0.0
    nli = 0.0
    disrupted_services = 0

    if accepted and analysis is not None and chosen_path_index is not None:
        try:
            path = analysis.paths[int(chosen_path_index)]
        except (IndexError, TypeError):
            path = None
        if path is not None:
            path_k = int(path.k)
            path_length = float(path.length_km)

    if accepted and chosen_modulation_index is not None and simulator is not None:
        try:
            modulation = simulator.config.modulations[int(chosen_modulation_index)]
        except (IndexError, TypeError, AttributeError):
            modulation = None
        if modulation is not None:
            modulation_name = modulation.name
            modulation_efficiency = int(modulation.spectral_efficiency)
            min_osnr = float(modulation.minimum_osnr)

    if accepted and simulator is not None and getattr(simulator, "state", None) is not None:
        state = simulator.state
        if state is not None:
            service = state.active_services_by_id.get(request.service_id)
            if service is not None:
                ase = float(service.ase)
                nli = float(service.nli)
                disrupted_services = int(service.disruption_count)
                if not modulation_name and service.modulation is not None:
                    modulation_name = service.modulation.name
                    modulation_efficiency = int(service.modulation.spectral_efficiency)
                    min_osnr = float(service.modulation.minimum_osnr)

    return {
        "date": case.date_label,
        "topology_id": case.scenario.topology_id,
        "policy": case.policy_name,
        "sweep_name": case.sweep_name,
        "sweep_value": float(case.sweep_value),
        "load": float(case.scenario.load),
        "margin": float(case.scenario.margin),
        "launch_power_dbm": float(case.scenario.launch_power_dbm),
        "episode_index": int(case.episode_index),
        "episode_seed": int(episode_seed),
        "request_index": int(request.request_index),
        "service_id": int(request.service_id),
        "source_id": int(request.source_id),
        "destination_id": int(request.destination_id),
        "bit_rate": int(request.bit_rate),
        "accepted": accepted,
        "status": status,
        "chosen_path_index": -1 if chosen_path_index is None else int(chosen_path_index),
        "path_k": int(path_k),
        "path_length": float(path_length),
        "chosen_modulation_index": -1 if chosen_modulation_index is None else int(chosen_modulation_index),
        "modulation": modulation_name,
        "modulation_spectral_efficiency": modulation_efficiency,
        "min_osnr": float(min_osnr),
        "osnr": float(osnr),
        "osnr_req": float(osnr_req),
        "osnr_margin": float(osnr_margin),
        "ase": float(ase),
        "nli": float(nli),
        "disrupted_services": int(disrupted_services),
        "fragmentation_shannon_entropy": float(info.get("fragmentation_shannon_entropy", 0.0)),
        "fragmentation_route_cuts": float(info.get("fragmentation_route_cuts", 0.0)),
        "fragmentation_route_rss": float(info.get("fragmentation_route_rss", 0.0)),
    }


def _build_standard_summary_rows(episode_rows: list[dict[str, object]], report_utils) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted({(str(row["policy"]), float(row["sweep_value"])) for row in episode_rows})
    for policy, value in keys:
        value_rows = [
            row
            for row in episode_rows
            if str(row["policy"]) == policy and float(row["sweep_value"]) == value
        ]
        first = value_rows[0]
        row = {
            "date": first["date"],
            "topology_id": first["topology_id"],
            "policy": first["policy"],
            "sweep_name": first["sweep_name"],
            "sweep_value": float(value),
            "episodes": len(value_rows),
            "requests_per_episode": int(first["requests_per_episode"]),
            "load": float(first["load"]),
            "margin": float(first["margin"]),
            "launch_power_dbm": float(first["launch_power_dbm"]),
        }
        row.update(report_utils.aggregate_summary_metrics(value_rows, metric_names=STANDARD_SUMMARY_METRIC_NAMES))
        rows.append(row)
    return rows


def compare_summary_rows(
    reference_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    *,
    key_fields: tuple[str, ...],
    metric_tolerances: dict[str, float],
) -> list[ComparisonResult]:
    reference_by_key = {_row_key(row, key_fields): row for row in reference_rows}
    candidate_by_key = {_row_key(row, key_fields): row for row in candidate_rows}
    results: list[ComparisonResult] = []
    for key, reference_row in reference_by_key.items():
        if key not in candidate_by_key:
            raise ValueError(f"candidate summary is missing key {key!r}")
        candidate_row = candidate_by_key[key]
        for metric, tolerance in metric_tolerances.items():
            reference_value = float(reference_row[metric])
            candidate_value = float(candidate_row[metric])
            absolute_delta = abs(reference_value - candidate_value)
            results.append(
                ComparisonResult(
                    key=key,
                    metric=metric,
                    reference_value=reference_value,
                    candidate_value=candidate_value,
                    tolerance=float(tolerance),
                    absolute_delta=absolute_delta,
                    passed=absolute_delta <= float(tolerance),
                )
            )
    return results


def _row_key(row: dict[str, object], key_fields: tuple[str, ...]) -> tuple[str, ...]:
    if not key_fields:
        raise ValueError("key_fields must be non-empty")
    return tuple(str(row[field]) for field in key_fields)


__all__ = [
    "DEFAULT_MODULATION_NAMES",
    "ComparisonResult",
    "EpisodePolicy",
    "SimulationUtils",
    "StandardSweepOutputs",
    "STANDARD_SERVICE_FIELDNAMES",
    "StepProgress",
    "SweepCase",
    "add_progress_arguments",
    "build_standard_sweep_parser",
    "build_modulation_index_to_name",
    "compare_summary_rows",
    "episode_modulation_counts",
    "float_mean",
    "float_percentile",
    "float_std",
    "run_first_fit_sweep",
    "select_disruption_aware_first_fit_policy",
    "select_masked_first_fit_policy",
    "select_policy_action",
]
