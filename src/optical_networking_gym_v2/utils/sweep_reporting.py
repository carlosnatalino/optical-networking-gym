from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

from .experiment_utils import float_mean, float_std


Scalar = str | int | float | bool
CaseT = TypeVar("CaseT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class Parallelism:
    workers: int | None = None
    envs_per_worker: int = 1

    @classmethod
    def auto(cls) -> "Parallelism":
        return cls()

    def resolve(self) -> "Parallelism":
        workers = max(1, (os.cpu_count() or 1) - 2) if self.workers is None else self.workers
        if workers <= 0:
            raise ValueError("workers must be a positive integer")
        if self.envs_per_worker <= 0:
            raise ValueError("envs_per_worker must be a positive integer")
        return Parallelism(workers=int(workers), envs_per_worker=int(self.envs_per_worker))

    @property
    def max_active_envs(self) -> int:
        resolved = self.resolve()
        return int(resolved.workers or 1) * int(resolved.envs_per_worker)


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    run_dir: Path
    metadata_path: Path
    artifacts: Mapping[str, Path]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def artifact_path(self, name: str) -> Path:
        try:
            return self.artifacts[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.artifacts))
            raise ValueError(f"Unknown artifact {name!r}. Available artifacts: {available}") from exc


def date_prefix(now: datetime | None = None) -> str:
    current = datetime.now() if now is None else now
    return current.strftime("%d-%m")


def build_sweep_output_paths(
    *,
    base_dir: Path,
    sweep_name: str,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    prefix = date_prefix(now)
    return (
        Path(base_dir) / f"{prefix}-{sweep_name}-episodes.csv",
        Path(base_dir) / f"{prefix}-{sweep_name}-summary.csv",
    )


def create_experiment_run(
    *,
    script_path: Path,
    base_dir: Path,
    family: str,
    now: datetime | None = None,
    scenario_name: str | None = None,
    scenario_id: str | None = None,
    overrides: Mapping[str, object] | None = None,
    parallelism: Parallelism | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> ExperimentRun:
    current = datetime.now() if now is None else now
    resolved_parallelism = (parallelism or Parallelism.auto()).resolve()
    script = Path(script_path)
    run_dir = Path(base_dir) / family / script.stem / current.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = dict(artifacts or {"episodes": "episodes.csv", "summary": "summary.csv"})
    artifact_paths = {name: run_dir / filename for name, filename in artifact_names.items()}
    metadata: dict[str, object] = {
        "script_name": script.name,
        "script_stem": script.stem,
        "timestamp": current.isoformat(),
        "scenario_name": scenario_name,
        "scenario_id": scenario_id,
        "overrides": dict(overrides or {}),
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "workers": resolved_parallelism.workers,
        "envs_per_worker": resolved_parallelism.envs_per_worker,
        "max_active_envs": resolved_parallelism.max_active_envs,
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return ExperimentRun(
        run_dir=run_dir,
        metadata_path=metadata_path,
        artifacts=artifact_paths,
        metadata=metadata,
    )


def run_cases(
    cases: Sequence[CaseT],
    worker_fn: Callable[[CaseT], ResultT],
    *,
    parallelism: Parallelism | None = None,
) -> list[ResultT]:
    resolved = (parallelism or Parallelism.auto()).resolve()
    if resolved.workers == 1:
        return [worker_fn(case) for case in cases]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=resolved.workers, mp_context=context) as executor:
        return list(executor.map(worker_fn, cases))


def build_summary_fieldnames(
    *,
    base_fields: Sequence[str],
    metric_names: Sequence[str],
) -> list[str]:
    summary_fields = list(base_fields)
    for metric_name in metric_names:
        summary_fields.append(f"{metric_name}_mean")
        summary_fields.append(f"{metric_name}_std")
    return summary_fields


def aggregate_summary_metrics(
    rows: Sequence[Mapping[str, Scalar]],
    *,
    metric_names: Sequence[str],
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        aggregated[f"{metric_name}_mean"] = float_mean(values)
        aggregated[f"{metric_name}_std"] = float_std(values)
    return aggregated


def write_csv_rows(
    *,
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Scalar]],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "Scalar",
    "ExperimentRun",
    "Parallelism",
    "aggregate_summary_metrics",
    "build_summary_fieldnames",
    "build_sweep_output_paths",
    "create_experiment_run",
    "date_prefix",
    "run_cases",
    "write_csv_rows",
]
