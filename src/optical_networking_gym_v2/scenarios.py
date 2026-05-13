from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import fields, replace
from itertools import product
from typing import Any

from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.contracts.modulation import Modulation
from optical_networking_gym_v2.defaults import (
    BUILTIN_TOPOLOGY_DIR,
    DEFAULT_K_PATHS,
    DEFAULT_LAUNCH_POWER_DBM,
    DEFAULT_LOAD,
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_MODULATIONS_TO_CONSIDER,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    DEFAULT_SEED,
    get_modulations,
)


_SCENARIO_FIELDS = frozenset(field.name for field in fields(ScenarioConfig))
_ALIASES = {"modulation_names": "modulations"}
_TRAFFIC_SOURCE_FIELDS = frozenset(
    {
        "bit_rates",
        "bit_rate_probabilities",
        "load",
        "mean_holding_time",
        "mean_inter_arrival_time",
        "static_traffic_path",
        "traffic_mode",
    }
)


def list_scenarios() -> tuple[str, ...]:
    return tuple(_PRESETS)


def build_scenario(name: str, **overrides: Any) -> ScenarioConfig:
    try:
        builder = _PRESETS[name]
    except KeyError as exc:
        available = ", ".join(list_scenarios())
        raise ValueError(f"Unknown scenario {name!r}. Available scenarios: {available}") from exc

    config = builder()
    normalized = _normalize_overrides(overrides)
    if normalized and "traffic_source" not in normalized and normalized.keys() & _TRAFFIC_SOURCE_FIELDS:
        normalized["traffic_source"] = None
    if normalized:
        config = replace(config, **normalized)
    return config


def iter_scenarios(
    name: str,
    axes: Mapping[str, Iterable[object]] | None = None,
    **fixed_overrides: Any,
) -> Iterator[ScenarioConfig]:
    normalized_axes = {
        _normalize_key(key): tuple(_plain_scalar(value) for value in values)
        for key, values in (axes or {}).items()
    }
    for key, values in normalized_axes.items():
        if key not in _SCENARIO_FIELDS:
            raise ValueError(f"Unknown ScenarioConfig override: {key}")
        if not values:
            raise ValueError(f"Scenario axis {key!r} must contain at least one value")

    if not normalized_axes:
        yield build_scenario(name, **fixed_overrides)
        return

    axis_names = tuple(normalized_axes)
    for combination in product(*(normalized_axes[key] for key in axis_names)):
        axis_overrides = dict(zip(axis_names, combination))
        config = build_scenario(name, **fixed_overrides, **axis_overrides)
        yield replace(config, scenario_id=_scenario_id(config.scenario_id, axis_overrides))


def build_scenario_grid(
    name: str,
    axes: Mapping[str, Iterable[object]] | None = None,
    **fixed_overrides: Any,
) -> tuple[ScenarioConfig, ...]:
    return tuple(iter_scenarios(name, axes=axes, **fixed_overrides))


def _normalize_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        key = _normalize_key(raw_key)
        if key not in _SCENARIO_FIELDS:
            raise ValueError(f"Unknown ScenarioConfig override: {raw_key}")
        value = _plain_scalar(raw_value)
        if key == "modulations":
            value = _normalize_modulations(value)
        normalized[key] = value
    return normalized


def _normalize_key(key: str) -> str:
    return _ALIASES.get(key, key)


def _normalize_modulations(value: object) -> tuple[Modulation, ...]:
    if isinstance(value, str):
        return get_modulations(value)
    values = tuple(value)  # type: ignore[arg-type]
    if not values:
        raise ValueError("modulations must contain at least one modulation")
    if all(isinstance(item, Modulation) for item in values):
        return values  # type: ignore[return-value]
    if all(isinstance(item, str) for item in values):
        return get_modulations(values)  # type: ignore[arg-type]
    raise ValueError("modulations must be names or Modulation objects")


def _plain_scalar(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            return value
    return value


def _scenario_id(base: str, overrides: Mapping[str, object]) -> str:
    suffix = "_".join(f"{key}{_slug(value)}" for key, value in overrides.items())
    return f"{base}_{suffix}" if suffix else base


def _slug(value: object) -> str:
    return str(value).replace(" ", "").replace(".", "p").replace("+", "plus").replace("-", "_")


def _ring4_quickstart() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="ring4_quickstart",
        topology_id="ring_4",
        topology_dir=BUILTIN_TOPOLOGY_DIR,
        k_paths=2,
        num_spectrum_resources=24,
        episode_length=8,
        load=10.0,
        mean_holding_time=8.0,
        modulations=get_modulations("BPSK,QPSK,16QAM"),
        modulations_to_consider=3,
        seed=DEFAULT_SEED,
    )


def _nobel_eu_baseline() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="nobel_eu_baseline",
        topology_id="nobel-eu",
        topology_dir=BUILTIN_TOPOLOGY_DIR,
        k_paths=DEFAULT_K_PATHS,
        num_spectrum_resources=DEFAULT_NUM_SPECTRUM_RESOURCES,
        episode_length=1_000,
        max_span_length_km=80.0,
        load=DEFAULT_LOAD,
        mean_holding_time=DEFAULT_MEAN_HOLDING_TIME,
        launch_power_dbm=DEFAULT_LAUNCH_POWER_DBM,
        bandwidth=4e12,
        modulations=get_modulations("BPSK,QPSK,8QAM,16QAM,32QAM,64QAM"),
        modulations_to_consider=DEFAULT_MODULATIONS_TO_CONSIDER,
        seed=DEFAULT_SEED,
    )


def _nobel_eu_legacy_benchmark() -> ScenarioConfig:
    return replace(
        _nobel_eu_baseline(),
        scenario_id="nobel_eu_legacy_benchmark",
        launch_power_dbm=2.0,
        modulations_to_consider=6,
    )


def _nobel_eu_publication() -> ScenarioConfig:
    return replace(_nobel_eu_legacy_benchmark(), scenario_id="nobel_eu_publication")


def _nobel_eu_disruptions() -> ScenarioConfig:
    return replace(
        _nobel_eu_publication(),
        scenario_id="nobel_eu_disruptions",
        measure_disruptions=True,
        drop_on_disruption=True,
    )


def _jocn_modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("BPSK", 100_000.0, 1, minimum_osnr=3.71, inband_xt=-14.0),
        Modulation("QPSK", 2_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("8QAM", 1_000.0, 3, minimum_osnr=10.84, inband_xt=-20.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        Modulation("32QAM", 250.0, 5, minimum_osnr=16.16, inband_xt=-26.0),
        Modulation("64QAM", 125.0, 6, minimum_osnr=19.01, inband_xt=-29.0),
    )


def _jocn_benchmark() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="jocn_benchmark",
        topology_id="nobel-eu",
        topology_dir=BUILTIN_TOPOLOGY_DIR,
        k_paths=5,
        num_spectrum_resources=320,
        episode_length=1_000,
        max_span_length_km=80.0,
        default_attenuation_db_per_km=0.2,
        default_noise_figure_db=4.5,
        bit_rates=(10, 40, 100, 400),
        load=210.0,
        mean_holding_time=10_800.0,
        qot_constraint="ASE+NLI",
        frequency_start=(3e8 / 1565e-9),
        frequency_slot_bandwidth=12.5e9,
        launch_power_dbm=-4.0,
        margin=0.0,
        bandwidth=4e12,
        modulations=_jocn_modulations(),
        modulations_to_consider=6,
        seed=DEFAULT_SEED,
    )


_PRESETS = {
    "ring4_quickstart": _ring4_quickstart,
    "nobel_eu_baseline": _nobel_eu_baseline,
    "nobel_eu_legacy_benchmark": _nobel_eu_legacy_benchmark,
    "nobel_eu_publication": _nobel_eu_publication,
    "nobel_eu_disruptions": _nobel_eu_disruptions,
    "jocn_benchmark": _jocn_benchmark,
}


__all__ = [
    "build_scenario",
    "build_scenario_grid",
    "iter_scenarios",
    "list_scenarios",
]
