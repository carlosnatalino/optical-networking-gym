from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.contracts.enums import MaskMode, RewardProfile, TrafficMode
from optical_networking_gym_v2.config.defaults import get_modulations, resolve_topology, set_topology_dir
from optical_networking_gym_v2.envs.optical_env import OpticalEnv
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.scenarios import build_scenario


_UNSET = object()


def make_env(
    topology_name: str | None = None,
    modulation_names: str | tuple[str, ...] | object = _UNSET,
    *,
    config: ScenarioConfig | None = None,
    scenario: str | ScenarioConfig | None = None,
    overrides: Mapping[str, object] | None = None,
    topology_dir: str | Path | None = None,
    seed: int | object = _UNSET,
    bit_rates: tuple[int, ...] | object = _UNSET,
    bit_rate_probabilities: tuple[float, ...] | None = None,
    load: float | object = _UNSET,
    mean_holding_time: float | object = _UNSET,
    mean_inter_arrival_time: float | None = None,
    num_spectrum_resources: int | object = _UNSET,
    episode_length: int | object = _UNSET,
    modulations_to_consider: int | None = None,
    modulations: str | tuple[str, ...] | tuple[object, ...] | object = _UNSET,
    k_paths: int | object = _UNSET,
    max_span_length_km: float | object = _UNSET,
    default_attenuation_db_per_km: float | object = _UNSET,
    default_noise_figure_db: float | object = _UNSET,
    traffic_mode: TrafficMode = TrafficMode.DYNAMIC,
    traffic_source: object | None = None,
    static_traffic_path: str | Path | None = None,
    mask_mode: MaskMode = MaskMode.RESOURCE_AND_QOT,
    reward_profile: RewardProfile = RewardProfile.BALANCED,
    qot_constraint: str = "ASE+NLI",
    channel_width: float | object = _UNSET,
    frequency_start: float | object = _UNSET,
    frequency_slot_bandwidth: float | object = _UNSET,
    launch_power_dbm: float | object = _UNSET,
    margin: float | object = _UNSET,
    bandwidth: float | None = None,
    measure_disruptions: bool = False,
    drop_on_disruption: bool = False,
    enable_observation: bool = True,
    enable_action_mask: bool = True,
    include_mask_in_info: bool = True,
    capture_traffic_table: bool = False,
    capture_step_trace: bool = False,
) -> OpticalEnv:
    """Build an :class:`OpticalEnv` from a scenario preset, complete config, or flat facade.

    Topology:
        `scenario`: Canonical preset name such as `ring4_quickstart` or an
        explicit `ScenarioConfig`. This is the standard public API.
        `config`: Complete flat scenario definition. When provided, `topology_name`
        and `scenario` cannot also be provided.
        `topology_name`: Topology identifier resolved from `topology_dir` or the
        previously configured global topology directory.
        `topology_dir`: Directory containing `.xml` or `.txt` topology files.
        `k_paths`: Number of K-shortest paths to precompute.
        `max_span_length_km`, `default_attenuation_db_per_km`,
        `default_noise_figure_db`: Physical defaults applied while parsing the topology.

    Traffic:
        `traffic_mode`: `dynamic` by default; use `static` with `traffic_source`
        or `static_traffic_path` for replay.
        `bit_rates`, `bit_rate_probabilities`, `load`, `mean_holding_time`,
        `mean_inter_arrival_time`, `seed`: Dynamic-traffic controls.
        `traffic_source`: Explicit dynamic mapping or static table input.
        `static_traffic_path`: Convenience path for static trace replay.
        `episode_length`: Number of processed requests before termination.

    Physical layer:
        `num_spectrum_resources`, `channel_width`, `frequency_start`,
        `frequency_slot_bandwidth`, `launch_power_dbm`, `margin`, `bandwidth`,
        `measure_disruptions`, `drop_on_disruption`.

    Routing / modulation:
        `modulation_names`, `modulations_to_consider`, `mask_mode`,
        `reward_profile`, `qot_constraint`.

    Outputs:
        `enable_observation`: If `False`, observations are returned as an empty
        `float32` array and the env advertises an empty Box observation space.
        `enable_action_mask`: If `False`, no action mask is materialized.
        `include_mask_in_info`: If `False`, a generated mask is exposed only
        through `env.action_masks()`.

    Instrumentation:
        `capture_traffic_table`, `capture_step_trace`: Optional capture paths
        for replay/debug artifacts. They stay disabled by default because they
        add runtime and allocation overhead.
    """
    if config is not None or isinstance(scenario, ScenarioConfig):
        if config is not None and scenario is not None:
            raise ValueError("config cannot be combined with scenario")
        if topology_name is not None:
            raise ValueError("topology_name cannot be combined with config")
        if overrides:
            raise ValueError("overrides cannot be combined with config")
        if modulations is not _UNSET:
            raise ValueError("modulations cannot be combined with config")
        resolved_config = config
        if isinstance(scenario, ScenarioConfig):
            resolved_config = scenario
    elif isinstance(scenario, str):
        if topology_name is not None:
            raise ValueError("topology_name cannot be combined with scenario")
        scenario_overrides = dict(overrides or {})
        _add_if_set(scenario_overrides, "topology_dir", topology_dir)
        _add_if_set(scenario_overrides, "seed", seed)
        _add_if_set(scenario_overrides, "bit_rates", bit_rates)
        _add_if_set(scenario_overrides, "bit_rate_probabilities", bit_rate_probabilities)
        _add_if_set(scenario_overrides, "load", load)
        _add_if_set(scenario_overrides, "mean_holding_time", mean_holding_time)
        _add_if_set(scenario_overrides, "mean_inter_arrival_time", mean_inter_arrival_time)
        _add_if_set(scenario_overrides, "num_spectrum_resources", num_spectrum_resources)
        _add_if_set(scenario_overrides, "episode_length", episode_length)
        _add_if_set(scenario_overrides, "modulations_to_consider", modulations_to_consider)
        _add_if_set(scenario_overrides, "modulations", modulations)
        _add_if_set(scenario_overrides, "modulation_names", modulation_names)
        _add_if_set(scenario_overrides, "k_paths", k_paths)
        _add_if_set(scenario_overrides, "max_span_length_km", max_span_length_km)
        _add_if_set(scenario_overrides, "default_attenuation_db_per_km", default_attenuation_db_per_km)
        _add_if_set(scenario_overrides, "default_noise_figure_db", default_noise_figure_db)
        _add_if_set(scenario_overrides, "channel_width", channel_width)
        _add_if_set(scenario_overrides, "frequency_start", frequency_start)
        _add_if_set(scenario_overrides, "frequency_slot_bandwidth", frequency_slot_bandwidth)
        _add_if_set(scenario_overrides, "launch_power_dbm", launch_power_dbm)
        _add_if_set(scenario_overrides, "margin", margin)
        _add_if_set(scenario_overrides, "bandwidth", bandwidth)
        resolved_config = build_scenario(scenario, **scenario_overrides)
    else:
        if topology_name is None:
            raise ValueError("topology_name is required when config is not provided")
        if overrides:
            raise ValueError("overrides require scenario mode")
        modulation_input = _value_or_default(
            modulations if modulations is not _UNSET else modulation_names,
            "BPSK,QPSK,8QAM,16QAM,32QAM,64QAM",
        )
        resolved_config = ScenarioConfig(
            scenario_id=f"{topology_name}_seed{_value_or_default(seed, 42)}",
            topology_id=topology_name,
            topology_dir=topology_dir,
            k_paths=_value_or_default(k_paths, 5),
            num_spectrum_resources=_value_or_default(num_spectrum_resources, 320),
            episode_length=_value_or_default(episode_length, 1_000),
            max_span_length_km=_value_or_default(max_span_length_km, 100.0),
            default_attenuation_db_per_km=_value_or_default(default_attenuation_db_per_km, 0.2),
            default_noise_figure_db=_value_or_default(default_noise_figure_db, 4.5),
            traffic_mode=traffic_mode,
            traffic_source=traffic_source,
            bit_rates=_value_or_default(bit_rates, (10, 40, 100, 400)),
            bit_rate_probabilities=bit_rate_probabilities,
            load=_value_or_default(load, 300.0),
            mean_holding_time=_value_or_default(mean_holding_time, 100.0),
            mean_inter_arrival_time=mean_inter_arrival_time,
            static_traffic_path=static_traffic_path,
            mask_mode=mask_mode,
            reward_profile=reward_profile,
            qot_constraint=qot_constraint,
            measure_disruptions=measure_disruptions,
            drop_on_disruption=drop_on_disruption,
            channel_width=_value_or_default(channel_width, 12.5),
            frequency_start=_value_or_default(frequency_start, (3e8 / 1565e-9)),
            frequency_slot_bandwidth=_value_or_default(frequency_slot_bandwidth, 12.5e9),
            launch_power_dbm=_value_or_default(launch_power_dbm, 0.0),
            margin=_value_or_default(margin, 0.0),
            bandwidth=bandwidth,
            modulations=get_modulations(modulation_input),  # type: ignore[arg-type]
            modulations_to_consider=modulations_to_consider,
            enable_observation=enable_observation,
            enable_action_mask=enable_action_mask,
            include_mask_in_info=include_mask_in_info,
            capture_traffic_table=capture_traffic_table,
            capture_step_trace=capture_step_trace,
            seed=_value_or_default(seed, 42),
        )

    if resolved_config.topology_dir is not None:
        set_topology_dir(resolved_config.topology_dir)

    topology_path = resolve_topology(resolved_config.topology_id)
    topology = TopologyModel.from_file(
        topology_path,
        topology_id=resolved_config.topology_id,
        k_paths=resolved_config.k_paths,
        max_span_length_km=resolved_config.max_span_length_km,
        default_attenuation_db_per_km=resolved_config.default_attenuation_db_per_km,
        default_noise_figure_db=resolved_config.default_noise_figure_db,
    )
    return OpticalEnv(
        resolved_config,
        topology,
        episode_length=resolved_config.episode_length,
        capture_traffic_table=resolved_config.capture_traffic_table,
        capture_step_trace=resolved_config.capture_step_trace,
    )


def _value_or_default(value: Any, default: Any) -> Any:
    return default if value is _UNSET else value


def _add_if_set(target: dict[str, object], key: str, value: object) -> None:
    if value is not _UNSET and value is not None:
        target[key] = value


__all__ = ["make_env"]
