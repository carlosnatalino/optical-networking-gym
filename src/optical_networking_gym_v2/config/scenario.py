from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from collections.abc import Sequence

from optical_networking_gym_v2.contracts.enums import MaskMode, RewardProfile, TrafficMode
from optical_networking_gym_v2.contracts.modulation import Modulation


_VALID_QOT_CONSTRAINTS = frozenset({"ASE+NLI", "DIST"})


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    scenario_id: str
    topology_id: str
    k_paths: int
    num_spectrum_resources: int
    topology_dir: str | Path | None = None
    episode_length: int = 1_000
    max_span_length_km: float = 100.0
    default_attenuation_db_per_km: float = 0.2
    default_noise_figure_db: float = 4.5
    traffic_mode: TrafficMode = TrafficMode.DYNAMIC
    traffic_source: object | None = None
    bit_rates: tuple[int, ...] | None = None
    bit_rate_probabilities: tuple[float, ...] | None = None
    load: float = 300.0
    mean_holding_time: float = 100.0
    mean_inter_arrival_time: float | None = None
    static_traffic_path: str | Path | None = None
    mask_mode: MaskMode = MaskMode.RESOURCE_AND_QOT
    reward_profile: RewardProfile = RewardProfile.BALANCED
    qot_constraint: str = "ASE+NLI"
    measure_disruptions: bool = False
    drop_on_disruption: bool = False
    channel_width: float = 12.5
    frequency_start: float = (3e8 / 1565e-9)
    frequency_slot_bandwidth: float = 12.5e9
    launch_power_dbm: float = 0.0
    margin: float = 0.0
    bandwidth: float | None = None
    modulations: tuple[Modulation, ...] = ()
    modulations_to_consider: int | None = None
    enable_observation: bool = True
    enable_action_mask: bool = True
    include_mask_in_info: bool = True
    capture_traffic_table: bool = False
    capture_step_trace: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if not self.topology_id:
            raise ValueError("topology_id must be a non-empty string")
        if self.k_paths <= 0:
            raise ValueError("k_paths must be positive")
        if self.num_spectrum_resources <= 0:
            raise ValueError("num_spectrum_resources must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.max_span_length_km <= 0:
            raise ValueError("max_span_length_km must be positive")
        if self.default_attenuation_db_per_km <= 0:
            raise ValueError("default_attenuation_db_per_km must be positive")
        if self.default_noise_figure_db <= 0:
            raise ValueError("default_noise_figure_db must be positive")
        if self.channel_width <= 0:
            raise ValueError("channel_width must be positive")
        if self.frequency_slot_bandwidth <= 0:
            raise ValueError("frequency_slot_bandwidth must be positive")
        if self.qot_constraint not in _VALID_QOT_CONSTRAINTS:
            raise ValueError(
                "qot_constraint must be one of: " + ", ".join(sorted(_VALID_QOT_CONSTRAINTS))
            )
        if self.bandwidth is None:
            object.__setattr__(
                self,
                "bandwidth",
                self.num_spectrum_resources * self.frequency_slot_bandwidth,
            )
        elif self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.modulations_to_consider is None:
            object.__setattr__(self, "modulations_to_consider", len(self.modulations))
        elif self.modulations_to_consider < 0:
            raise ValueError("modulations_to_consider must be non-negative")
        elif self.modulations:
            object.__setattr__(
                self,
                "modulations_to_consider",
                min(self.modulations_to_consider, len(self.modulations)),
            )
        if self.mean_holding_time <= 0:
            raise ValueError("mean_holding_time must be positive")
        if self.load <= 0:
            raise ValueError("load must be positive")
        if self.mean_inter_arrival_time is not None and self.mean_inter_arrival_time <= 0:
            raise ValueError("mean_inter_arrival_time must be positive")
        normalized_bit_rates = self._normalize_bit_rates(self.bit_rates)
        object.__setattr__(self, "bit_rates", normalized_bit_rates)
        normalized_probabilities = self._normalize_bit_rate_probabilities(
            normalized_bit_rates,
            self.bit_rate_probabilities,
        )
        object.__setattr__(self, "bit_rate_probabilities", normalized_probabilities)
        if not self.enable_action_mask:
            object.__setattr__(self, "include_mask_in_info", False)
        resolved_static_path = Path(self.static_traffic_path) if self.static_traffic_path is not None else None
        object.__setattr__(self, "static_traffic_path", resolved_static_path)
        if self.topology_dir is not None:
            object.__setattr__(self, "topology_dir", Path(self.topology_dir))
        if self.traffic_source is None:
            object.__setattr__(self, "traffic_source", self._build_default_traffic_source())
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.traffic_mode is TrafficMode.STATIC and self.traffic_source is None:
            raise ValueError("traffic_source is required when traffic_mode is static")

    def _normalize_bit_rates(self, bit_rates: tuple[int, ...] | None) -> tuple[int, ...]:
        raw_values = (10, 40, 100, 400) if bit_rates is None else bit_rates
        if not isinstance(raw_values, Sequence) or len(raw_values) == 0:
            raise ValueError("bit_rates must be a non-empty sequence")
        return tuple(int(bit_rate) for bit_rate in raw_values)

    def _normalize_bit_rate_probabilities(
        self,
        bit_rates: tuple[int, ...],
        probabilities: tuple[float, ...] | None,
    ) -> tuple[float, ...] | None:
        if probabilities is None:
            return None
        if not isinstance(probabilities, Sequence) or len(probabilities) != len(bit_rates):
            raise ValueError("bit_rate_probabilities must match bit_rates length")
        normalized = tuple(float(probability) for probability in probabilities)
        if any(probability < 0.0 for probability in normalized):
            raise ValueError("bit_rate_probabilities must be non-negative")
        total = sum(normalized)
        if total <= 0.0:
            raise ValueError("bit_rate_probabilities must sum to a positive value")
        return normalized

    def _build_default_traffic_source(self) -> object | None:
        if self.traffic_mode is TrafficMode.STATIC:
            if self.static_traffic_path is not None:
                return self.static_traffic_path
            return None
        traffic_source: dict[str, object] = {
            "bit_rates": self.bit_rates,
            "load": float(self.load),
            "mean_holding_time": float(self.mean_holding_time),
        }
        if self.bit_rate_probabilities is not None:
            traffic_source["bit_rate_probabilities"] = self.bit_rate_probabilities
        if self.mean_inter_arrival_time is not None:
            traffic_source["mean_inter_arrival_time"] = float(self.mean_inter_arrival_time)
        return traffic_source

    def runtime_structure_key(self) -> tuple[object, ...]:
        return (
            self.topology_id,
            self.k_paths,
            self.num_spectrum_resources,
            self.max_span_length_km,
            self.default_attenuation_db_per_km,
            self.default_noise_figure_db,
            self.mask_mode,
            self.reward_profile,
            self.qot_constraint,
            self.measure_disruptions,
            self.drop_on_disruption,
            self.channel_width,
            self.frequency_start,
            self.frequency_slot_bandwidth,
            self.launch_power_dbm,
            self.margin,
            self.bandwidth,
            self.modulations,
            self.modulations_to_consider,
            self.enable_observation,
            self.enable_action_mask,
            self.include_mask_in_info,
        )
