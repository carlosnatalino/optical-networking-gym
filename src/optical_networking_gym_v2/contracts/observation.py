from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysis


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    request_feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    path_feature_names: tuple[str, ...]
    path_mod_feature_names: tuple[str, ...]
    path_slot_feature_names: tuple[str, ...]
    k_paths: int
    modulation_count: int
    num_spectrum_resources: int
    _request_index: dict[str, int] = field(init=False, repr=False)
    _global_index: dict[str, int] = field(init=False, repr=False)
    _path_index: dict[str, int] = field(init=False, repr=False)
    _path_mod_index: dict[str, int] = field(init=False, repr=False)
    _path_slot_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_request_index", {name: index for index, name in enumerate(self.request_feature_names)})
        object.__setattr__(self, "_global_index", {name: index for index, name in enumerate(self.global_feature_names)})
        object.__setattr__(self, "_path_index", {name: index for index, name in enumerate(self.path_feature_names)})
        object.__setattr__(self, "_path_mod_index", {name: index for index, name in enumerate(self.path_mod_feature_names)})
        object.__setattr__(self, "_path_slot_index", {name: index for index, name in enumerate(self.path_slot_feature_names)})

    @property
    def total_size(self) -> int:
        return (
            len(self.request_feature_names)
            + len(self.global_feature_names)
            + (self.k_paths * len(self.path_feature_names))
            + (self.k_paths * self.modulation_count * len(self.path_mod_feature_names))
            + (self.k_paths * self.num_spectrum_resources * len(self.path_slot_feature_names))
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        names.extend(f"request.{name}" for name in self.request_feature_names)
        names.extend(f"global.{name}" for name in self.global_feature_names)
        for path_index in range(self.k_paths):
            names.extend(f"path[{path_index}].{name}" for name in self.path_feature_names)
        for path_index in range(self.k_paths):
            for modulation_offset in range(self.modulation_count):
                names.extend(
                    f"path_mod[{path_index},{modulation_offset}].{name}"
                    for name in self.path_mod_feature_names
                )
        for path_index in range(self.k_paths):
            for slot_index in range(self.num_spectrum_resources):
                names.extend(
                    f"path_slot[{path_index},{slot_index}].{name}"
                    for name in self.path_slot_feature_names
                )
        return tuple(names)

    def request_feature_index(self, name: str) -> int:
        return self._request_index[name]

    def global_feature_index(self, name: str) -> int:
        return self._global_index[name]

    def path_feature_index(self, name: str) -> int:
        return self._path_index[name]

    def path_mod_feature_index(self, name: str) -> int:
        return self._path_mod_index[name]

    def path_slot_feature_index(self, name: str) -> int:
        return self._path_slot_index[name]


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    schema: ObservationSchema
    analysis: "RequestAnalysis"
    request: np.ndarray
    global_features: np.ndarray
    path: np.ndarray
    path_mod: np.ndarray
    path_slot: np.ndarray
    flat: np.ndarray


__all__ = ["ObservationSchema", "ObservationSnapshot"]
