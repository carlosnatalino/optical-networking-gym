from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Modulation:
    name: str
    maximum_length: float
    spectral_efficiency: int
    minimum_osnr: float = 0.0
    inband_xt: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if self.maximum_length <= 0:
            raise ValueError("maximum_length must be positive")
        if self.spectral_efficiency <= 0:
            raise ValueError("spectral_efficiency must be positive")
