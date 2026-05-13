from __future__ import annotations

import math

import numpy as np

from optical_networking_gym_v2.contracts import Allocation, Status
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState
from optical_networking_gym_v2.optical.kernels.allocation_kernel import candidate_starts_array
from .topology import PathRecord


def compute_required_slots(*, bit_rate: float, spectral_efficiency: float, channel_width: float) -> int:
    if bit_rate <= 0:
        raise ValueError("bit_rate must be positive")
    if spectral_efficiency <= 0:
        raise ValueError("spectral_efficiency must be positive")
    if channel_width <= 0:
        raise ValueError("channel_width must be positive")
    return int(math.ceil(bit_rate / (spectral_efficiency * channel_width)))


def occupied_slot_range(
    *,
    service_slot_start: int,
    service_num_slots: int,
    total_slots: int,
) -> tuple[int, int]:
    if service_slot_start < 0:
        raise ValueError("service_slot_start must be non-negative")
    if service_num_slots <= 0:
        raise ValueError("service_num_slots must be positive")
    if total_slots <= 0:
        raise ValueError("total_slots must be positive")

    occupied_start = service_slot_start
    occupied_end = service_slot_start + service_num_slots
    if occupied_end > total_slots:
        raise ValueError("service slot range exceeds total_slots")
    if occupied_end < total_slots:
        occupied_end += 1
    return occupied_start, occupied_end


def available_slots_for_path(state: RuntimeState, path: PathRecord) -> np.ndarray:
    if not path.link_ids:
        return np.ones(state.config.num_spectrum_resources, dtype=np.bool_)
    link_indices = np.asarray(path.link_ids, dtype=np.intp)
    return np.all(state.slot_allocation[link_indices, :] == -1, axis=0)


def candidate_starts(
    available_slots: np.ndarray,
    *,
    required_slots: int,
    total_slots: int,
) -> tuple[int, ...]:
    if required_slots <= 0:
        raise ValueError("required_slots must be positive")
    if total_slots <= 0:
        raise ValueError("total_slots must be positive")

    free = np.asarray(available_slots, dtype=np.bool_)
    if free.ndim != 1:
        raise ValueError("available_slots must be a 1D array")
    if free.size != total_slots:
        raise ValueError("available_slots length must match total_slots")

    return tuple(int(candidate) for candidate in candidate_starts_array(free, required_slots))


def path_is_free(
    state: RuntimeState,
    path: PathRecord,
    *,
    service_slot_start: int,
    service_num_slots: int,
) -> bool:
    occupied_start, occupied_end = occupied_slot_range(
        service_slot_start=service_slot_start,
        service_num_slots=service_num_slots,
        total_slots=state.config.num_spectrum_resources,
    )
    if not path.link_ids:
        return True
    link_indices = np.asarray(path.link_ids, dtype=np.intp)
    return bool(np.all(state.slot_allocation[link_indices, occupied_start:occupied_end] == -1))


def build_first_fit_allocation(
    state: RuntimeState,
    *,
    path: PathRecord,
    path_index: int,
    modulation_index: int,
    service_num_slots: int,
) -> Allocation:
    available_slots = available_slots_for_path(state, path)
    candidates = candidate_starts(
        available_slots,
        required_slots=service_num_slots,
        total_slots=state.config.num_spectrum_resources,
    )
    if not candidates:
        return Allocation.reject(Status.BLOCKED_RESOURCES)

    service_slot_start = candidates[0]
    occupied_start, occupied_end = occupied_slot_range(
        service_slot_start=service_slot_start,
        service_num_slots=service_num_slots,
        total_slots=state.config.num_spectrum_resources,
    )
    return Allocation.accept(
        path_index=path_index,
        modulation_index=modulation_index,
        service_slot_start=service_slot_start,
        service_num_slots=service_num_slots,
        occupied_slot_start=occupied_start,
        occupied_slot_end_exclusive=occupied_end,
    )


__all__ = [
    "available_slots_for_path",
    "build_first_fit_allocation",
    "candidate_starts",
    "compute_required_slots",
    "occupied_slot_range",
    "path_is_free",
]
