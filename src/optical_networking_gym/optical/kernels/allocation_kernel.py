from __future__ import annotations

import numpy as np


def candidate_starts_array(available_slots: np.ndarray, required_slots: int) -> np.ndarray:
    if required_slots <= 0:
        raise ValueError("required_slots must be positive")
    free_slots = np.asarray(available_slots, dtype=np.bool_)
    if free_slots.ndim != 1:
        raise ValueError("available_slots must be a 1D array")

    total_slots = free_slots.size
    candidates: list[int] = []
    run_start = -1

    for slot_index in range(total_slots):
        if free_slots[slot_index]:
            if run_start < 0:
                run_start = slot_index
            continue

        if run_start >= 0:
            block_end = slot_index
            block_length = block_end - run_start
            min_block_length = required_slots + 1
            if block_length >= min_block_length:
                last_candidate = block_end - min_block_length
                candidates.extend(range(run_start, last_candidate + 1))
            run_start = -1

    if run_start >= 0:
        block_end = total_slots
        block_length = block_end - run_start
        min_block_length = required_slots
        if block_length >= min_block_length:
            last_candidate = block_end - min_block_length
            candidates.extend(range(run_start, last_candidate + 1))

    return np.asarray(candidates, dtype=np.int32)


def block_is_free(
    slot_allocation: np.ndarray,
    link_indices: np.ndarray,
    slot_start: int,
    slot_end_exclusive: int,
) -> bool:
    slots = np.asarray(slot_allocation, dtype=np.int32)
    indices = np.asarray(link_indices, dtype=np.intp)
    for link_index in indices:
        for slot_index in range(slot_start, slot_end_exclusive):
            if slots[link_index, slot_index] != -1:
                return False
    return True


def fill_range(
    slot_allocation: np.ndarray,
    link_indices: np.ndarray,
    slot_start: int,
    slot_end_exclusive: int,
    value: int,
) -> None:
    slots = np.asarray(slot_allocation, dtype=np.int32)
    indices = np.asarray(link_indices, dtype=np.intp)
    for link_index in indices:
        slots[link_index, slot_start:slot_end_exclusive] = value
