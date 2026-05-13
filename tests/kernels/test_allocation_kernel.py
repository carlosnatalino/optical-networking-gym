from __future__ import annotations

import numpy as np

from optical_networking_gym_v2.optical.kernels.allocation_kernel import (
    block_is_free,
    candidate_starts_array,
    fill_range,
)


def test_candidate_starts_array_matches_guard_band_rule() -> None:
    available = np.array([False, True, True, True, True, True, False, False], dtype=np.bool_)

    candidates = candidate_starts_array(available, required_slots=3)

    assert candidates.tolist() == [1, 2]


def test_candidate_starts_array_allows_exact_fit_at_spectrum_end() -> None:
    available = np.array([False, False, False, False, True, True, True], dtype=np.bool_)

    candidates = candidate_starts_array(available, required_slots=3)

    assert candidates.tolist() == [4]


def test_block_is_free_detects_busy_slot() -> None:
    slots = np.full((3, 8), -1, dtype=np.int32)
    slots[1, 3] = 9
    link_indices = np.array([0, 1], dtype=np.intp)

    assert block_is_free(slots, link_indices, 1, 3) is True
    assert block_is_free(slots, link_indices, 2, 5) is False


def test_fill_range_writes_all_links_in_place() -> None:
    slots = np.full((3, 8), -1, dtype=np.int32)
    link_indices = np.array([0, 2], dtype=np.intp)

    fill_range(slots, link_indices, 2, 5, 17)

    assert (slots[0, 2:5] == 17).all()
    assert (slots[2, 2:5] == 17).all()
    assert (slots[1, 2:5] == -1).all()
