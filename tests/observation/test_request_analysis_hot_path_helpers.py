from __future__ import annotations

import math

import numpy as np
import pytest

from optical_networking_gym_v2.simulation.request_analysis import (
    _analyze_free_mask,
    _fragmentation_damage_by_candidates,
    _summary_after_allocation,
)


def test_analyze_free_mask_returns_expected_run_metadata() -> None:
    analysis = _analyze_free_mask(
        np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=np.bool_)
    )

    assert analysis.summary.count == 3
    assert analysis.summary.largest == 3
    assert analysis.summary.total_free == 6
    assert analysis.summary.entropy == pytest.approx(
        -(2 / 6) * math.log(2 / 6) / math.log(3)
        - (3 / 6) * math.log(3 / 6) / math.log(3)
        - (1 / 6) * math.log(1 / 6) / math.log(3)
    )
    assert analysis.summary.rss == pytest.approx(math.sqrt(14) / 6)
    assert np.array_equal(analysis.run_starts, np.array([1, 4, 9], dtype=np.int32))
    assert np.array_equal(analysis.run_ends, np.array([3, 7, 10], dtype=np.int32))
    assert np.array_equal(analysis.run_lengths, np.array([2, 3, 1], dtype=np.int32))
    assert np.array_equal(
        analysis.slot_to_run_index,
        np.array([-1, 0, 0, -1, 1, 1, 1, -1, -1, 2], dtype=np.int32),
    )
    assert np.array_equal(analysis.largest_other_by_run, np.array([3, 2, 3], dtype=np.int32))


def test_summary_after_allocation_accounts_for_guard_band_when_capacity_remains() -> None:
    free_runs = _analyze_free_mask(np.ones(6, dtype=np.bool_))

    post = _summary_after_allocation(
        free_runs,
        service_slot_start=1,
        service_num_slots=2,
        total_slots=6,
    )

    assert post.count == 2
    assert post.largest == 2
    assert post.total_free == 3
    assert post.entropy == pytest.approx(0.9182958340544894)
    assert post.rss == pytest.approx(math.sqrt(5) / 3)


def test_summary_after_allocation_does_not_consume_guard_band_at_upper_boundary() -> None:
    free_runs = _analyze_free_mask(np.ones(6, dtype=np.bool_))

    post = _summary_after_allocation(
        free_runs,
        service_slot_start=4,
        service_num_slots=2,
        total_slots=6,
    )

    assert post.count == 1
    assert post.largest == 4
    assert post.total_free == 4
    assert post.entropy == 0.0
    assert post.rss == 1.0


def test_fragmentation_damage_by_candidates_matches_scalar_summary_calculation() -> None:
    free_runs = _analyze_free_mask(np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=np.bool_))
    candidate_indices = np.array([0, 1, 5], dtype=np.int32)

    num_blocks_damage, largest_block_damage = _fragmentation_damage_by_candidates(
        free_runs=free_runs,
        candidate_indices=candidate_indices,
        service_num_slots=2,
        total_slots=8,
        block_count_scale=4,
    )

    expected_num_blocks = []
    expected_largest_block = []
    for initial_slot in candidate_indices:
        post = _summary_after_allocation(
            free_runs,
            int(initial_slot),
            2,
            8,
        )
        expected_num_blocks.append(max(post.count - free_runs.summary.count, 0) / 4)
        expected_largest_block.append(max(free_runs.summary.largest - post.largest, 0) / 8)

    assert np.allclose(num_blocks_damage, np.asarray(expected_num_blocks, dtype=np.float32))
    assert np.allclose(largest_block_damage, np.asarray(expected_largest_block, dtype=np.float32))
