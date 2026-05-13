from __future__ import annotations

import pytest

from optical_networking_gym_v2 import Allocation, Status


def test_accept_decision_factory_populates_selection_fields() -> None:
    decision = Allocation.accept(
        path_index=1,
        modulation_index=0,
        service_slot_start=8,
        service_num_slots=4,
        occupied_slot_start=7,
        occupied_slot_end_exclusive=12,
    )

    assert decision.accepted is True
    assert decision.status is Status.ACCEPTED
    assert decision.path_index == 1
    assert decision.service_slot_start == 8
    assert decision.service_num_slots == 4


def test_reject_decision_factory_keeps_selection_fields_empty() -> None:
    decision = Allocation.reject(Status.BLOCKED_RESOURCES)

    assert decision.accepted is False
    assert decision.status is Status.BLOCKED_RESOURCES
    assert decision.path_index is None
    assert decision.modulation_index is None
    assert decision.service_slot_start is None


def test_reject_decision_cannot_use_accepted_block_reason() -> None:
    with pytest.raises(ValueError, match="accepted decisions"):
        Allocation.reject(Status.ACCEPTED)
