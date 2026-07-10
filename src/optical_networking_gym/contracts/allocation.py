from __future__ import annotations

from dataclasses import dataclass

from .enums import Status


@dataclass(slots=True)
class Allocation:
    accepted: bool
    status: Status
    path_index: int | None = None
    modulation_index: int | None = None
    service_slot_start: int | None = None
    service_num_slots: int | None = None
    occupied_slot_start: int | None = None
    occupied_slot_end_exclusive: int | None = None

    @classmethod
    def accept(
        cls,
        *,
        path_index: int,
        modulation_index: int,
        service_slot_start: int,
        service_num_slots: int,
        occupied_slot_start: int,
        occupied_slot_end_exclusive: int,
    ) -> "Allocation":
        return cls(
            accepted=True,
            status=Status.ACCEPTED,
            path_index=path_index,
            modulation_index=modulation_index,
            service_slot_start=service_slot_start,
            service_num_slots=service_num_slots,
            occupied_slot_start=occupied_slot_start,
            occupied_slot_end_exclusive=occupied_slot_end_exclusive,
        )

    @classmethod
    def reject(cls, status: Status) -> "Allocation":
        if status is Status.ACCEPTED:
            raise ValueError("accepted decisions must use the accept factory")
        return cls(accepted=False, status=status)
