from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Static-analysis view of the lazy export table below. Kept lazy at
    # runtime because eager subpackage imports create import cycles
    # (e.g. contracts -> network -> runtime -> contracts and
    # optical -> envs -> runtime -> features -> optical).
    from .action_mask import ActionSelection
    from .allocation import Allocation
    from .enums import MaskMode, RewardProfile, Status, TrafficMode
    from .modulation import Modulation
    from .observation import ObservationSchema, ObservationSnapshot
    from .qot import QoTRequest, QoTResult, ServiceQoTUpdate
    from .reward import CandidateRewardMetrics, RewardBreakdown, RewardInput
    from .step import StatisticsSnapshot, StepTransition
    from .traffic import ServiceRequest, TrafficRecord, TrafficTable

_EXPORTS: dict[str, tuple[str, str]] = {
    "ActionSelection": (".action_mask", "ActionSelection"),
    "Allocation": (".allocation", "Allocation"),
    "CandidateRewardMetrics": (".reward", "CandidateRewardMetrics"),
    "MaskMode": (".enums", "MaskMode"),
    "Modulation": (".modulation", "Modulation"),
    "ObservationSchema": (".observation", "ObservationSchema"),
    "ObservationSnapshot": (".observation", "ObservationSnapshot"),
    "QoTRequest": (".qot", "QoTRequest"),
    "QoTResult": (".qot", "QoTResult"),
    "RewardBreakdown": (".reward", "RewardBreakdown"),
    "RewardInput": (".reward", "RewardInput"),
    "RewardProfile": (".enums", "RewardProfile"),
    "ServiceQoTUpdate": (".qot", "ServiceQoTUpdate"),
    "ServiceRequest": (".traffic", "ServiceRequest"),
    "StatisticsSnapshot": (".step", "StatisticsSnapshot"),
    "Status": (".enums", "Status"),
    "StepTransition": (".step", "StepTransition"),
    "TrafficMode": (".enums", "TrafficMode"),
    "TrafficRecord": (".traffic", "TrafficRecord"),
    "TrafficTable": (".traffic", "TrafficTable"),
}

__all__ = [
    "ActionSelection",
    "Allocation",
    "CandidateRewardMetrics",
    "MaskMode",
    "Modulation",
    "ObservationSchema",
    "ObservationSnapshot",
    "QoTRequest",
    "QoTResult",
    "RewardBreakdown",
    "RewardInput",
    "RewardProfile",
    "ServiceQoTUpdate",
    "ServiceRequest",
    "StatisticsSnapshot",
    "Status",
    "StepTransition",
    "TrafficMode",
    "TrafficRecord",
    "TrafficTable",
]


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
