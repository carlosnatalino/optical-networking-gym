from __future__ import annotations

from importlib import import_module

_EXPORTS = {
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

__all__ = list(_EXPORTS)


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
