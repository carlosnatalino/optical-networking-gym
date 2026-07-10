from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "RequestAnalysis": (".request_analysis", "RequestAnalysis"),
    "RequestAnalysisEngine": (".request_analysis", "RequestAnalysisEngine"),
    "ScenarioConfig": (".scenario", "ScenarioConfig"),
    "Simulator": (".simulator", "Simulator"),
    "TrafficModel": (".traffic_model", "TrafficModel"),
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
