from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "benchmark_action_mask": (".benchmarking", "benchmark_action_mask"),
    "benchmark_allocation": (".benchmarking", "benchmark_allocation"),
    "benchmark_integrated_episode_vs_legacy": (
        ".integrated_benchmarking",
        "benchmark_integrated_episode_vs_legacy",
    ),
    "benchmark_observation": (".benchmarking", "benchmark_observation"),
    "benchmark_qot_engine": (".benchmarking", "benchmark_qot_engine"),
    "benchmark_request_analysis": (".benchmarking", "benchmark_request_analysis"),
    "benchmark_reward_function": (".benchmarking", "benchmark_reward_function"),
    "benchmark_runtime_state": (".benchmarking", "benchmark_runtime_state"),
    "benchmark_simulator_episode": (".integrated_benchmarking", "benchmark_simulator_episode"),
    "benchmark_statistics_step_info": (".benchmarking", "benchmark_statistics_step_info"),
    "compare_simulator_episode_with_legacy": (
        ".integrated_benchmarking",
        "compare_simulator_episode_with_legacy",
    ),
    "profile_simulator_episode": (".integrated_benchmarking", "profile_simulator_episode"),
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
